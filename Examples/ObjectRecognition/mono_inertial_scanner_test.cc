//
// Created by zhangye on 2020/9/21.
//
#include <ctime>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "ORBSLAM3/System.h"
#include "Utility/GlobalSummary.h"
#include "Utility/FileIO.h"
#include "ORBSLAM3/ImuTypes.h"
#include "Utility/Camera.h"
#include "ObjectRecognitionSystem/ObjectRecognitionManager.h"
#include "ORBSLAM3/FrameObjectProcess.h"
#include "ORBSLAM3/ViewerAR.h"
#include "mode.h"

using namespace std;
class TestViewer {
public:
    bool InitSLAM(char **argv);
    bool RunScanner(char **argv);
    bool SaveMappointFor3DObject(const std::string save_path);

private:
    void LoadImages(
        const string &strImagePath, const string &strPathTimes,
        vector<string> &vstrImages, vector<double> &vTimeStamps);
    void LoadIMU(
        const string &strImuPath, vector<double> &vTimeStamps,
        vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

    void ScanDebugMode();
    std::string m_result_dir;
    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    vector<cv::Point3f> vAcc, vGyro;
    vector<double> vTimestampsImu;
    int nImages;
    int nImu;
    int first_imu;
    ORB_SLAM3::System *SLAM;

    // ar
    ORB_SLAM3::ViewerAR viewerAR;
    bool bRGB = true;
    cv::Mat K;
    cv::Mat DistCoef;
    std::vector<Eigen::Vector3d> m_boundingbox_w;
};

bool TestViewer::SaveMappointFor3DObject(const std::string save_path) {
    char *buffer = NULL;
    int buffer_size = 0;
    SLAM->SetScanBoundingbox_W(m_boundingbox_w);

    bool save_result = SLAM->PackAtlasToMemoryFor3DObject(&buffer, buffer_size);
    if (save_result) {
        std::ofstream out(save_path, std::ios::out | std::ios::binary);
        if (out.is_open()) {
            out.write(buffer, buffer_size);
            delete[] buffer;
            return true;
        } else {
            delete[] buffer;
            LOG(FATAL) << "Error opening the pointCloud file!";
            return false;
        }
    }
    delete[] buffer;
    return false;
}

void TestViewer::LoadImages(
    const string &strImagePath, const string &strPathTimes,
    vector<string> &vstrImages, vector<double> &vTimeStamps) {
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            /*euroc data*/
            // vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            /*realsense data*/
            vstrImages.push_back(strImagePath + "/" + ss.str());
            double t;
            ss >> t;
            vTimeStamps.push_back(t / 1e9);
        }
    }
}

void TestViewer::LoadIMU(
    const string &strImuPath, vector<double> &vTimeStamps,
    vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro) {
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);
    first_imu = 0;

    while (!fImu.eof()) {
        string s;
        getline(fImu, s);
        if (s[0] == '#')
            continue;

        if (!s.empty()) {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            /*euroc data*/
            vTimeStamps.push_back(data[0] / 1e9);
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
        }
    }
}

bool TestViewer::InitSLAM(char **argv) {
    // Load all sequences:
    int tot_images = 0;

    VLOG(0) << "Loading images "
            << "...";

    string pathSeq(argv[3]);
    string pathTimeStamps(argv[4]);

    /* euroc data*/
    string pathCam0 = pathSeq + "/cam0/data";
    string pathImu = pathSeq + "/imu0/data.csv";

    LoadImages(pathCam0, pathTimeStamps, vstrImageFilenames, vTimestampsCam);
    VLOG(0) << "LOADED!";

    VLOG(0) << "Loading IMU "
            << "...";
    LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro);
    VLOG(0) << "LOADED!";

    nImages = vstrImageFilenames.size();
    tot_images += nImages;
    nImu = vTimestampsImu.size();

    if ((nImages <= 0) || (nImu <= 0)) {
        cerr << "ERROR: Failed to load images or IMU" << endl;
        return false;
    }

    // Find first imu to be considered, supposing imu measurements start
    // first

    while (vTimestampsImu[first_imu] <= vTimestampsCam[0])
        first_imu++;
    first_imu--; // first imu measurement to be considered

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    double fx = fsSettings["Camera.fx"];
    double fy = fsSettings["Camera.fy"];
    double cx = fsSettings["Camera.cx"];
    double cy = fsSettings["Camera.cy"];
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];

    ObjRecognition::CameraIntrinsic::GetInstance().SetParameters(
        fx, fy, cx, cy, width, height);

    double scaleFactor = fsSettings["ORBextractor.scaleFactor"];
    double nlevels = fsSettings["ORBextractor.nLevels"];
    double fastInitThreshold = fsSettings["ORBextractor.iniThFAST"];
    double fastMinThreshold = fsSettings["ORBextractor.minThFAST"];

    double SP_scaleFactor = fsSettings["SPextractor.scaleFactor"];
    int SP_nLevels = fsSettings["SPextractor.nLevels"];
    int SP_nFeatures = fsSettings["SPextractor.nFeatures"];

    int ObjRecognition_ORB_nFeatures =
        fsSettings["ORBextractor.objRecognitionFeatures"];

    Parameters::GetInstance().SetScaleFactor(scaleFactor);
    Parameters::GetInstance().SetLevels(nlevels);
    Parameters::GetInstance().SetFastInitThreshold(fastInitThreshold);
    Parameters::GetInstance().SetFastMinThreshold(fastMinThreshold);

    Parameters::GetInstance().SetSPScaleFactor(SP_scaleFactor);
    Parameters::GetInstance().SetSPFeatures(SP_nFeatures);
    Parameters::GetInstance().SetSPLevels(SP_nLevels);

    Parameters::GetInstance().SetObjRecognitionORBFeatures(
        ObjRecognition_ORB_nFeatures);

    SLAM = new ORB_SLAM3::System(
        argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, false, false);

    /*cout << endl << endl;
    cout << "-----------------------" << endl;
    cout << "Augmented Reality Demo" << endl;
    cout << "1) Translate the camera to initialize SLAM." << endl;
    cout << "2) Look at a planar region and translate the camera." << endl;
    cout << "3) Press Insert Cube to place a virtual cube in the plane. " <<
    endl; cout << endl; cout << "You can place several cubes in different
    planes." << endl; cout << "-----------------------" << endl; cout << endl;*/

    return true;
}

void TestViewer::ScanDebugMode() {
    while (true) {
        if (!viewerAR.GetScanDebugFlag()) {
            return;
        }
        usleep(1 * 1e5); // 1e6
        if (viewerAR.GetStopFlag()) {
            return;
        }
        if (viewerAR.GetFixFlag()) {
            // get boundingbox in slam word coords
            m_boundingbox_w = viewerAR.GetScanBoundingbox_W();
            if (m_boundingbox_w.empty()) {
                LOG(FATAL) << "error in save boundingbox";
            }

            VLOG(0) << "boundingbox coords: \n";
            for (const auto &coords : m_boundingbox_w) {
                VLOG(0) << coords;
            }
            // obstract more keypoihts
            ORB_SLAM3::FrameObjectProcess::GetInstance()->SetBoundingBox(
                m_boundingbox_w);
            viewerAR.SetFixFlag(false);
        }
    }
}

// click boundingbox fix button:
// 1. get the boundingbox
// 2. setboundingbox
// 3. run slam and extract more orb features
// 4. save mappoint
bool TestViewer::RunScanner(char **argv) {
    viewerAR.SetSLAM(SLAM);
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    bRGB = static_cast<bool>((int)fSettings["Camera.RGB"]);
    float fps = fSettings["Camera.fps"];
    viewerAR.SetFPS(fps);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    viewerAR.SetCameraCalibration(fx, fy, cx, cy);

    K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;

    DistCoef = cv::Mat::zeros(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    thread tViewer = thread(&ORB_SLAM3::ViewerAR::Run, &viewerAR);

    // Main loop
    cv::Mat im;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int proccIm = 0;
    // nImages = 20;
    for (int ni = 0; ni < nImages; ni++, proccIm++) {
        if (viewerAR.GetFixFlag()) {
            // get boundingbox in slam word coords
            m_boundingbox_w = viewerAR.GetScanBoundingbox_W();
            if (m_boundingbox_w.empty()) {
                LOG(FATAL) << "error in save boundingbox";
            }

            VLOG(0) << "boundingbox coords: \n";
            for (const auto &coords : m_boundingbox_w) {
                VLOG(0) << coords;
            }
            // obstract more keypoihts
            ORB_SLAM3::FrameObjectProcess::GetInstance()->SetBoundingBox(
                m_boundingbox_w);
            viewerAR.SetFixFlag(false);
        }
        ScanDebugMode();
        if (viewerAR.GetStopFlag()) {
            break;
        }

        // Read image from file
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);

        double tframe = vTimestampsCam[ni];

        if (im.empty()) {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni]
                 << endl;
            return false;
        }

        // Load imu measurements from previous frame
        vImuMeas.clear();

        if (ni > 0) {
            // cout << "t_cam " << tframe << endl;

            while (vTimestampsImu[first_imu] <= vTimestampsCam[ni]) {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                    vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                    vTimestampsImu[first_imu]));
                first_imu++;
            }
        }

        /*cout << "first imu: " << first_imu << endl;
        cout << "first imu time: " << fixed << vTimestampsImu[first_imu] <<
        endl; cout << "size vImu: " << vImuMeas.size() << endl;*/
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 =
            std::chrono::monotonic_clock::now();
#endif

        cv::Mat Tcw = SLAM->TrackMonocular(
            im, tframe, vImuMeas); // TODO change to monocular_inertial

        cv::Mat im_clone = im.clone();
        cv::Mat imu;
        int state = SLAM->GetTrackingState();
        vector<ORB_SLAM3::MapPoint *> vMPs = SLAM->GetTrackedMapPoints();
        vector<cv::KeyPoint> vKeys = SLAM->GetTrackedKeyPointsUn();
        cv::undistort(im_clone, imu, K, DistCoef);
        if (bRGB)
            viewerAR.SetImagePose(imu, Tcw, state, vKeys, vMPs);
        else {
            cv::cvtColor(imu, imu, CV_RGB2BGR);
            viewerAR.SetImagePose(imu, Tcw, state, vKeys, vMPs);
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 =
            std::chrono::monotonic_clock::now();
#endif

        double ttrack =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
                .count();

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6); // 1e6
    }

    // Stop all threads
    SLAM->Shutdown();

    std::string mappoint_save_path = argv[5];
    if (SaveMappointFor3DObject(mappoint_save_path)) {
        VLOG(0) << "save mappoint for 3dobject success!";
    }

    // Save camera trajectory
    const string kf_file = "kf_" + string(argv[6]) + ".txt";
    const string f_file = "f_" + string(argv[6]) + ".txt";
    SLAM->SaveTrajectoryEuRoC(f_file);
    SLAM->SaveKeyFrameTrajectoryEuRoC(kf_file);

    tViewer.join();
    return true;
}

int main(int argc, char *argv[]) {

    if (argc < 6) {
        cerr << endl
             << "Usage: ./mono_inertial_euroc path_to_vocabulary "
                "path_to_settings path_to_sequence_folder_1 "
                "path_to_times_file_1 (path_to_image_folder_2 "
                "path_to_times_file_2 ... path_to_image_folder_N "
                "path_to_times_file_N) "
             << endl;
        return 1;
    }

    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    TestViewer testViewer;

    bool initial_slam_result = testViewer.InitSLAM(argv);

    if (!initial_slam_result) {
        LOG(FATAL) << "slam initialize fail!";
        return 0;
    }

    testViewer.RunScanner(argv);
    return 0;
}
