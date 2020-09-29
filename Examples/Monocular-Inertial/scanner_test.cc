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
    bool InitSLAM(int argc, char **argv);
    bool RunSLAM(int argc, char *argv[]);
    bool RunSLAMScanner(const std::string& mappoint_save_dir);
    bool SaveMappointFor3DObject(const std::string save_path);
    void SetBondingBox(std::vector<Eigen::Vector3d> boundingbox);
    std::vector<Eigen::Vector3d> GetBoundingbox();
private:
    void LoadImages(
        const string &strImagePath, const string &strPathTimes,
        vector<string> &vstrImages, vector<double> &vTimeStamps);
    void LoadIMU(
        const string &strImuPath, vector<double> &vTimeStamps,
        vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

    std::string m_result_dir;
    vector<vector<string>> vstrImageFilenames;
    vector<vector<double>> vTimestampsCam;
    vector<vector<cv::Point3f>> vAcc, vGyro;
    vector<vector<double>> vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    int num_seq;
    vector<int> first_imu;
    double ttrack_tot = 0;
    bool bFileName;
    vector<float> vTimesTrack;
    ORB_SLAM3::System *SLAM;

    // ar
    ORB_SLAM3::ViewerAR viewerAR;
    bool bRGB = true;
    cv::Mat K;
    cv::Mat DistCoef;
    std::vector<Eigen::Vector3d> m_boundingbox;

};

void TestViewer::SetBondingBox(std::vector<Eigen::Vector3d> boundingbox) {
    m_boundingbox.clear();
    m_boundingbox = boundingbox;
}

std::vector<Eigen::Vector3d> TestViewer::GetBoundingbox() {
    return m_boundingbox;
}

bool TestViewer::SaveMappointFor3DObject(const std::string save_path) {
    char *buffer = NULL;
    int buffer_size = 0;
    SLAM->SetBoundingbox(m_boundingbox);

    bool save_result = SLAM->PackAtlasToMemoryFor3DObject(&buffer, buffer_size);
    if(save_result) {
        std::ofstream out(save_path, std::ios::out | std::ios::binary);
        if (out.is_open()) {
            out.write(buffer, buffer_size);
            delete[] buffer;
            return true;
        } else {
            delete[] buffer;
            std::cout << "Error opening the pointCloud file!";
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
            /*mydata*/
            //vstrImages.push_back(strImagePath + "/" + ss.str());
            /*euroc data*/
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");

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
    first_imu.resize(num_seq);
    for (size_t i = 0; i < num_seq; i++) {
        first_imu[i] = 0;
    }

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

            /* mydata*/
            //vTimeStamps.push_back(data[0]);
            /*euroc*/
            vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
        }
    }
}

bool TestViewer::InitSLAM(int argc, char **argv) {
    num_seq = (argc - 3) / 2;
    cout << "num_seq = " << num_seq << endl;
    bFileName = (((argc - 3) % 2) == 1);
    string file_name;
    if (bFileName) {
        file_name = string(argv[argc - 1]);
        cout << "file name: " << file_name << endl;
    }

    // Load all sequences:
    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (int seq = 0; seq < num_seq; seq++) {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(2 * seq) + 3]);
        string pathTimeStamps(argv[(2 * seq) + 4]);

        /*my data
        string pathCam0 = pathSeq + "/camera/images";
        string pathImu = pathSeq + "/imu/data.csv";*/

        /* euroc data*/
        string pathCam0 = pathSeq + "/cam0/data";
        string pathImu = pathSeq + "/imu0/data.csv";

        LoadImages(
            pathCam0, pathTimeStamps, vstrImageFilenames[seq],
            vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if ((nImages[seq] <= 0) || (nImu[seq] <= 0)) {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq
                 << endl;
            return false;
        }

        // Find first imu to be considered, supposing imu measurements start
        // first

        while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }

    // Vector for tracking time statistics
    vTimesTrack.resize(tot_images);
    cout.precision(17);


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
    double fastInit = fsSettings["ORBextractor.iniThFAST"];
    double fastThreathold = fsSettings["ORBextractor.minThFAST"];

    Parameters::GetInstance().SetScaleFactor(scaleFactor);
    Parameters::GetInstance().SetLevels(nlevels);
    Parameters::GetInstance().SetFastInit(fastInit);
    Parameters::GetInstance().SetFastThreathold(fastThreathold);

    SLAM = new ORB_SLAM3::System(
        argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, false, false);

    cout << endl << endl;
    cout << "-----------------------" << endl;
    cout << "Augmented Reality Demo" << endl;
    cout << "1) Translate the camera to initialize SLAM." << endl;
    cout << "2) Look at a planar region and translate the camera." << endl;
    cout << "3) Press Insert Cube to place a virtual cube in the plane. " << endl;
    cout << endl;
    cout << "You can place several cubes in different planes." << endl;
    cout << "-----------------------" << endl;
    cout << endl;

    return true;
}

bool TestViewer::RunSLAMScanner(const std::string& mappoint_save_dir) {

    // std::string mappoint_save_dir = "/home/zhangye/data/ObjectRecognition/shoe.bin";
    // click boundingbox fix button:

    //1. get the boundingbox
    std::vector<Eigen::Vector3d> boundingbox;
    //2. setboundingbox
    ORB_SLAM3::FrameObjectProcess::GetInstance()->SetBoundingBox(boundingbox);

    //3. run slam and extract more orb features

    //4. save mappoint

    if(SaveMappointFor3DObject(mappoint_save_dir)) {
        std::cout << "save mappoint for 3dobject success!";
    }
}

bool TestViewer::RunSLAM(int argc, char *argv[]) {
    viewerAR.SetSLAM(SLAM);
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    bRGB = static_cast<bool>((int)fSettings["Camera.RGB"]);
    float fps = fSettings["Camera.fps"];
    viewerAR.SetFPS(fps);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    viewerAR.SetCameraCalibration(fx,fy,cx,cy);

    K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;

    DistCoef = cv::Mat::zeros(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    thread tViewer = thread(&ORB_SLAM3::ViewerAR::Run,&viewerAR);

    int proccIm = 0;
    for (int seq = 0; seq < num_seq; seq++) {
        // Main loop
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        //nImages[seq] = 20;
        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {
            if(viewerAR.GetFixFlag()) {
                // get boundingbox in slam word coords
                std::vector<Eigen::Vector3d> boundingbox_slam_coords = viewerAR.GetBoundingbox();
                if(boundingbox_slam_coords.empty()) {
                    LOG(FATAL) << "error in save boundingbox";
                }


                // obstract more keypoihts
                ORB_SLAM3::FrameObjectProcess::GetInstance()->SetBoundingBox(boundingbox_slam_coords);

                SetBondingBox(boundingbox_slam_coords);
            }

            if(viewerAR.GetStopFlag()) {
                break;
            }
            while (true) {
                if (!viewerAR.GetDebugFlag()) {
                    break;
                }
                usleep(1 * 1e6); // 1e6
            }

            // Read image from file
            im = cv::imread(
                vstrImageFilenames[seq][ni], CV_LOAD_IMAGE_UNCHANGED);

            double tframe = vTimestampsCam[seq][ni];

            if (im.empty()) {
                cerr << endl
                     << "Failed to load image at: "
                     << vstrImageFilenames[seq][ni] << endl;
                return false;
            }

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if (ni > 0) {
                // cout << "t_cam " << tframe << endl;

                while (vTimestampsImu[seq][first_imu[seq]] <=
                       vTimestampsCam[seq][ni]) {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                        vAcc[seq][first_imu[seq]].x,
                        vAcc[seq][first_imu[seq]].y,
                        vAcc[seq][first_imu[seq]].z,
                        vGyro[seq][first_imu[seq]].x,
                        vGyro[seq][first_imu[seq]].y,
                        vGyro[seq][first_imu[seq]].z,
                        vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
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
            vector<ORB_SLAM3::MapPoint*> vMPs = SLAM->GetTrackedMapPoints();
            vector<cv::KeyPoint> vKeys = SLAM->GetTrackedKeyPointsUn();
            cv::undistort(im_clone,imu,K,DistCoef);
            if(bRGB)
                viewerAR.SetImagePose(imu,Tcw,state,vKeys,vMPs);
            else
            {
                cv::cvtColor(imu,imu,CV_RGB2BGR);
                viewerAR.SetImagePose(imu,Tcw,state,vKeys,vMPs);
            }


#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 =
                std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 =
                std::chrono::monotonic_clock::now();
#endif

            double ttrack =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    t2 - t1)
                    .count();
            ttrack_tot += ttrack;

            vTimesTrack[ni] = ttrack;

            // Wait to load the next frame
            double T = 0;
            if (ni < nImages[seq] - 1)
                T = vTimestampsCam[seq][ni + 1] - tframe;
            else if (ni > 0)
                T = tframe - vTimestampsCam[seq][ni - 1];

            if (ttrack < T)
                usleep((T - ttrack) * 1e6); // 1e6
        }
        if (seq < num_seq - 1) {
            cout << "Changing the dataset" << endl;
            SLAM->ChangeDataset();
        }
    }


    // Stop all threads
    SLAM->Shutdown();

    std::string mappoint_save_path = "/home/zhangye/data/ObjectRecognition/shoe.bin";
    if(SaveMappointFor3DObject(mappoint_save_path)) {
        std::cout << "save mappoint for 3dobject success!";
    }

    // Save camera trajectory
    if (bFileName) {
        const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
        const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
        SLAM->SaveTrajectoryEuRoC(f_file);
        SLAM->SaveKeyFrameTrajectoryEuRoC(kf_file);
    } else {
        SLAM->SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    tViewer.join();
    return true;
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        cerr << endl
             << "Usage: ./mono_inertial_euroc path_to_vocabulary "
                "path_to_settings path_to_sequence_folder_1 "
                "path_to_times_file_1 (path_to_image_folder_2 "
                "path_to_times_file_2 ... path_to_image_folder_N "
                "path_to_times_file_N) "
             << endl;
        return 1;
    }
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO,"./myInfo");
    TestViewer testViewer;

    bool initial_slam_result = testViewer.InitSLAM(argc, argv);

    if(!initial_slam_result) {
        std::cout << "slam initialize fail!" << std::endl;
        return 0;
    }

    testViewer.RunSLAM(argc, argv);



    return 0;
}
