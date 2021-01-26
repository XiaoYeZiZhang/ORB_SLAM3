#include <iostream>
#include <algorithm>
#include <chrono>
#include <opencv2/core/core.hpp>
#include "GlobalSummary.h"
#include "ORBSLAM3/System.h"
#include "FileIO.h"
#include "ImuTypes.h"
#include "Utility/Camera.h"
#include "Statistics.h"
#include "ObjectRecognitionThread.h"
#include "FrameObjectProcess.h"
#include "mode.h"

using namespace std;
class TestViewer {
public:
    TestViewer() {
    }
    ~TestViewer() {
        delete SLAM;
    }
    bool InitSLAM();
    bool InitObjectRecognition();
    bool RunObjectRecognition();

    cv::Mat M2l;
    cv::Mat M1r;
    cv::Mat M2r;
    cv::Mat M1l;

    // yaml
    std::string m_voc_path;
    std::string m_voc_path_superpoint;
    std::string m_data_path;
    std::string m_config_path;
    std::string m_slam_saved_path;
    std::string m_mappoint_filename;
    std::string m_mappoint_filename_superpoint;
    std::string m_dataset_name;

private:
    void LoadIMU(
        const string &strImuPath, vector<double> &vTimeStamps,
        vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

    void LoadImages(
        const string &strPathLeft, const string &strPathTimes,
        vector<string> &vstrImages, vector<double> &vTimeStamps);

    vector<cv::Point3f> vAcc, vGyro;
    vector<double> vTimestampsImu;
    vector<string> vstrImages;
    vector<double> vTimestampsCam;
    int nImages;
    int nImu;
    int first_imu = 0;
    bool mRGB;
    ORB_SLAM3::System *SLAM;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    // 3d object
    std::shared_ptr<ObjRecognition::Object> m_pointCloud =
        std::make_shared<ObjRecognition::Object>(0);
    ObjRecognition::ObjRecogResult m_objrecog_result;

    std::shared_ptr<ObjRecognition::ObjRecogThread> m_objrecog_thread =
        std::make_shared<ObjRecognition::ObjRecogThread>();
    std::shared_ptr<DBoW3::Vocabulary> m_voc =
        std::make_shared<DBoW3::Vocabulary>();
};

void TestViewer::LoadIMU(
    const string &strImuPath, vector<double> &vTimeStamps,
    vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro) {
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

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

            vTimeStamps.push_back(data[0] / 1e9);
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
        }
    }
}

bool TestViewer::InitObjectRecognition() {
    m_objrecog_thread->SetVocabulary(m_voc);
    m_objrecog_thread->Init();
    if (!m_objrecog_thread->StartThread()) {
        VLOG(0) << "create objRecognition thread failed";
        return -1;
    }

    // set slam data callback
#ifdef SUPERPOINT
    std::string cloud_point_model_dir =
        m_slam_saved_path + "/" + m_mappoint_filename_superpoint;
#else
    std::string cloud_point_model_dir =
        m_slam_saved_path + "/" + m_mappoint_filename;
#endif

    VLOG(0) << "Load Vocabulary Start";
#ifdef SUPERPOINT
#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#else
    if (!m_voc.get()) {
        LOG(ERROR) << "vocabulary load fail!";
    }
    m_voc.get()->load(m_voc_path_superpoint);

#endif
#else
    if (!m_voc.get()) {
        LOG(ERROR) << "vocabulary load fail!";
    }
    m_voc.get()->load(m_voc_path);
#endif
    VLOG(0) << "Load Vocabulary Done!";

    int model_id = 0;
    char *cloud_point_model_buffer = nullptr;
    long long cloud_point_model_buf_size = 0;
    ReadPointCloudModelToBuffer(
        cloud_point_model_dir, &cloud_point_model_buffer,
        cloud_point_model_buf_size);
    m_pointCloud = std::make_shared<ObjRecognition::Object>(model_id);
    if (!m_pointCloud->LoadPointCloud(
            cloud_point_model_buf_size, cloud_point_model_buffer)) {
        LOG(ERROR) << "Load PointCloud failed, not set model";
        return -1;
    }
#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#else
    m_pointCloud->SetVocabulary(m_voc);
#endif
    m_objrecog_thread->SetModel(m_pointCloud);

    STATISTICS_UTILITY::StatsCollector pointCloudNum("Mappoint num");
    pointCloudNum.AddSample(m_pointCloud->GetPointCloudsNum());

    delete[] cloud_point_model_buffer;
    SLAM->SetPointCloudModel(m_pointCloud);
    SLAM->mpViewer->SetPointCloudModel(m_pointCloud);
    SLAM->mpViewer->SetThreadHandler(m_objrecog_thread);
    ObjRecognition::ObjRecongManager::Instance().SetThreadHandler(
        m_objrecog_thread);

    return true;
}

void TestViewer::LoadImages(
    const string &strPathLeft, const string &strPathTimes,
    vector<string> &vstrImages, vector<double> &vTimeStamps) {
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    if (!fTimes.is_open()) {
        LOG(FATAL) << "error open timestamp.txt";
    }
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strPathLeft + "/" + ss.str());
            double t;
            ss >> t;
            vTimeStamps.push_back(t / 1e9);
        }
    }
}

bool TestViewer::InitSLAM() {

    VLOG(0) << "Loading images ...";
    string pathTimeStamps = m_data_path + "/cam0/timestamp.txt";
    string pathCam0 = m_data_path + "/cam0/data";
    string pathImu = m_data_path + "/imu0/data.csv";

    LoadImages(pathCam0, pathTimeStamps, vstrImages, vTimestampsCam);
    VLOG(0) << "LOADED!";
    nImages = vstrImages.size();

    VLOG(0) << "Loading IMU ...";
    LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro);
    VLOG(0) << "LOADED!" << endl;
    nImu = vTimestampsImu.size();

    if ((nImages <= 0) || (nImu <= 0)) {
        cerr << "ERROR: Failed to load images or IMU" << endl;
        return 1;
    }

    while (vTimestampsImu[first_imu] <= vTimestampsCam[0])
        first_imu++;
    first_imu--;

    cv::FileStorage fsSettings(m_config_path, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    double fx = fsSettings["Camera.fx"];
    double fy = fsSettings["Camera.fy"];
    double cx = fsSettings["Camera.cx"];
    double cy = fsSettings["Camera.cy"];
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];

    ObjRecognition::CameraIntrinsic::GetInstance().SetParameters(
        fx, fy, cx, cy, width, height);

    double scaleFactor = fsSettings["ORBextractor.scaleFactor"];
    int nlevels = fsSettings["ORBextractor.nLevels"];
    int fastInitThreshold = fsSettings["ORBextractor.iniThFAST"];
    int fastMinThreshold = fsSettings["ORBextractor.minThFAST"];

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

    mRGB = static_cast<bool>((int)fsSettings["Camera.RGB"]);
    bool is_recognition = false;

#ifdef OBJECTRECOGNITION
    is_recognition = true;
#endif
    SLAM = new ORB_SLAM3::System(
        m_voc_path, m_config_path, ORB_SLAM3::System::IMU_MONOCULAR, true,
        is_recognition);

    return true;
}

bool TestViewer::RunObjectRecognition() {
    cv::Mat im;
    int proccIm = 0;
    for (int ni = 0; ni < nImages; ni++, proccIm++) {
        if (SLAM->mpViewer->GetIsStopFlag()) {
            break;
        }
        im = cv::imread(vstrImages[ni], cv::IMREAD_UNCHANGED);

        if (im.empty()) {
            cerr << endl
                 << "Failed to load image at: " << string(vstrImages[ni])
                 << endl;
            return 1;
        }

        double tframe = vTimestampsCam[ni];

        // Load imu measurements from previous frame
        vImuMeas.clear();

        if (ni > 0)
            while (vTimestampsImu[first_imu] <= vTimestampsCam[ni]) {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                    vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                    vTimestampsImu[first_imu]));
                first_imu++;
            }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 =
            std::chrono::monotonic_clock::now();
#endif

        cv::Mat camPos = SLAM->TrackMonocular(im, tframe, vImuMeas);

        cv::Mat im_clone = im.clone();
        int slam_state = SLAM->GetTrackingState();

        if (mRGB)
            SLAM->mpViewer->SetSLAMInfo(im_clone, slam_state, ni, camPos);
        else {
            cv::cvtColor(im_clone, im_clone, CV_RGB2BGR);
            SLAM->mpViewer->SetSLAMInfo(im_clone, slam_state, ni, camPos);
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

        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6); // 1e6
    }
#ifdef OBJECTRECOGNITION
    std::string statics_result_filename;
#ifdef SUPERPOINT
    statics_result_filename = "statics_result_SUPERPOINT.txt";
#else
    statics_result_filename = "statics_result_ORB.txt";
#endif
    ObjRecognition::GlobalSummary::SaveStatics(
        m_slam_saved_path, STATISTICS_UTILITY::Statistics::Print(),
        statics_result_filename);

    m_objrecog_thread->RequestReset();
    m_objrecog_thread->WaitEndReset();
    m_objrecog_thread->StartThread();
    m_objrecog_thread->RequestStop();
    m_objrecog_thread->WaitEndStop();
#endif

    SLAM->Shutdown();
    return true;
}

int main(int argc, char *argv[]) {
#ifdef OBJECTRECOGNITION
#else
    LOG(FATAL) << "not in the object detection and tracking mode";
#endif
#ifdef MONO
#else
    LOG(FATAL) << "not in the mono mode";
#endif
    if (argc < 1) {
        cerr << endl << "Usage: ./ path_to_yaml " << endl;
        return 1;
    }

    // vlog setting
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    TestViewer testViewer;

    cv::FileStorage fsSettings(argv[1], cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    fsSettings["voc_path_superpoint"] >> testViewer.m_voc_path_superpoint;
    fsSettings["voc_path"] >> testViewer.m_voc_path;
    fsSettings["data_path_objRecognition"] >> testViewer.m_data_path;
    fsSettings["config_path"] >> testViewer.m_config_path;
    fsSettings["saved_path"] >> testViewer.m_slam_saved_path;
    fsSettings["mappoint_filename"] >> testViewer.m_mappoint_filename;
    fsSettings["mappoint_filename_superpoint"] >>
        testViewer.m_mappoint_filename_superpoint;
    fsSettings["dataset_name"] >> testViewer.m_dataset_name;
    bool initial_slam_result = testViewer.InitSLAM();

    if (!initial_slam_result) {
        LOG(FATAL) << "slam initialize fail!";
        return 0;
    }

#ifdef OBJECTRECOGNITION
    bool initialize_objectRecognition_result =
        testViewer.InitObjectRecognition();
    if (!initialize_objectRecognition_result) {
        LOG(FATAL) << "objectRecognition initialize fail!";
        return 0;
    }
#endif

    testViewer.RunObjectRecognition();
    return 0;
}
