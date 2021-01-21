#include <iostream>
#include <algorithm>
#include <chrono>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <GlobalSummary.h>
#include "System.h"
#include "FileIO.h"
#include "Camera.h"
#include "Statistics.h"
#include "ObjectRecognitionManager.h"
#include "FrameObjectProcess.h"
#include "mode.h"

using namespace std;
using namespace std::chrono;
class TestViewer {
public:
    bool InitSLAM();
    bool InitObjectRecognition();
    bool RunObjectRecognition();

    cv::Mat M2l;
    cv::Mat M1r;
    cv::Mat M2r;
    cv::Mat M1l;
    // yaml
    std::string voc_path;
    std::string voc_path_superpoint;
    std::string data_path;
    std::string config_path;
    std::string slam_saved_path;
    std::string mappoint_filename;
    std::string mappoint_filename_superpoint;
    std::string dataset_name;

private:
    void LoadImages(
        const string &strPath, const string &strPathTimes,
        vector<string> &vstrImages, vector<double> &vTimeStamps);

    void ObjectResultParse(const ObjRecognition::ObjRecogResult &result);

    vector<string> vstrImages;
    vector<string> vstrImageColor;
    vector<double> vTimestampsCam;
    int nImages;
    bool mRGB;
    ORB_SLAM3::System *SLAM;

    // 3d object
    std::shared_ptr<ObjRecognition::Object> m_pointCloud =
        std::make_shared<ObjRecognition::Object>(0);
    ObjRecognition::ObjRecogResult m_objrecog_result;
    Eigen::Matrix<double, 3, 3> m_Row = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 1> m_tow = Eigen::Matrix<double, 3, 1>::Zero();
};

bool TestViewer::InitObjectRecognition() {
    ObjRecognitionExd::ObjRecongManager::Instance().CreateWithConfig();

#ifdef SUPERPOINT
    std::string cloud_point_model_dir =
        slam_saved_path + "/" + mappoint_filename_superpoint;
#else
    std::string cloud_point_model_dir =
        slam_saved_path + "/" + mappoint_filename;
#endif

    VLOG(0) << "Load Vocabulary Start";
#ifdef SUPERPOINT
#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#else
    bool voc_load_res = ObjRecognitionExd::ObjRecongManager::Instance().LoadVoc(
        voc_path_superpoint);
#endif
#else
    bool voc_load_res =
        ObjRecognitionExd::ObjRecongManager::Instance().LoadVoc(voc_path);
#endif
    VLOG(0) << "Load Vocabulary Done!";

#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#else
    if (!voc_load_res) {
        LOG(ERROR) << "vocabulary load fail!";
    }
#endif

    int model_id = 0;
    char *cloud_point_model_buffer = nullptr;
    long long cloud_point_model_buf_size = 0;
    ReadPointCloudModelToBuffer(
        cloud_point_model_dir, &cloud_point_model_buffer,
        cloud_point_model_buf_size);
    ObjRecognitionExd::ObjRecongManager::Instance().LoadModel(
        model_id, cloud_point_model_buffer, cloud_point_model_buf_size,
        m_pointCloud);

    STATISTICS_UTILITY::StatsCollector pointCloudNum("Mappoint num");
    pointCloudNum.AddSample(m_pointCloud->GetPointCloudsNum());

    delete[] cloud_point_model_buffer;
    SLAM->SetPointCloudModel(m_pointCloud);
    SLAM->mpViewer->SetPointCloudModel(m_pointCloud);
    return true;
}

void TestViewer::LoadImages(
    const string &strPath, const string &strPathTimes,
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
            vstrImages.push_back(strPath + "/" + ss.str());
            double t;
            ss >> t;
            vTimeStamps.push_back(t / 1e9);
        }
    }
}

bool TestViewer::InitSLAM() {

    VLOG(0) << "Loading images ...";
    string pathTimeStamps = data_path + "/cam0/timestamp.txt";
    string pathCam0 = data_path + "/cam0/data";

    LoadImages(pathCam0, pathTimeStamps, vstrImages, vTimestampsCam);
    VLOG(0) << "LOADED!";
    nImages = vstrImages.size();

    cv::FileStorage fsSettings(config_path, cv::FileStorage::READ);
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
        voc_path, config_path, ORB_SLAM3::System::MONOCULAR, true,
        is_recognition);

    return true;
}

void TestViewer::ObjectResultParse(
    const ObjRecognition::ObjRecogResult &result) {
    m_objrecog_result = result;
    Eigen::Matrix<float, 3, 3> Rcw;
    Rcw.row(0) =
        Eigen::Matrix<float, 3, 1>::Map(m_objrecog_result.R_camera[0], 3);
    Rcw.row(1) =
        Eigen::Matrix<float, 3, 1>::Map(m_objrecog_result.R_camera[1], 3);
    Rcw.row(2) =
        Eigen::Matrix<float, 3, 1>::Map(m_objrecog_result.R_camera[2], 3);
    Eigen::Vector3f tcw = Eigen::Vector3f::Map(m_objrecog_result.t_camera, 3);
    Eigen::Matrix<float, 3, 3> Rwo;
    Rwo.col(0) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[0], 3);
    Rwo.col(1) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[3], 3);
    Rwo.col(2) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[6], 3);
    Eigen::Vector3f two =
        Eigen::Vector3f::Map(&m_objrecog_result.t_obj_buffer[0], 3);

    Eigen::Matrix3f Rco = Eigen::Matrix3f::Identity();
    Rco = Rcw * Rwo;
    Rwo = Rcw.transpose() * Rco;
    Eigen::Matrix3f Row = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tow = Eigen::Vector3f::Zero();
    Row = Rwo.transpose();
    tow = -Row * two;

    if (result.num == 1) {
        m_Row = Row.cast<double>(); // world -> obj
        m_tow = tow.cast<double>();
    } else {
        m_Row = Eigen::Matrix3d::Identity();
        m_tow = Eigen::Vector3d::Zero();
    }

    SLAM->mpViewer->SetObjectRecognitionPose(m_Row, m_tow);
}

bool TestViewer::RunObjectRecognition() {
    cv::Mat im;
    cv::Mat imColor, imColorRect;
    int proccIm = 0;

    auto start_time = high_resolution_clock::now();
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

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 =
            std::chrono::monotonic_clock::now();
#endif

        cv::Mat camPos = SLAM->TrackMonocular(im, tframe);

        cv::Mat im_clone = im.clone();
        int slam_state = SLAM->GetTrackingState();
        SLAM->mpViewer->SetSLAMInfo(im_clone, slam_state, ni, camPos);

#ifdef OBJECTRECOGNITION
        ObjectResultParse(ObjRecognitionExd::ObjRecongManager::Instance()
                              .GetObjRecognitionResult());
#endif

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
    auto end =
        (duration_cast<microseconds>(high_resolution_clock::now() - start_time))
            .count() /
        1000.0;

    std::cout << "time total: " << end << std::endl;
    std::cout << "finish!" << std::endl;

#ifdef OBJECTRECOGNITION
    std::string statics_result_filename;
#ifdef SUPERPOINT
    statics_result_filename =
        mappoint_filename_superpoint + "_result_SUPERPOINT.txt";
#else
    statics_result_filename = mappoint_filename + "_result_ORB.txt";
#endif
    ObjRecognition::GlobalSummary::SaveStatics(
        slam_saved_path, STATISTICS_UTILITY::Statistics::Print(),
        statics_result_filename);

    ObjRecognitionExd::ObjRecongManager::Instance().Destroy();
#endif

    SLAM->Shutdown();
    return true;
}

int main(int argc, char *argv[]) {
#ifdef OBJECTRECOGNITION
#else
    LOG(FATAL) << "not in the object detection and tracking mode";
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

    fsSettings["voc_path_superpoint"] >> testViewer.voc_path_superpoint;
    fsSettings["voc_path"] >> testViewer.voc_path;
    fsSettings["data_path_objRecognition"] >> testViewer.data_path;
    fsSettings["config_path"] >> testViewer.config_path;
    fsSettings["saved_path"] >> testViewer.slam_saved_path;
    fsSettings["mappoint_filename"] >> testViewer.mappoint_filename;
    fsSettings["mappoint_filename_superpoint"] >>
        testViewer.mappoint_filename_superpoint;
    fsSettings["dataset_name"] >> testViewer.dataset_name;
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
