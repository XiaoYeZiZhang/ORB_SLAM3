//
// Created by root on 2020/10/13.
//
#include <iostream>
#include <algorithm>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <include/ObjectRecognition/Utility/GlobalSummary.h>
#include "ORBSLAM3/System.h"
#include "Utility/FileIO.h"
#include "ORBSLAM3/ImuTypes.h"
#include "Utility/Camera.h"
#include "ObjectRecognitionSystem/ObjectRecognitionManager.h"
#include "ORBSLAM3/FrameObjectProcess.h"
#include "mode.h"

using namespace std;
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
    std::string dataset_name;

private:
    void LoadIMU(
        const string &strImuPath, vector<double> &vTimeStamps,
        vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

    void LoadImages(
        const string &strPathLeft, const string &strPathRight,
        const string &strPathTimes, vector<string> &vstrImageLeft,
        vector<string> &vstrImageRight, vector<double> &vTimeStamps);
    bool SaveResultInit();
    void ObjectResultParse(const ObjRecognition::ObjRecogResult &result);
    void SaveObjRecogResult();

    vector<cv::Point3f> vAcc, vGyro;
    vector<double> vTimestampsImu;
    std::string m_result_dir;
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
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
    std::string camera_pose_result_file_;
    std::string object_pose_result_file_;
    std::ofstream camera_pose_result_stream_;
    std::ofstream object_pose_result_stream_;
    ObjRecognition::ObjRecogResult m_objrecog_result;
    Eigen::Matrix<double, 3, 3> m_Row = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 1> m_tow = Eigen::Matrix<double, 3, 1>::Zero();
    std::string objrecog_info_str;
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

bool TestViewer::SaveResultInit() {
    // STObjRecognition::GlobalSummary::SetDatasetPath(m_dataset_dir);
    m_result_dir = m_result_dir + "/" + GetTimeStampString();
    if (!CreateFolder(m_result_dir)) {
        LOG(INFO) << "can't create the result dir" << m_result_dir;
    }

    camera_pose_result_file_ = m_result_dir + "/camera_pose_result.txt";
    object_pose_result_file_ = m_result_dir + "/object_pose_result.txt";

    camera_pose_result_stream_.open(camera_pose_result_file_);
    object_pose_result_stream_.open(object_pose_result_file_);

    if (!camera_pose_result_stream_.is_open()) {
        LOG(WARNING) << "camera pose result can't open "
                     << camera_pose_result_file_;
        return false;
    }

    if (!object_pose_result_stream_.is_open()) {
        LOG(WARNING) << "object pose result can't open "
                     << object_pose_result_file_;
        return false;
    }
    return true;
}

bool TestViewer::InitObjectRecognition() {
    ObjRecognitionExd::ObjRecongManager::Instance().CreateWithConfig();

    // set slam data callback

    // char *voc_buf = nullptr;
    // unsigned int voc_buf_size = 0;
    // LoadVoc("/home/zhangye/Develope/ObjectRecognition_ORBSLAM3/Vocabulary/voc.dat.zip",
    // &voc_buf, voc_buf_size);
    // ObjRecognitionExd::ObjRecongManager::Instance().LoadDic(voc_buf,
    // voc_buf_size);

    std::string cloud_point_model_dir =
        slam_saved_path + "/" + mappoint_filename;

#ifdef SUPERPOINT
    bool voc_load_res =
        ObjRecognitionExd::ObjRecongManager::Instance().LoadORBVoc(
            voc_path_superpoint);
#else
    bool voc_load_res =
        ObjRecognitionExd::ObjRecongManager::Instance().LoadORBVoc(voc_path);
#endif
    if (!voc_load_res) {
        LOG(ERROR) << "vocabulary load fail!";
    }

    int model_id = 0;
    char *cloud_point_model_buffer = nullptr;
    int cloud_point_model_buf_size = 0;
    LoadPointCloudModel(cloud_point_model_dir, m_pointCloud);
    ReadPointCloudModelToBuffer(
        cloud_point_model_dir, &cloud_point_model_buffer,
        cloud_point_model_buf_size);
    ObjRecognitionExd::ObjRecongManager::Instance().LoadModel(
        model_id, cloud_point_model_buffer, cloud_point_model_buf_size);
    SLAM->SetPointCloudModel(m_pointCloud);
    SLAM->mpViewer->SetPointCloudModel(m_pointCloud);
    SaveResultInit();
    return true;
}

void TestViewer::LoadImages(
    const string &strPathLeft, const string &strPathRight,
    const string &strPathTimes, vector<string> &vstrImageLeft,
    vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    if (!fTimes.is_open()) {
        LOG(FATAL) << "error open timestamp.txt";
    }
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str());
            vstrImageRight.push_back(strPathRight + "/" + ss.str());
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
    string pathCam1 = data_path + "/cam1/data";
    string pathImu = data_path + "/imu0/data.csv";

    LoadImages(
        pathCam0, pathCam1, pathTimeStamps, vstrImageLeft, vstrImageRight,
        vTimestampsCam);
    VLOG(0) << "LOADED!";
    nImages = vstrImageLeft.size();

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
    first_imu--; // first imu measurement to be considered

    cv::FileStorage fsSettings(config_path, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;
    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;
    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;
    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;
    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
        R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!"
             << endl;
        return -1;
    }

    cv::initUndistortRectifyMap(
        K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3),
        cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(
        K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3),
        cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);

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
        voc_path, config_path, ORB_SLAM3::System::IMU_STEREO, true,
        is_recognition);

    return true;
}

void TestViewer::SaveObjRecogResult() {
    Eigen::Matrix3f R_obj;
    Eigen::Matrix3f R_camera =
        ObjRecognition::TypeConverter::Mat3Array2Mat3Eigen(
            m_objrecog_result.R_camera);
    Eigen::Vector3f t_camera =
        Eigen::Vector3f::Map(m_objrecog_result.t_camera, 3);
    Eigen::Vector3f t_obj =
        Eigen::Vector3f::Map(m_objrecog_result.t_obj_buffer, 3);

    R_obj.row(0) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[0], 3);
    R_obj.row(1) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[3], 3);
    R_obj.row(2) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[6], 3);

    Eigen::Quaternionf q_camera(R_camera);
    Eigen::Quaternionf q_obj(R_obj);

    if (m_objrecog_result.frame_index <= 0) {
        VLOG(10) << "result frame index: " << m_objrecog_result.frame_index;
        return;
    }

    camera_pose_result_stream_
        << std::to_string(m_objrecog_result.time_stamp) << ","
        << std::setprecision(7) << t_camera(0) << "," << t_camera(1) << ","
        << t_camera(2) << "," << q_camera.w() << "," << q_camera.x() << ","
        << q_camera.y() << "," << q_camera.z() << std::endl;

    if (m_objrecog_result.num) {
        object_pose_result_stream_
            << std::to_string(m_objrecog_result.time_stamp) << ","
            << std::setprecision(7) << t_obj(0) << "," << t_obj(1) << ","
            << t_obj(2) << "," << q_obj.w() << "," << q_obj.x() << ","
            << q_obj.y() << "," << q_obj.z() << std::endl;
    }
}

void TestViewer::ObjectResultParse(
    const ObjRecognition::ObjRecogResult &result) {
    m_objrecog_result = result;
    Eigen::Matrix<float, 3, 3> Rcw =
        ObjRecognition::TypeConverter::Mat3Array2Mat3Eigen(
            m_objrecog_result.R_camera);
    Eigen::Vector3f tcw = Eigen::Vector3f::Map(m_objrecog_result.t_camera, 3);
    Eigen::Matrix<float, 3, 3> Rwo;
    Rwo.col(0) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[0], 3);
    Rwo.col(1) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[3], 3);
    Rwo.col(2) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[6], 3);
    Eigen::Vector3f two =
        Eigen::Vector3f::Map(&m_objrecog_result.t_obj_buffer[0], 3);

    Eigen::Matrix3f Rco = Eigen::Matrix3f::Identity();
    Rco = Rcw * Rwo;

    Eigen::Matrix3f Rslam2gl = Eigen::Matrix3f::Zero();
    Rslam2gl(0, 0) = 1;
    Rslam2gl(1, 2) = -1;
    Rslam2gl(2, 1) = 1;
    Rco = Rco; // * Rslam2gl.transpose();
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
    int info_size = m_objrecog_result.info_length;
    const char *info_char = m_objrecog_result.info;
    objrecog_info_str = std::string(info_char);
    SaveObjRecogResult();
    // ObjectResultTransmitMultiTabs();
}

bool TestViewer::RunObjectRecognition() {
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    int proccIm = 0;
    for (int ni = 0; ni < nImages; ni++, proccIm++) {
        // Read image from file
        imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);

        if (imLeft.empty()) {
            cerr << endl
                 << "Failed to load image at: " << string(vstrImageLeft[ni])
                 << endl;
            return 1;
        }

        if (imRight.empty()) {
            cerr << endl
                 << "Failed to load image at: " << string(vstrImageRight[ni])
                 << endl;
            return 1;
        }

        cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        double tframe = vTimestampsCam[ni];

        // Load imu measurements from previous frame
        vImuMeas.clear();

        if (ni > 0)
            while (
                vTimestampsImu[first_imu] <=
                vTimestampsCam
                    [ni]) // while(vTimestampsImu[first_imu]<=vTimestampsCam[ni])
            {
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

        cv::Mat camPos = SLAM->TrackStereo(
            imLeftRect, imRightRect, tframe,
            vImuMeas); // TODO change to monocular_inertial

        cv::Mat im_clone_left = imLeftRect.clone();
        int slam_state = SLAM->GetTrackingState();

        if (mRGB)
            SLAM->mpViewer->SetSLAMInfo(im_clone_left, slam_state, ni, camPos);
        else {
            cv::cvtColor(im_clone_left, im_clone_left, CV_RGB2BGR);
            SLAM->mpViewer->SetSLAMInfo(im_clone_left, slam_state, ni, camPos);
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

#ifdef OBJECTRECOGNITION
        ObjectResultParse(ObjRecognitionExd::ObjRecongManager::Instance()
                              .GetObjRecognitionResult());
#endif

        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6); // 1e6
    }
#ifdef OBJECTRECOGNITION
    ObjRecognition::GlobalSummary::SaveAllPoses(m_result_dir);

    /*ObjRecognition::GlobalSummary::SaveTimer(
        m_result_dir, STSLAMCommon::Timing::Print());

    ObjRecognition::GlobalSummary::SaveStatics(
        m_result_dir, STSLAMCommon::Statistics::Print());*/

    ObjRecognitionExd::ObjRecongManager::Instance().Destroy();
#endif

    // Stop all threads
    SLAM->Shutdown();

    const string kf_file = slam_saved_path + "/kf_" + dataset_name + ".txt";
    const string f_file = slam_saved_path + "/f_" + dataset_name + ".txt";
    SLAM->SaveTrajectoryEuRoC(f_file);
    SLAM->SaveKeyFrameTrajectoryEuRoC(kf_file);

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
    fsSettings["data_path"] >> testViewer.data_path;
    fsSettings["config_path"] >> testViewer.config_path;
    fsSettings["saved_path"] >> testViewer.slam_saved_path;
    fsSettings["mappoint_filename"] >> testViewer.mappoint_filename;
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
