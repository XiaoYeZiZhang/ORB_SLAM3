//
// Created by zhangye on 2020-09-14.
//
#include <iostream>
#include <algorithm>
#include <chrono>
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
#include "mode.h"

using namespace std;
class TestViewer {
public:
    bool InitSLAM(int argc, char **argv);
    bool InitObjectRecognition();
    bool RunSLAM(int argc, char *argv[]);
    bool RunSLAMScanner(const std::string& mappoint_save_dir);

    ORB_SLAM3::System* GetSystem() {
        return SLAM;
    }
    bool SaveMappointFor3DObject(const std::string save_path);
private:
    void LoadImages(
        const string &strImagePath, const string &strPathTimes,
        vector<string> &vstrImages, vector<double> &vTimeStamps);
    void LoadIMU(
        const string &strImuPath, vector<double> &vTimeStamps,
        vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);
    bool SaveResultInit();
    void ObjectResultParse(const ObjRecognition::ObjRecogResult &result);
    void SaveObjRecogResult();


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

    // 3d object
    std::shared_ptr<ObjRecognition::Object> m_pointCloud =
        std::make_shared<ObjRecognition::Object>(0);
    std::string camera_pose_result_file_;
    std::string object_pose_result_file_;
    std::ofstream camera_pose_result_stream_;
    std::ofstream object_pose_result_stream_;
    ObjRecognition::ObjRecogResult m_objrecog_result;
    Eigen::Matrix<double, 3, 3> m_Row = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 1> m_Tow = Eigen::Matrix<double, 3, 1>::Zero();
    std::string objrecog_info_str;

};

bool TestViewer::SaveResultInit() {
    //STObjRecognition::GlobalSummary::SetDatasetPath(m_dataset_dir);
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

    //char *voc_buf = nullptr;
    //unsigned int voc_buf_size = 0;
    //LoadVoc("/home/zhangye/Develope/ObjectRecognition_ORBSLAM3/Vocabulary/voc.dat.zip", &voc_buf, voc_buf_size);
    //ObjRecognitionExd::ObjRecongManager::Instance().LoadDic(voc_buf, voc_buf_size);

    std::string voc_path = "/home/zhangye/Develope/ObjectRecognition_ORBSLAM3/Vocabulary/ORBvoc.txt";
    std::string cloud_point_model_dir = "/home/zhangye/data/ObjectRecognition/shoe.bin";

    bool voc_load_res = ObjRecognitionExd::ObjRecongManager::Instance().LoadORBVoc(voc_path);
    if(!voc_load_res) {
        LOG(ERROR) << "vocabulary load fail!";
    }

    int model_id = 0;
    char *cloud_point_model_buffer = nullptr;
    int cloud_point_model_buf_size = 0;
    LoadPointCloudModel(cloud_point_model_dir, m_pointCloud);
    ReadPointCloudModelToBuffer(
        cloud_point_model_dir, &cloud_point_model_buffer,
        cloud_point_model_buf_size);
    ObjRecognitionExd::ObjRecongManager::Instance().LoadModel(model_id,
         cloud_point_model_buffer, cloud_point_model_buf_size);
    SLAM->SetPointCloudModel(m_pointCloud);
    SLAM->mpViewer->SetPointCloudModel(m_pointCloud);
    SaveResultInit();
    return true;
}

bool TestViewer::SaveMappointFor3DObject(const std::string save_path) {
    char *buffer = NULL;
    int buffer_size = 0;
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

            vTimeStamps.push_back(data[0]);
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

        string pathCam0 = pathSeq + "/camera/images";
        string pathImu = pathSeq + "/imu/data.csv";

        // 图像要和时间戳对齐
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

    /*cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU data in the sequence: " << nImu << endl << endl;*/

    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    double fx = fsSettings["Camera.fx"];
    double fy = fsSettings["Camera.fy"];
    double cx = fsSettings["Camera.cx"];
    double cy = fsSettings["Camera.cy"];
    int width = fsSettings["Camera.width"];
    int height = fsSettings["Camera.height"];

    ObjRecognition::CameraIntrinsic::GetInstance().SetParameters(
        fx, fy, cx, cy, width, height);

#ifdef OBJECTRECOGNITION
    SLAM = new ORB_SLAM3::System(
        argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true, true);
#else
    SLAM = new ORB_SLAM3::System(
        argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true, false);
#endif
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

void TestViewer::ObjectResultParse(const ObjRecognition::ObjRecogResult &result) {

    m_objrecog_result = result;
    Eigen::Matrix<float, 3, 3> Rcw =
        ObjRecognition::TypeConverter::Mat3Array2Mat3Eigen(
            m_objrecog_result.R_camera);
    Eigen::Vector3f Tcw = Eigen::Vector3f::Map(m_objrecog_result.t_camera, 3);
    Eigen::Matrix<float, 3, 3> Rwo;
    Rwo.col(0) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[0], 3);
    Rwo.col(1) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[3], 3);
    Rwo.col(2) = Eigen::Vector3f::Map(&m_objrecog_result.R_obj_buffer[6], 3);
    Eigen::Vector3f Two =
        Eigen::Vector3f::Map(&m_objrecog_result.t_obj_buffer[0], 3);

    Eigen::Matrix3f Rco = Eigen::Matrix3f::Identity();
    Rco = Rcw * Rwo;

    Eigen::Matrix3f Rslam2gl = Eigen::Matrix3f::Zero();
    Rslam2gl(0, 0) = 1;
    Rslam2gl(1, 2) = -1;
    Rslam2gl(2, 1) = 1;
    Rco = Rco * Rslam2gl.transpose();
    Rwo = Rcw.transpose() * Rco;
    Eigen::Matrix3f Row = Eigen::Matrix3f::Identity();
    Eigen::Vector3f Tow = Eigen::Vector3f::Zero();
    Row = Rwo.transpose();
    Tow = -Row * Two;

    if (result.num == 1) {
        m_Row = Row.cast<double>(); // world -> obj
        m_Tow = Tow.cast<double>();
    } else {
        m_Row = Eigen::Matrix3d::Identity();
        m_Tow = Eigen::Vector3d::Zero();
    }

    int info_size = m_objrecog_result.info_length;
    const char *info_char = m_objrecog_result.info;
    objrecog_info_str = std::string(info_char);

    // save the object recognition result to file.
    SaveObjRecogResult();

    //ObjectResultTransmitMultiTabs();
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
    int proccIm = 0;
    for (int seq = 0; seq < num_seq; seq++) {
        // Main loop
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        //nImages[seq] = 20;
        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {
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

            // Pass the image to the SLAM system
            // cout << "tframe = " << tframe << endl;
            SLAM->TrackMonocular(
                im, tframe, vImuMeas); // TODO change to monocular_inertial

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
            // std::cout << "ttrack: " << ttrack << std::endl;

            vTimesTrack[ni] = ttrack;


            ObjectResultParse(ObjRecognitionExd::ObjRecongManager::Instance().GetObjRecognitionResult());

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


    ObjRecognition::GlobalSummary::SaveAllPoses(m_result_dir);

    /*ObjRecognition::GlobalSummary::SaveTimer(
        m_result_dir, STSLAMCommon::Timing::Print());

    ObjRecognition::GlobalSummary::SaveStatics(
        m_result_dir, STSLAMCommon::Statistics::Print());*/

    ObjRecognitionExd::ObjRecongManager::Instance().Destroy();

    // Stop all threads
    SLAM->Shutdown();

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
    return true;
}

int main(int argc, char *argv[]) {

    Eigen::Vector2d x;
    x.homogeneous();
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

#ifdef OBJECTRECOGNITION
    bool initialize_objectRecognition_result =
        testViewer.InitObjectRecognition();
    if(!initialize_objectRecognition_result) {
        std::cout << "objectRecognition initialize fail!" << std::endl;
        return 0;
    }

#endif

    testViewer.RunSLAM(argc, argv);

#ifdef SCANNER
    std::string mappoint_save_path = "/home/zhangye/data/ObjectRecognition/shoe.bin";
    if(testViewer.SaveMappointFor3DObject(mappoint_save_path)) {
        std::cout << "save mappoint for 3dobject success!";
    }
#endif

    return 0;
}
