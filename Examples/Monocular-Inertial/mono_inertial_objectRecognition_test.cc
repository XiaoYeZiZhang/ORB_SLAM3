//
// Created by zhangye on 2020-09-14.
//
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <include/ORBSLAM3/System.h>
#include "include/ORBSLAM3/ImuTypes.h"
#include "ObjectRecognition/Utility/Camera.h"

//#define SCANNER;

using namespace std;
class TestViewer {
public:
    bool InitializeSLAM(int argc, char *argv[]);
    bool InitializeObjectRecognition();
    bool RunSLAM(int argc, char *argv[]);
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
};

bool TestViewer::InitializeObjectRecognition() {
    return true;
}

bool TestViewer::SaveMappointFor3DObject(const std::string save_path) {
    char *buffer = NULL;
    int buffer_size = 0;

    std::ofstream out(save_path, std::ios::out | std::ios::binary);
    if (out.is_open()) {
        out.write(buffer, buffer_size);
        delete[] buffer;
    } else {
        delete[] buffer;
        std::cout << "Error opening the pointCloud file!";
        return false;
    }
    return SLAM->PackAtlasToMemoryFor3DObject(&buffer, buffer_size);
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

bool TestViewer::InitializeSLAM(int argc, char *argv[]) {
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

    SLAM = new ORB_SLAM3::System(
        argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true);
    return true;
}

bool TestViewer::RunSLAM(int argc, char *argv[]) {
    int proccIm = 0;
    for (int seq = 0; seq < num_seq; seq++) {

        // Main loop
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        nImages[seq] = 20;
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

    TestViewer testViewer;

    bool initial_slam_result = testViewer.InitializeSLAM(argc, argv);

    if(!initial_slam_result) {
        std::cout << "slam initialize fail!" << std::endl;
        return 0;
    }

    bool initialize_objectRecognition_result = testViewer.InitializeObjectRecognition();
    if(!initialize_objectRecognition_result) {
        std::cout << "objectRecognition initialize fail!" << std::endl;
        return 0;
    }

    testViewer.RunSLAM(argc, argv);

#ifdef SCANNER
        std::string mappoint_save_path = "/home/zhangye/data/ObjectRecognition/shoe.bin";
        if(testViewer.SaveMappointFor3DObject(mappoint_save_path)) {
            std::cout << "save mappoint for 3dobject success!";
        }
#endif
    return 0;
}
