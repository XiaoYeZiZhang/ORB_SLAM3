//
// Created by root on 2020/10/12.
//
#include <ctime>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <include/ORBSLAM3/SPextractor.h>
#include <include/ObjectRecognition/Utility/Parameters.h>
#include <include/ORBSLAM3/FrameObjectProcess.h>
#include "ORBSLAM3/System.h"
#include "Utility/GlobalSummary.h"
#include "Utility/FileIO.h"
#include "Utility/Camera.h"
#include "ObjectRecognitionSystem/ObjectRecognitionManager.h"
#include "include/Tools.h"
#include "ORBSLAM3/ViewerAR.h"
#include "mode.h"

using namespace std;
class TestViewer {
public:
    bool InitSLAM();
    bool RunScanner();
    bool SaveMappointFor3DObject(const std::string save_path);
    bool SaveMappointFor3DObject_SuperPoint(const std::string save_path);
    cv::Mat M2l;
    cv::Mat M1r;
    cv::Mat M2r;
    cv::Mat M1l;
    // yaml
    std::string voc_path;
    std::string data_path;
    std::string config_path;
    std::string slam_saved_path;
    std::string mappoint_filename;
    std::string dataset_name;

private:
    void LoadImages(
        const string &strPathLeft, const string &strPathRight,
        const string &strPathTimes, vector<string> &vstrImageLeft,
        vector<string> &vstrImageRight, vector<double> &vTimeStamps);
    void SfMIfSp();
    void FindMatchByKNN(
        const cv::Mat &frmDesp, const cv::Mat &pcDesp,
        std::vector<cv::DMatch> &goodMatches);
    void DebugMode();
    std::string m_result_dir;
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;

    int nImages;
    ORB_SLAM3::System *SLAM;

    // ar
    ORB_SLAM3::ViewerAR viewerAR;
    bool bRGB = true;
    float fps = -1;
    cv::Mat K;
    std::vector<Eigen::Vector3d> m_boundingbox_w;
};

bool TestViewer::SaveMappointFor3DObject_SuperPoint(
    const std::string save_path) {
    char *buffer = NULL;
    int buffer_size = 0;
    SLAM->SetScanBoundingbox_W(m_boundingbox_w);

    bool save_result =
        SLAM->PackAtlasToMemoryFor3DObject_SuperPoint(&buffer, buffer_size);
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

    // data root path
    string pathTimeStamps = data_path + "/cam0/timestamp.txt";
    string pathCam0 = data_path + "/cam0/data";
    string pathCam1 = data_path + "/cam1/data";

    // load images
    LoadImages(
        pathCam0, pathCam1, pathTimeStamps, vstrImageLeft, vstrImageRight,
        vTimestampsCam);
    VLOG(0) << "LOADED!";

    nImages = vstrImageLeft.size();

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

    // get map, use remap() to get the rectified image
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
    int width = cols_l;
    int height = rows_l;
    ObjRecognition::CameraIntrinsic::GetInstance().SetParameters(
        fx, fy, cx, cy, width, height);

    double scaleFactor = fsSettings["ORBextractor.scaleFactor"];
    int nlevels = fsSettings["ORBextractor.nLevels"];
    double fastInit = fsSettings["ORBextractor.iniThFAST"];
    double fastThreathold = fsSettings["ORBextractor.minThFAST"];
    Parameters::GetInstance().SetScaleFactor(scaleFactor);
    Parameters::GetInstance().SetLevels(nlevels);
    Parameters::GetInstance().SetFastInit(fastInit);
    Parameters::GetInstance().SetFastThreathold(fastThreathold);

    bRGB = static_cast<bool>((int)fsSettings["Camera.RGB"]);
    fps = fsSettings["Camera.fps"];

    SLAM = new ORB_SLAM3::System(
        voc_path, config_path, ORB_SLAM3::System::STEREO, false, false);

    return true;
}

void TestViewer::DebugMode() {
    while (true) {
        if (!viewerAR.GetDebugFlag()) {
            return;
        }
        usleep(1 * 1e5);
        if (viewerAR.GetStopFlag()) {
#ifdef SUPERPOINT
            SfMIfSp();
#endif
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

#ifdef ORBPOINT
            // extract more keypoihts
            ORB_SLAM3::FrameObjectProcess::GetInstance()->SetBoundingBox(
                m_boundingbox_w);
#endif
            viewerAR.SetFixFlag(false);
        }
    }
}

void TestViewer::FindMatchByKNN(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    // use L2 norm instead of Hamming distance
    cv::BFMatcher matcher(cv::NormTypes::NORM_L2);
    matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);
    VLOG(5) << "KNN Matches size: " << knnMatches.size();

    for (size_t i = 0; i < knnMatches.size(); i++) {
        cv::DMatch &bestMatch = knnMatches[i][0];
        cv::DMatch &betterMatch = knnMatches[i][1];
        const float distanceRatio = bestMatch.distance / betterMatch.distance;
        VLOG(50) << "distanceRatio = " << distanceRatio;
        // the farest distance, the better result
        const float kMinDistanceRatioThreshld = 0.80;
        if (distanceRatio < kMinDistanceRatioThreshld) {
            matches.push_back(bestMatch);
        }
    }

    VLOG(15) << "after distance Ratio matches size: " << matches.size();

    double minDisKnn = 9999.0;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance < minDisKnn) {
            minDisKnn = matches[i].distance;
        }
    }
    VLOG(15) << "minDisKnn = " << minDisKnn;

    // set good_matches_threshold
    const int kgoodMatchesThreshold = 200;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= kgoodMatchesThreshold) {
            goodMatches.push_back(matches[i]);
        }
    }
}

void TestViewer::SfMIfSp() {
    // TODO(zhangye): DO SFM USING SUPERPOINT
    VLOG(0) << "DOING SFM USING SUPERPOINT, PLEASE WAIT...";
    std::vector<ORB_SLAM3::KeyFrame *> keyframes =
        SLAM->mpAtlas->GetAllKeyFrames();

    ORB_SLAM3::SPextractor *SPextractor =
        new ORB_SLAM3::SPextractor(2000, 1.2, 3, 0.015, 0.007, true);
    for (size_t i = 0; i < keyframes.size(); i++) {
        ORB_SLAM3::KeyFrame *keyframe = keyframes[i];
        cv::Mat Tcw_cv = keyframe->GetPose();
        Eigen::Matrix4d Tcw_eigen;
        cv::cv2eigen(Tcw_cv, Tcw_eigen);
        Eigen::Matrix3d Rcw = Tcw_eigen.block<3, 3>(0, 0);
        Eigen::Vector3d tcw = Tcw_eigen.block<3, 1>(0, 3);
        cv::Mat mask;
        Tools::GetBoundingBoxMask(
            keyframe->imgLeft,
            ObjRecognition::CameraIntrinsic::GetInstance().GetEigenK(), Rcw,
            tcw, m_boundingbox_w, mask);

        (*SPextractor)(
            keyframe->imgLeft, mask, keyframe->mvKeys_superpoint,
            keyframe->mDescriptors_superpoint);

        keyframe->mvKeysUn_superpoint = keyframe->mvKeys_superpoint;
        keyframe->N_superpoint = keyframe->mvKeys_superpoint.size();

        // compute dbow
        keyframe->ComputeBoW_SuperPoint();
    }
    SLAM->mpLocalMapper->TriangulateForSuperPoint();
}
// click boundingbox fix button:
// 1. get the boundingbox
// 2. setboundingbox
// 3. run slam and extract more orb features
// 4. save mappoint
bool TestViewer::RunScanner() {
    viewerAR.SetSLAM(SLAM);
    viewerAR.SetFPS(fps);
    viewerAR.SetCameraCalibration(
        ObjRecognition::CameraIntrinsic::GetInstance().FX(),
        ObjRecognition::CameraIntrinsic::GetInstance().FY(),
        ObjRecognition::CameraIntrinsic::GetInstance().CX(),
        ObjRecognition::CameraIntrinsic::GetInstance().CY());

    K = cv::Mat::eye(3, 3, CV_32F);
    K = ObjRecognition::CameraIntrinsic::GetInstance().GetCVK();

    thread tViewer = thread(&ORB_SLAM3::ViewerAR::Run, &viewerAR);

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    int proccIm = 0;
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

        DebugMode();

        if (viewerAR.GetStopFlag()) {
#ifdef SUPERPOINT
            SfMIfSp();
#endif
            break;
        }

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

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 =
            std::chrono::monotonic_clock::now();
#endif

        // TODO(zhangye): check the pose: left image pose??
        cv::Mat Tcw = SLAM->TrackStereo(
            imLeftRect, imRightRect, tframe, vector<ORB_SLAM3::IMU::Point>(),
            vstrImageLeft[ni]); // TODO change to monocular_inertial

        // TODO(zhangye): imLeft or imLeftRect
        cv::Mat im_clone_left = imLeftRect.clone();
        int state = SLAM->GetTrackingState();
        vector<ORB_SLAM3::MapPoint *> vMPs = SLAM->GetTrackedMapPoints();
        vector<cv::KeyPoint> vKeys = SLAM->GetTrackedKeyPointsUn();

        if (bRGB)
            viewerAR.SetImagePose(im_clone_left, Tcw, state, vKeys, vMPs);
        else {
            cv::cvtColor(im_clone_left, im_clone_left, CV_RGB2BGR);
            viewerAR.SetImagePose(im_clone_left, Tcw, state, vKeys, vMPs);
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

    std::string mappoint_save_path = slam_saved_path + "/" + mappoint_save_path;
#ifdef SUPERPOINT
    if (SaveMappointFor3DObject_SuperPoint(mappoint_save_path)) {
        VLOG(0) << "save mappoint_superpoint for 3dobject success!";
    }
#else
    if (SaveMappointFor3DObject(mappoint_save_path)) {
        VLOG(0) << "save mappoint for 3dobject success!";
    }
#endif

    // Save camera trajectory
    const string kf_file = slam_saved_path + "/kf_" + dataset_name + ".txt";
    const string f_file = slam_saved_path + "/f_" + dataset_name + ".txt";
    SLAM->SaveTrajectoryEuRoC(f_file);
    SLAM->SaveKeyFrameTrajectoryEuRoC(kf_file);

    tViewer.join();
    return true;
}

int main(int argc, char *argv[]) {
#ifdef SCANNER
#else
    LOG(FATAL) << "not in the scanner mode";
#endif
    if (argc < 1) {
        cerr << endl << "Usage: ./ path_to_yaml " << endl;
        return 1;
    }

    // vlog settings
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

    testViewer.RunScanner();
    return 0;
}
