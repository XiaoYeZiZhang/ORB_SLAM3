//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include <cv.hpp>
#include "ObjectRecognition/Utility/Utility.h"
#include "Detector/DetectorCommon.h"
#include "Utility/Camera.h"
#include "Utility/Parameters.h"
#include "Detector/PointCloudDetector.h"
#include "Optimizer/LBAOptimizer.h"
namespace ObjRecognition {

PointCloudObjDetector::PointCloudObjDetector() {
    Clear();
}

PointCloudObjDetector::~PointCloudObjDetector() {
}

void PointCloudObjDetector::SetPointCloudObj(
    const std::shared_ptr<Object> &pObj) {
    VLOG(0) << "PointCloud detector set obj begin";
    CHECK_NOTNULL(pObj.get());
    mObj = pObj;

    VLOG(0) << "PointCloud detector set obj success";
}

void PointCloudObjDetector::SetVoc(
    const std::shared_ptr<DBoW3::Vocabulary> &pVoc) {
    CHECK_NOTNULL(pVoc.get());
    voc_ = pVoc;
}

static PS::Point2D NormalizePoint2D(
    const cv::Point2d &pt, const float &fx, const float &fy, const float &cx,
    const float &cy) {
    PS::Point2D p2d;
    p2d(0) = (pt.x - cx) / fx;
    p2d(1) = (pt.y - cy) / fy;
    return std::move(p2d);
}

void MatchKeyFramesShow(
    const std::shared_ptr<DetectorFrame> &frm,
    const std::vector<KeyFrame::Ptr> &kf_matches) {
#ifdef MOBILE_PLATFORM
    return;
#endif
    for (int index = 0; index < kf_matches.size(); index++) {
        cv::Mat kf_raw_image = kf_matches.at(index)->GetRawImage();
        if (!kf_raw_image.empty()) {
            cv::Mat imshow;
            cv::drawMatches(
                frm->m_raw_image, frm->m_kpts, kf_raw_image,
                kf_matches.at(index)->GetKeyPoints(),
                frm->m_dmatches_2d.at(index), imshow);
            const std::string kf_match_result_name =
                "match 2D: " + std::to_string(index);
            // GlobalOcvViewer::UpdateView(kf_match_result_name, imshow);
        }
    }
}

std::vector<PS::MatchSet2D> Generate2DMatchesFromKeyFrame(
    const std::shared_ptr<DetectorFrame> &frm,
    const std::vector<KeyFrame::Ptr> &kf_matches) {

    float fx = CameraIntrinsic::GetInstance().GetEigenK()(0, 0);
    float fy = CameraIntrinsic::GetInstance().GetEigenK()(1, 1);
    float cx = CameraIntrinsic::GetInstance().GetEigenK()(0, 2);
    float cy = CameraIntrinsic::GetInstance().GetEigenK()(1, 2);

    int matches2d_count = 0;
    cv::Mat frmDesp = frm->m_desp;
    std::vector<PS::MatchSet2D> matchs_2ds;

    if (kf_matches.size() <= 0)
        return matchs_2ds;

    frm->m_dmatches_2d.clear();
    for (int index = 0; index < kf_matches.size(); index++) {
        CHECK_NOTNULL(kf_matches.at(index).get());
        VLOG(10) << "2DMatch Frame QuryMap id: "
                 << kf_matches.at(index)->GetID();
        PS::MatchSet2D matches;
        std::vector<cv::DMatch> dmatches;

        cv::Mat kfDesp = kf_matches.at(index)->GetDesciriptor();
        ObjDetectionCommon::FindMatchByKNN(frmDesp, kfDesp, dmatches);
        VLOG(15) << "2DMatch frm kpts num: " << frm->m_kpts.size();
        VLOG(15) << "2DMatch kf kpts num: "
                 << kf_matches.at(index)->GetKeyPoints().size();
        VLOG(10) << "2DMatch num: " << dmatches.size();

        frm->m_dmatches_2d.push_back(dmatches);

        matches2d_count = dmatches.size();
        for (const auto &it : dmatches) {
            PS::Point2D queryP2d =
                NormalizePoint2D(frm->m_kpts[it.queryIdx].pt, fx, fy, cx, cy);
            PS::Point2D matchP2d = NormalizePoint2D(
                kf_matches.at(index)->GetKeyPoints()[it.trainIdx].pt, fx, fy,
                cx, cy);
            matches.push_back(std::move(
                PS::Match2D(std::move(matchP2d), std::move(queryP2d))));
        }

        Eigen::Matrix3d Rcw_kf;
        Eigen::Vector3d tcw_kf;
        kf_matches.at(index)->GetPose(Rcw_kf, tcw_kf);
        matches.m_T1.m_R = Rcw_kf.cast<float>();
        matches.m_T1.m_t = tcw_kf.cast<float>();

        matchs_2ds.push_back(matches);
    }

    // STSLAMCommon::StatsCollector stats_collector("detector 2D matches num");
    // stats_collector.AddSample(matches2d_count);

    MatchKeyFramesShow(frm, kf_matches);

    return matchs_2ds;
}

void PointCloudObjDetector::PreProcess(
    const std::shared_ptr<ObjRecognition::FrameData> &frm) {

    m_frame_cur = std::make_shared<DetectorFrame>();
    m_frame_cur->m_time_stamp = frm->mTimeStamp;
    m_frame_cur->m_frame_index = frm->mFrmIndex;
    m_frame_cur->m_raw_image = frm->img.clone();
    m_frame_cur->m_desp = frm->mDesp.clone();
    m_frame_cur->m_kpts = frm->mKpts;
    m_frame_cur->SetCameraPose(frm->mRcw, frm->mTcw);

    Rcw_cur_ = frm->mRcw;
    tcw_cur_ = frm->mTcw;
}

std::vector<PS::MatchSet2D>
PointCloudObjDetector::Find2DMatches(const std::vector<KeyFrame::Ptr> &allKFs) {

    // STSLAMCommon::Timer timer_find_2dmatch("find 2d matches");

    std::vector<KeyFrame::Ptr> kf_mathceds = mObj->FrameQueryMap(m_frame_cur);
    std::vector<PS::MatchSet2D> matches_2ds;

    if (kf_mathceds.size() == 0) {
        LOG(ERROR) << "2DMatch Frames QuryMap is empty";
        return matches_2ds;
    }

    std::set<int> kf_matches_id;
    for (KeyFrame::Ptr kf : kf_mathceds) {
        kf_matches_id.insert(kf->GetID());
    }

    // GlobalKeyFrameMatchViewer::SetMatchedKeyFrames(kf_matches_id);
    matches_2ds = Generate2DMatchesFromKeyFrame(m_frame_cur, kf_mathceds);

    // timer_find_2dmatch.Stop();

    return matches_2ds;
}

PS::MatchSet3D PointCloudObjDetector::Find3DMatch() {
    PS::MatchSet3D matches_3d;
    cv::Mat pcDesp;

    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    float fx = static_cast<float>(Kcv.at<double>(0, 0));
    float fy = static_cast<float>(Kcv.at<double>(1, 1));
    float cx = static_cast<float>(Kcv.at<double>(0, 2));
    float cy = static_cast<float>(Kcv.at<double>(1, 2));

    pcDesp = ObjDetectionCommon::GetPointCloudDesp(mObj);

    cv::Mat frmDesp;
    frmDesp = m_frame_cur->m_desp;

    std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    std::map<int, MapPointIndex> matches2dTo3d;
    matches2dTo3d.clear();
    knn_match_num_ = -1;

    std::vector<cv::DMatch> goodMatches;
    ObjDetectionCommon::FindMatchByKNN(frmDesp, pcDesp, goodMatches);

    for (int i = 0; i < goodMatches.size(); i++) {
        matches2dTo3d.insert(std::pair<int, MapPointIndex>(
            goodMatches[i].queryIdx, goodMatches[i].trainIdx));
    }

    knn_match_num_ = goodMatches.size();
    VLOG(5) << "detection goodMatches num: " << goodMatches.size();
    // STSLAMCommon::StatsCollector stats_collector_knn(
    //"detector knn matches num");
    // stats_collector_knn.AddSample(knn_match_num_);

    const int kGoodMatchNumTh =
        Parameters::GetInstance().kDetectorKNNMatchNumTh;
    if (knn_match_num_ < kGoodMatchNumTh) {
        VLOG(5) << "The number of KNN match is small";
        return matches_3d;
    }

    std::vector<cv::Point3f> ptsObj;
    std::vector<cv::Point2f> ptsImg;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        ptsImg.emplace_back(cv::Point2f(keyPoints[iter->first].pt));
        cv::Point3f mapPoint =
            TypeConverter::Eigen2CVPoint(pointClouds[iter->second]->GetPose());
        ptsObj.emplace_back(mapPoint);
    }

    for (int i = 0; i < ptsObj.size(); i++) {
        PS::Point3D point_3d(
            (double)ptsObj[i].x, (double)ptsObj[i].y, (double)ptsObj[i].z);
        PS::Point2D point_2d =
            NormalizePoint2D((cv::Point2d)ptsImg[i], fx, fy, cx, cy);
        PS::Match3D match_3d(point_3d, point_2d);
        matches_3d.push_back(match_3d);
    }

    m_frame_cur->m_matches_3d.clear();
    m_frame_cur->m_matches_3d = matches2dTo3d;
    return matches_3d;
}

void GetMatch2dTo3dInliers(
    const std::map<int, MapPointIndex> &matches2dTo3d,
    const std::vector<int> &inliers_3d,
    std::map<int, MapPointIndex> &matches2dTo3dInliers) {

    matches2dTo3dInliers.clear();
    if (inliers_3d.size() == 0) {
        return;
    }
    int index_ = 0;
    int inliersNum = 0;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        if (index_ == inliers_3d[inliersNum]) {
            matches2dTo3dInliers.insert(
                std::pair<int, MapPointIndex>(iter->first, iter->second));
            inliersNum++;
        }
        if (inliersNum == inliers_3d.size())
            break;
        index_++;
    }
}

void PointCloudObjDetector::PoseOptimize(const std::vector<int> &inliers_3d) {
    m_frame_cur->m_matches2dto3d_inliers.clear();
    GetMatch2dTo3dInliers(
        m_frame_cur->m_matches_3d, inliers_3d,
        m_frame_cur->m_matches2dto3d_inliers);
    if (m_frame_cur->m_matches2dto3d_inliers.size() <
        Parameters::GetInstance().kDetectorPnPInliersUnreliableNumTh) {
        VLOG(0) << "not enough 3d inliers for ceres optimization!";
        return;
    }

    /*LBAOptimizer optimizer;
    std::vector<Eigen::Matrix3d> optimizedRcos;
    std::vector<Eigen::Vector3d> optimizedTcos;
    const Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();

    optimizedRcos.emplace_back(Rco_cur_);
    optimizedTcos.emplace_back(tco_cur_);

    int optimizeCameraPoseNum = optimizedRcos.size();
    // STSLAMCommon::Timer timer("Ceres Optimization");
    bool optimized = optimizer.PoseCeresOptimization(
        m_frame_cur->m_kpts, mObj->GetPointClouds(),
        m_frame_cur->m_matches2dto3d_inliers, K, optimizedRcos, optimizedTcos);
    // VLOG(0) << "detection ceres optimization time: " << timer.Stop();
    if (optimized) {
        for (int i = 0; i < optimizeCameraPoseNum; i++) {
            Rco_cur_ = optimizedRcos[i];
            tco_cur_ = optimizedTcos[i];
        }
    }*/
}

bool PointCloudObjDetector::PoseSolver(
    const PS::MatchSet3D &matches_3d,
    const std::vector<PS::MatchSet2D> &matches_2d, std::vector<int> &inliers_3d,
    std::vector<std::vector<int>> &inliers_2d) {
    // STSLAMCommon::Timer detectionPoseSolverTime("detection poseSolver
    // process");
    PS::Options options;
    PS::Pose T;
    pnp_inliers_num_ = 0;
    pnp_inliers_3d_num_ = 0;
    pnp_inliers_2d_num_ = 0;

    const double kPnpReprojectionError = 6.5;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, 1.0);
    Eigen::Vector3d gravityCamera = Rcw_cur_ * gravity;

    options.focal_length = static_cast<float>(Kcv.at<double>(0, 0));
    options.max_reproj_err = kPnpReprojectionError / options.focal_length;
    options.enable_2d_solver = false;
#ifdef OBJ_WITH_KF
    options.enable_2d_solver = true;
#endif
    options.enable_3d_solver = true;
    options.ransac_iterations = 100;
    options.ransac_confidence = 0.85;
    options.gravity_dir = gravityCamera.cast<float>();
    options.gravity_dir_max_err_deg = 180;
    options.enable_gravity_solver = true;
    options.prefer_pure_2d_solver = false;
    options.try_refine_translation_before_optimization_for_2d_only_matches =
        false;
    const int kPnpMinMatchesNum = 0;

#ifdef OBJ_WITH_KF
    const int kPnpMinInlierNum =
        Parameters::GetInstance().kDetectorPnPInliersGoodWithKFNumTh;
#else
    const int kPnpMinInlierNum =
        Parameters::GetInstance().kDetectorPnPInliersGoodNumTh;
#endif

    const double kPnpMinInlierRatio = 0.0;
    options.callbacks.emplace_back(PS::EarlyBreakBy3DInlierCounting(
        kPnpMinMatchesNum, kPnpMinInlierNum, kPnpMinInlierRatio));
    options.CheckValidity();

    pnp_solver_result_ = PS::Ransac(
        options, matches_3d, matches_2d, &T, &inliers_3d, &inliers_2d);

    VLOG(0) << "detectionPoseSolver result " << std::boolalpha
            << pnp_solver_result_;
    VLOG(0) << "detectionPoseSolver 3d matches size: " << matches_3d.size();
    VLOG(0) << "detectionPoseSolver 3d inliers size: " << inliers_3d.size();
    for (int i = 0; i < matches_2d.size(); i++) {
        VLOG(0) << "detectionPoseSolver 2d matches size: "
                << matches_2d.at(i).size();
        VLOG(0) << "detectionPoseSolver 2d inliers size: "
                << inliers_2d.at(i).size();
    }
    VLOG(20) << "detectionPoseSolver PoseSolve R: " << T.m_R;
    VLOG(20) << "detectionPoseSolver PoseSolve t: " << T.m_t;
    Eigen::Matrix3d Rco = T.m_R.cast<double>();
    Eigen::Vector3d tco = T.m_t.cast<double>();
    Rco_cur_ = Rco;
    tco_cur_ = tco;

    PoseOptimize(inliers_3d);

    Eigen::Matrix3d Row = Rco.transpose() * (Rcw_cur_);
    Eigen::Vector3d Tow = Rco.transpose() * (tcw_cur_ - tco);
    Rwo_cur_ = Row.transpose();
    two_cur_ = -Rwo_cur_ * Tow;

    m_frame_cur->SetObjectPose(Rwo_cur_, two_cur_);
    m_frame_cur->SetObjectPoseInCamemra(Rco_cur_, tco_cur_);
    pnp_inliers_num_ = inliers_3d.size();
    pnp_inliers_3d_num_ = pnp_inliers_num_;
    for (const auto &it : inliers_2d) {
        pnp_inliers_num_ += it.size();
        pnp_inliers_2d_num_ += it.size();
    }

    VLOG(5) << "detection inlier num: " << pnp_inliers_num_;
    // STSLAMCommon::StatsCollector stats_collector_pnp_3d(
    //"detector pnp 3d inlier num");
    // stats_collector_pnp_3d.AddSample(pnp_inliers_3d_num_);
    // STSLAMCommon::StatsCollector stats_collector_pnp_2d(
    //"detector pnp 2d inlier num");
    // stats_collector_pnp_2d.AddSample(pnp_inliers_2d_num_);

    // VLOG(10) << "detection poseSolver process time:"
    //<< detectionPoseSolverTime.Stop();
    return pnp_solver_result_;
}

void PointCloudObjDetector::PnPResultHandle() {

#ifdef OBJ_WITH_KF
    const int kPNPInliersThresholdGood =
        Parameters::GetInstance().kDetectorPnPInliersGoodWithKFNumTh;
    const int kPNPInliersThresholdUnreliable =
        Parameters::GetInstance().kDetectorPnPInliersUnreliableWithKFNumTh;
#else
    const int kPNPInliersThresholdGood =
        Parameters::GetInstance().kDetectorPnPInliersGoodNumTh;
    const int kPNPInliersThresholdUnreliable =
        Parameters::GetInstance().kDetectorPnPInliersUnreliableNumTh;
#endif

    if (pnp_solver_result_ && pnp_inliers_num_ >= kPNPInliersThresholdGood) {
        VLOG(5) << "detector PnP inlier num success:" << pnp_inliers_num_;
        detect_state_ = DetectionGood;
        VLOG(5) << "detectionObjPosePoseR: " << Rco_cur_;
        VLOG(5) << "detectionObjPosePoseT: " << tco_cur_;
        VLOG(5) << "detectionWOPosePoseR: " << Rwo_cur_;
        VLOG(5) << "detectionWOPosePoseT: " << two_cur_;
        mObj->SetPose(
            m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
            detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
    } else {
        if (pnp_inliers_num_ >= kPNPInliersThresholdUnreliable) {
            VLOG(5) << "detector PnP fail but has enough inliers";
            detect_state_ = DetectionUnreliable;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        } else {
            VLOG(5) << "detector PnP solve fail!";
            detect_state_ = DetectionBad;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        }
    }
}

void PointCloudObjDetector::ShowDetectResult() {
#ifdef MOBILE_PLATFORM
    return;
#endif

    const Eigen::Isometry3f T =
        ObjDetectionCommon::GetTMatrix(Rco_cur_, tco_cur_);
    cv::Mat imgRGB = m_frame_cur->m_raw_image.clone();
    const std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    const std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    std::vector<Eigen::Vector3d> pointBoxs; // the 8 point of box
    pointBoxs.resize(8);

    const cv::Mat &cameraMatrix = CameraIntrinsic::GetInstance().GetCVK();
    std::vector<cv::Point3f> point3f;
    std::vector<cv::Point2f> point2f;
    std::vector<cv::Point2f> imagePoint2f;
    cv::Point2f tempPoint2f;
    Eigen::Vector3f tempPoint3f;
    Eigen::Vector3f tempCameraPoint3f;

    if (detect_state_ == DetectionGood) {
        ObjDetectionCommon::GetBoxPoint(mObj, pointBoxs);
        ObjDetectionCommon::DrawBox(imgRGB, T, pointBoxs);
    }

    std::string matchTxt = "keyPoints 3dmatch num:" +
                           std::to_string(m_frame_cur->m_matches_3d.size());
    std::string inlierTxt =
        "keyPoints 3dinliers num: " +
        std::to_string(m_frame_cur->m_matches2dto3d_inliers.size());
    cv::putText(
        imgRGB, matchTxt, cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 0, 0), 1, 1, 0);
    cv::putText(
        imgRGB, inlierTxt, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 0, 0), 1, 1, 0);

    std::string detectionStateString;
    if (detect_state_ == DetectionGood) {
        detectionStateString = "DetectionGood";

    } else if (detect_state_ == DetectionUnreliable) {
        detectionStateString = "DetectionUnreliable";
    } else {
        detectionStateString = "DetectionBad";
    }
    cv::putText(
        imgRGB, detectionStateString, cv::Point(10, 50),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, 1, 0);
    std::vector<cv::KeyPoint> matcheskeyPointsShow;
    std::vector<MapPointIndex> mapPointId;

    for (auto iter = m_frame_cur->m_matches_3d.begin();
         iter != m_frame_cur->m_matches_3d.end(); iter++) {
        matcheskeyPointsShow.emplace_back(keyPoints[iter->first]);
        mapPointId.emplace_back(iter->second);
    }
    drawKeypoints(imgRGB, matcheskeyPointsShow, imgRGB, cv::Scalar(0, 0, 255));

    std::vector<cv::KeyPoint> matcheskeyPointsInliersShow;
    for (auto iter = m_frame_cur->m_matches2dto3d_inliers.begin();
         iter != m_frame_cur->m_matches2dto3d_inliers.end(); iter++) {
        matcheskeyPointsInliersShow.emplace_back(keyPoints[iter->first]);
    }
    drawKeypoints(
        imgRGB, matcheskeyPointsInliersShow, imgRGB, cv::Scalar(0, 255, 255));
    // GlobalPointCloudMatchViewer::SetMatchedMapPoint(mapPointId);
    // GlobalOcvViewer::UpdateView("ObjDetectorResult", imgRGB);
}

// detect algorithm start
void PointCloudObjDetector::Process(
    const std::shared_ptr<ObjRecognition::FrameData> &frm) {
    if (frm == nullptr || (frm != nullptr && frm->img.data == nullptr) ||
        mObj == nullptr) {
        return;
    }

    std::cout << "detector process start:" << std::endl;
    // STSLAMCommon::Timer timer("PointCloud detector process");

    PreProcess(frm);

    std::vector<PS::MatchSet2D> matchset_2d;
#ifdef OBJ_WITH_KF
    auto allKFs = mObj->GetKeyFrames();
    matchset_2d = Find2DMatches(allKFs);
#endif
    PS::MatchSet3D matchset_3d = Find3DMatch();

    std::vector<int> inliers_3d;
    std::vector<std::vector<int>> inliers_2d;
    PoseSolver(matchset_3d, matchset_2d, inliers_3d, inliers_2d);

    PnPResultHandle();

    // VLOG(10) << "PointCloud detector process time: " << timer.Stop();

    ShowDetectResult();

    ResultRecord();

    SetInfo();
}

void PointCloudObjDetector::Reset() {
    Clear();
    VLOG(10) << "PointCloudObjDetector::Reset";
}

void PointCloudObjDetector::Clear() {
    knn_match_num_ = 0;
    pnp_inliers_num_ = 0;
    detect_state_ = DetectionBad;
    VLOG(10) << "PointCloudObjDetector::Clear";
}

bool PointCloudObjDetector::Load(const int &mem_size, const char *mem) {
    VLOG(10) << "PointCloudObjDetector::Load";
    return mObj->LoadPointCloud(mem_size, mem);
}

bool PointCloudObjDetector::Save(int &mem_size, char **mem) {
    VLOG(10) << "PointCloudObjDetector::Save";
    return mObj->Save(mem_size, mem);
}

void PointCloudObjDetector::ResultRecord() {

    if (detect_state_ == DetectionGood) {
        Eigen::Matrix3d Rcw;
        Eigen::Vector3d tcw;
        m_frame_cur->GetCameraPose(Rcw, tcw);
        /*STSLAMCommon::StatsCollector pointCloudDetectionNum(
            "pointCloud detection good num");
        pointCloudDetectionNum.IncrementOne();
        GlobalSummary::AddPose(
            "detector_camera_pose",
            {m_frame_cur->m_time_stamp, {Eigen::Quaterniond(Rcw), tcw}});
        GlobalSummary::AddPose(
            "detector_obj_pose", {m_frame_cur->m_time_stamp,
                                  {Eigen::Quaterniond(Rco_cur_), tco_cur_}});*/
    }
}

void PointCloudObjDetector::SetInfo() {

    info_.clear();
    switch (detect_state_) {
    case DetectionGood:
        info_ += "detector state: good";
        break;
    case DetectionBad:
        info_ += "detector state: bad";
        break;
    case DetectionUnreliable:
        info_ += "detector state: unreliable";
        break;
    default:
        info_ += "detector state: unknow";
    }
    info_ += '\n';
    info_ +=
        "detector knn match size: " + std::to_string(knn_match_num_) + '\n';
    info_ += "detector pnp inliers num: " + std::to_string(pnp_inliers_num_) +
             " = " + std::to_string(pnp_inliers_3d_num_) + " + " +
             std::to_string(pnp_inliers_2d_num_) + '\n';
}

int PointCloudObjDetector::GetInfo(std::string &info) {

    info += info_;
    return 0;
}

} // namespace ObjRecognition
