//
// Created by zhangye on 2020/9/16.
//
#include <opencv2/imgproc.hpp>
#include <cv.hpp>
#include <glog/logging.h>
#include "Visualizer/GlobalImageViewer.h"
#include "Tracker/TrackerFrame.h"
#include "Tracker/TrackerPointCloud.h"
#include "Utility/Parameters.h"
#include "Utility/Camera.h"
#include "Utility/Statistics.h"
#include "Tracker/TrackerCommon.h"
#include "mode.h"

namespace ObjRecognition {

PointCloudObjTracker::PointCloudObjTracker() {
    Clear();
}

PointCloudObjTracker::~PointCloudObjTracker() {
}
void PointCloudObjTracker::SetPointCloudObj(
    const std::shared_ptr<Object> &pointCloudPtr) {
    mObj = pointCloudPtr;
}

static PS::Point2D NormalizePoint2D(
    const cv::Point2d &pt, const float &fx, const float &fy, const float &cx,
    const float &cy) {
    PS::Point2D p2d;
    p2d(0) = (pt.x - cx) / fx;
    p2d(1) = (pt.y - cy) / fy;
    return std::move(p2d);
}

void PointCloudObjTracker::ShowProjectedPointsAndMatchingKeyPoints(
    std::vector<Eigen::Vector2d> &projectPoints,
    std::vector<bool> &matchKeyPointsState) {

    cv::Mat showProjectedKeyPoints;
    cv::Mat imageCur = m_frame_cur->m_raw_image;
    cv::cvtColor(imageCur, showProjectedKeyPoints, cv::COLOR_GRAY2BGR);
    // pink points keypoints
    for (auto iter = m_projection_matches2dTo3d_cur.begin();
         iter != m_projection_matches2dTo3d_cur.end(); iter++) {
        cv::drawMarker(
            showProjectedKeyPoints, m_frame_cur->m_kpts[iter->first].pt,
            cv::Scalar(255, 0, 255));
    }

    //  blue points projected points
    for (int i = 0; i < projectPoints.size(); i++) {
        if (matchKeyPointsState[i])
            cv::drawMarker(
                showProjectedKeyPoints,
                cv::Point2d(projectPoints[i](0), projectPoints[i](1)),
                cv::Scalar(255, 0, 70));
    }

    // GlobalOcvViewer::UpdateView(
    //"projected cloud and matching keypoints", showProjectedKeyPoints);
}

PS::MatchSet3D Generate3DMatch(
    const TrackerFrame::Ptr frame,
    const std::map<int, MapPointIndex> &matches2dTo3d,
    const std::vector<cv::Point2d> &keyPoints,
    const std::vector<Eigen::Vector3d> &pointClouds3dObj) {
    PS::MatchSet3D matchset_3d;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    float fx = static_cast<float>(Kcv.at<double>(0, 0));
    float fy = static_cast<float>(Kcv.at<double>(1, 1));
    float cx = static_cast<float>(Kcv.at<double>(0, 2));
    float cy = static_cast<float>(Kcv.at<double>(1, 2));

    frame->m_projection_matches3d_vec.clear();
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        PS::Point3D point_3d(
            pointClouds3dObj[iter->second](0),
            pointClouds3dObj[iter->second](1),
            pointClouds3dObj[iter->second](2));
        PS::Point2D point_2d =
            NormalizePoint2D(keyPoints[iter->first], fx, fy, cx, cy);
        PS::Match3D match_3d(point_3d, point_2d);
        matchset_3d.push_back(std::move(match_3d));
        frame->m_projection_matches3d_vec.emplace_back(
            iter->first, iter->second);
    }

    return std::move(matchset_3d);
}

void PointCloudObjTracker::PreProcess(
    const std::shared_ptr<ObjRecognition::FrameData> &frm) {

    m_frame_cur = std::make_shared<TrackerFrame>();
    m_frame_cur->m_time_stamp = frm->mTimeStamp;
    m_frame_cur->m_frame_index = frm->mFrmIndex;
    m_frame_cur->m_raw_image = frm->img.clone();
    m_frame_cur->m_desp = frm->mDesp.clone();
    m_frame_cur->m_kpts = frm->mKpts;
    m_frame_cur->SetCameraPose(frm->mRcw, frm->mTcw);

    Rcw_cur_ = frm->mRcw;
    tcw_cur_ = frm->mTcw;
}

PS::MatchSet3D OpticalFlowGenerate3DMatch(
    const std::shared_ptr<TrackerFrame> &frame,
    const std::vector<Eigen::Vector3d> &pointClouds3dObj) {
    PS::MatchSet3D matchset_3d;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    auto fx = static_cast<float>(Kcv.at<double>(0, 0));
    auto fy = static_cast<float>(Kcv.at<double>(1, 1));
    auto cx = static_cast<float>(Kcv.at<double>(0, 2));
    auto cy = static_cast<float>(Kcv.at<double>(1, 2));

    // 2d pos, 3d pos
    frame->m_opticalflow_matches3d_vec.clear();
    for (const auto &it : frame->m_opticalflow_matches2dto3d) {
        PS::Point3D point_3d(
            pointClouds3dObj[it.second](0), pointClouds3dObj[it.second](1),
            pointClouds3dObj[it.second](2));
        PS::Point2D point_2d = NormalizePoint2D(
            frame->m_opticalflow_point2ds[it.first], fx, fy, cx, cy);
        PS::Match3D match_3d(point_3d, point_2d);
        matchset_3d.push_back(match_3d);
        frame->m_opticalflow_matches3d_vec.emplace_back(it.first, it.second);
    }
    return std::move(matchset_3d);
}

template <class T1, class T2>
void ReduceVector(std::vector<T1> &elements, const std::vector<T2> &status) {
    int j = 0;
    for (int i = 0; i < elements.size(); i++) {
        if (status[i])
            elements[j++] = elements[i];
    }
    elements.resize(j);
}

void PointCloudObjTracker::OpticalFlowRejectWithF(
    std::vector<cv::Point2d> &ptsPre,
    std::vector<MapPointIndex> &mapPointIndexes) {
    std::vector<uchar> status;
    // points far from the epipolar line
    const float kFThreshold = 1.0;
    cv::Mat p1(m_frame_cur->m_opticalflow_point2ds.size(), 2, CV_32F);
    cv::Mat p2(m_frame_cur->m_opticalflow_point2ds.size(), 2, CV_32F);
    for (int i = 0; i < m_frame_cur->m_opticalflow_point2ds.size(); i++) {
        p1.at<float>(i, 0) =
            static_cast<float>(m_frame_cur->m_opticalflow_point2ds[i].x);
        p1.at<float>(i, 1) =
            static_cast<float>(m_frame_cur->m_opticalflow_point2ds[i].y);
        p2.at<float>(i, 0) = static_cast<float>(ptsPre[i].x);
        p2.at<float>(i, 1) = static_cast<float>(ptsPre[i].y);
    }
    cv::findFundamentalMat(p1, p2, cv::FM_RANSAC, kFThreshold, 0.99, status);
    if (status.empty()) {
        ptsPre.clear();
        m_frame_cur->m_opticalflow_point2ds.clear();
        mapPointIndexes.clear();
    }
    ReduceVector(ptsPre, status);
    ReduceVector(m_frame_cur->m_opticalflow_point2ds, status);
    ReduceVector(mapPointIndexes, status);
}

void PointCloudObjTracker::ShowOpticalFlowpoints(
    const std::vector<cv::Point2d> &opticalFlowKeyPointsPre) {

    cv::Mat imagePre = m_frame_Pre->m_raw_image;
    cv::Mat imageCur = m_frame_cur->m_raw_image;
    cv::Mat img_match;
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> opPre;
    std::vector<cv::KeyPoint> opCur;
    for (int i = 0; i < opticalFlowKeyPointsPre.size(); i++) {
        matches.emplace_back(i, i, -1);
        opPre.emplace_back(cv::KeyPoint(opticalFlowKeyPointsPre[i], 1.f));
        opCur.emplace_back(
            cv::KeyPoint(m_frame_cur->m_opticalflow_point2ds[i], 1.f));
    }

    cv::drawMatches(imagePre, opPre, imageCur, opCur, matches, img_match);
    GlobalOcvViewer::UpdateView("optical match", img_match);
}

void PointCloudObjTracker::RemoveOpticalFlow3dMatchOutliers(
    const std::vector<uchar> &status,
    const std::vector<cv::Point2f> &points_cur) {

    std::vector<MapPointIndex> mapPointMatchesIndices;
    std::vector<cv::Point2d> OpticalFlowPointsPreMatches;

    OpticalFlowPointsPreMatches.reserve(
        m_frame_Pre->m_opticalflow_point2ds.size());
    mapPointMatchesIndices.reserve(m_frame_Pre->m_opticalflow_point2ds.size());

    for (int i = 0, j = 0; i < m_frame_Pre->m_opticalflow_point2ds.size();
         i++) {
        if (status[j]) {
            auto iter = m_frame_Pre->m_opticalflow_matches2dto3d.find(i);
            if (iter != m_frame_Pre->m_opticalflow_matches2dto3d.end()) {
                mapPointMatchesIndices.emplace_back(iter->second);
                OpticalFlowPointsPreMatches.emplace_back(
                    m_frame_Pre->m_opticalflow_point2ds[i]);
                m_frame_cur->m_opticalflow_point2ds.emplace_back(
                    (cv::Point2d)points_cur[j]);
            }
        }
        j++;
    }

    if (OpticalFlowPointsPreMatches.empty() ||
        m_frame_cur->m_opticalflow_point2ds.empty()) {
        return;
    }

    OpticalFlowRejectWithF(OpticalFlowPointsPreMatches, mapPointMatchesIndices);

    for (int i = 0; i < mapPointMatchesIndices.size(); i++) {
        m_frame_cur->m_opticalflow_matches2dto3d[i] = mapPointMatchesIndices[i];
    }
}

PS::MatchSet3D PointCloudObjTracker::FindOpticalFlow3DMatch() {
    PS::MatchSet3D matchset_3d;
    if (m_frame_Pre->m_opticalflow_matches2dto3d.empty() ||
        m_frame_Pre->m_opticalflow_point2ds.empty()) {
        return matchset_3d;
    }

    const std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    if (pointClouds.empty()) {
        LOG(ERROR) << "No mapPoints here!!!";
        return matchset_3d;
    }

    // STSLAMCommon::Timer timer("PointCloud tracker find opticalFlow match");
    m_frame_cur->m_opticalflow_matches2dto3d.clear();
    m_frame_cur->m_opticalflow_point2ds.clear();

    FrameIndex frmIndex = m_frame_cur->m_frame_index;
    double timeStamp;
    ObjRecogState state;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    mObj->GetPose(frmIndex, timeStamp, state, Rcw, tcw, Rwo_cur_, two_cur_);
    const Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();

    std::vector<Eigen::Vector3d> mapPointsObj;
    std::vector<Eigen::Vector3d> mapPointsWorld;
    ObjTrackerCommon::GetMapPointPositions(
        pointClouds, Rwo_cur_, two_cur_, mapPointsObj, mapPointsWorld);

    // opticalflow 2d point position of previous frame
    std::vector<cv::Point2f> opticalFlowKeyPointsPreFloat;
    opticalFlowKeyPointsPreFloat.reserve(
        m_frame_Pre->m_opticalflow_point2ds.size());

    // opticalflow need cv::point2f data
    for (const auto &it : m_frame_Pre->m_opticalflow_point2ds) {
        opticalFlowKeyPointsPreFloat.push_back(static_cast<cv::Point2f>(it));
    }

    std::vector<uchar> opticalFlowStatus;
    std::vector<float> err;
    // opticalflow 2d point position of current frame using opticalflow
    std::vector<cv::Point2f> opticalKeyPointsCur;
    cv::Mat imagePre = m_frame_Pre->m_raw_image;
    cv::Mat imageCur = m_frame_cur->m_raw_image;
    cv::calcOpticalFlowPyrLK(
        imagePre, imageCur, opticalFlowKeyPointsPreFloat, opticalKeyPointsCur,
        opticalFlowStatus, err, cv::Size(21, 21), 3);

    // get current frame opticalflow position, and correspondance to mappoint id
    RemoveOpticalFlow3dMatchOutliers(opticalFlowStatus, opticalKeyPointsCur);

    m_match_points_opticalFlow_num =
        m_frame_cur->m_opticalflow_matches2dto3d.size();
    STATISTICS_UTILITY::StatsCollector stats_collector_opt(
        "tracker opticalFlow match 2d-3d num");
    stats_collector_opt.AddSample(m_match_points_opticalFlow_num);
    // ShowOpticalFlowpoints(m_frame_cur->m_opticalflow_point2ds);

    // get currentframe 2d<->3d correspondance
    matchset_3d = OpticalFlowGenerate3DMatch(m_frame_cur, mapPointsObj);

    // VLOG(20) << "PointCloud tracker opticalFlowMatch process time: "
    //<< timer.Stop();
    return matchset_3d;
}

void RemoveOpticalFlowAndProjectionCommonMatch(
    const TrackerFrame::Ptr &frame_cur,
    std::map<int, MapPointIndex> &projection_match) {

    std::map<MapPointIndex, int> projection_mappoint_id;
    for (const auto &it : projection_match) {
        projection_mappoint_id.insert({it.second, it.first});
    }
    for (const auto &it : frame_cur->m_opticalflow_matches2dto3d) {
        auto repeat = projection_mappoint_id.find(it.second);
        if (repeat != projection_mappoint_id.end()) {
            projection_match.erase(repeat->second);
        }
    }

    //    std::set<MapPointIndex> mappoint_id;
    //    for (const auto &it : projection_match) {
    //        mappoint_id.insert(it.second);
    //    }
    //    for (const auto &it : frame_cur->m_opticalflow_matches2dto3d) {
    //        CHECK(mappoint_id.find(it.second) == mappoint_id.end());
    //    }
}

PS::MatchSet3D PointCloudObjTracker::FindProjection3DMatch() {
    PS::MatchSet3D matchset_3d;
    const std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    if (pointClouds.empty()) {
        LOG(ERROR) << "No pointclouds model here!";
        return matchset_3d;
    }

    FrameIndex frmIndex = m_frame_cur->m_frame_index;
    double timeStamp;
    ObjRecogState state;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    mObj->GetPose(frmIndex, timeStamp, state, Rcw, tcw, Rwo_cur_, two_cur_);

    std::vector<Eigen::Vector3d> mapPointsObj;
    std::vector<Eigen::Vector3d> mapPointsWorld;
    ObjTrackerCommon::GetMapPointPositions(
        pointClouds, Rwo_cur_, two_cur_, mapPointsObj, mapPointsWorld);

    const std::vector<cv::KeyPoint> keyPointsOrigin = m_frame_cur->m_kpts;
    const cv::Mat descriptors = m_frame_cur->m_desp;
    m_projection_points2d_cur.clear();
    m_projection_points2d_cur.reserve(keyPointsOrigin.size());
    ObjTrackerCommon::KeyPointsToPoints(
        keyPointsOrigin, m_projection_points2d_cur);

    std::vector<bool> projectFailState =
        std::vector<bool>(mapPointsWorld.size(), false);
    std::vector<Eigen::Vector2d> projectPoints;
    projectPoints.reserve(mapPointsWorld.size());

    ObjTrackerCommon::ProjectSearch(
        mapPointsWorld, Rcw_cur_, tcw_cur_, projectFailState, projectPoints,
        false);
    m_project_success_mappoint_num = projectPoints.size();

    STATISTICS_UTILITY::StatsCollector stats_collector_project(
        "tracker mappoint project to image num");
    stats_collector_project.AddSample(m_project_success_mappoint_num);

    // 50
    if (m_project_success_mappoint_num <
        Parameters::GetInstance().kTrackerProjectSuccessNumTh) {
        return matchset_3d;
    }

    std::vector<bool> matchKeyPointsState;
    m_projection_matches2dTo3d_cur.clear();

#ifdef SUPERPOINT
    // TODO(zhangye): use only norm2 for projection search?
    ObjTrackerCommon::SearchByProjection_Superpoint(
        projectPoints, pointClouds, projectFailState, keyPointsOrigin,
        descriptors, matchKeyPointsState, m_projection_matches2dTo3d_cur);
#else
    ObjTrackerCommon::SearchByProjection(
        projectPoints, pointClouds, projectFailState, keyPointsOrigin,
        descriptors, matchKeyPointsState, m_projection_matches2dTo3d_cur);
#endif

    RemoveOpticalFlowAndProjectionCommonMatch(
        m_frame_cur, m_projection_matches2dTo3d_cur);

    m_match_points_projection_num = m_projection_matches2dTo3d_cur.size();
    m_match_points_num =
        m_match_points_projection_num + m_match_points_opticalFlow_num;

    STATISTICS_UTILITY::StatsCollector stats_projection_match(
        "tracker projection 2D-3D match num");
    stats_projection_match.AddSample(m_projection_matches2dTo3d_cur.size());

    matchset_3d = Generate3DMatch(
        m_frame_cur, m_projection_matches2dTo3d_cur, m_projection_points2d_cur,
        mapPointsObj);

    // ShowProjectedPointsAndMatchingKeyPoints(projectPoints,
    // matchKeyPointsState);
    return matchset_3d;
}

bool PointCloudObjTracker::PoseSolver(
    const PS::MatchSet3D &matches_3d,
    const std::vector<PS::MatchSet2D> &matches_2d,
    std::vector<int> &inliers_3d) {

    m_pnp_solver_result = false;
    m_pnp_inliers_num = 0;
    // 40
    if (matches_3d.size() <
        Parameters::GetInstance().kTrackerMatchPointsNumTh) {
        VLOG(5) << "Not enough opticalFlow 3d mathces";
        return false;
    }

    // STSLAMCommon::Timer trackingPoseSolverTime(
    //"tracking opticalFlow poseSolver process");
    PS::Options options;
    PS::Pose T;
    inliers_3d.clear();
    std::vector<std::vector<int>> inliers_2d;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();

    const float kPnpReprojectionError = 6.5;
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, 1.0);
    Eigen::Vector3d gravityCamera = Rcw_cur_ * gravity;

    options.focal_length = static_cast<float>(Kcv.at<double>(0, 0));
    options.max_reproj_err = kPnpReprojectionError / options.focal_length;
    options.enable_2d_solver = false;
    options.enable_3d_solver = true;
    options.ransac_iterations = 100;
    options.ransac_confidence = 0.90;
    options.gravity_dir = gravityCamera.cast<float>();
    options.gravity_dir_max_err_deg = 180;
    options.enable_gravity_solver = true;
    options.prefer_pure_2d_solver = false;
    options.try_refine_translation_before_optimization_for_2d_only_matches =
        false;

    const int kPnpMinMatchesNum = 0;
    // 80
    const int kPnpMinInlierNum =
        Parameters::GetInstance().kTrackerPnPInliersGoodNumTh;
    const double kPnpMinInlierRatio = 0.0;
    options.callbacks.emplace_back(PS::EarlyBreakBy3DInlierCounting(
        kPnpMinMatchesNum, kPnpMinInlierNum, kPnpMinInlierRatio));
    options.CheckValidity();

    m_pnp_solver_result = PS::Ransac(
        options, matches_3d, matches_2d, &T, &inliers_3d, &inliers_2d);

    Eigen::Matrix3d Rco = T.m_R.cast<double>();
    Eigen::Vector3d tco = T.m_t.cast<double>();

    Eigen::Matrix3d Row = Rco.transpose() * (Rcw_cur_);
    Eigen::Vector3d tow = Rco.transpose() * (tcw_cur_ - tco);
    Rwo_cur_ = Row.transpose();
    two_cur_ = -Rwo_cur_ * tow;
    Rco_cur_ = Rco;
    tco_cur_ = tco;

    m_pnp_inliers_num = inliers_3d.size();
    // VLOG(20) << "tracking opticalFlow poseSolver process time: "
    //<< trackingPoseSolverTime.Stop();
    STATISTICS_UTILITY::StatsCollector stats_collector_pnp(
        "tracker pnp inlier num");
    stats_collector_pnp.AddSample(m_pnp_inliers_num);
    return m_pnp_solver_result;
}

void PointCloudObjTracker::PnPResultHandle() {
    FrameIndex frame_index;
    double time_stamp;
    ObjRecogState state;
    Eigen::Matrix3d Rcw_old = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rwo_old = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Tcw_old = Eigen::Vector3d::Zero();
    Eigen::Vector3d Two_old = Eigen::Vector3d::Zero();

    mObj->GetPose(
        frame_index, time_stamp, state, Rcw_old, Tcw_old, Rwo_old, Two_old);
    Rwo_cur_ = Rwo_old;
    two_cur_ = Two_old;

#ifdef USE_INLIER
    const int KTrackerPnPInliersGoodTh =
        Parameters::GetInstance().kTrackerPnPInliersGoodNumTh;
    const int KTrackerPnP3DInliersGoodTh =
        Parameters::GetInstance().kTrackerPnP3DInliersGoodNumTh;
    const int KTrackerPnPInliersUnreliableTh =
        Parameters::GetInstance().kTrackerPnPInliersUnreliableNumTh;

    if (Rwo_cur_ == Eigen::Matrix3d::Identity() &&
        two_cur_ == Eigen::Vector3d::Zero()) {
        m_tracker_state = TrackingBad;
    } else {
        // 80
#ifdef SUPERPOINT
        int proj_success_num = 120;
#else
        int proj_success_num = m_projection_matches2dTo3d_cur.size() * 0.13;
#endif
        if (m_pnp_solver_result &&
            m_pnp_inliers_num > KTrackerPnPInliersGoodTh &&
            m_pnp_inliers_projection_num > proj_success_num) {
            VLOG(5) << "tracker pnp inlier num success:" << m_pnp_inliers_num;
            m_tracker_state = TrackingGood;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                m_tracker_state, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);

            if (reproj_error >= 0) {
                // VLOG(0) << "tracker reproj error: " << reproj_error;
                STATISTICS_UTILITY::StatsCollector stats_collector_project(
                    "tracker inlier reproj error");
                stats_collector_project.AddSample(reproj_error);
            }

        } else if (m_pnp_inliers_num > KTrackerPnPInliersUnreliableTh) {
            VLOG(5) << "PNP fail but has enough inliers";
            m_tracker_state = TrackingUnreliable;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                m_tracker_state, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        } else {
            VLOG(5) << "PNP solve fail!";
            m_tracker_state = TrackingBad;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                m_tracker_state, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        }
        if (m_tracker_state == TrackingGood) {
            m_frame_Pre = m_frame_cur;
        } else {
            m_frame_Pre = std::make_shared<TrackerFrame>();
        }
    }
#endif

#ifdef USE_REPROJ
    if (Rwo_cur_ == Eigen::Matrix3d::Identity() &&
        two_cur_ == Eigen::Vector3d::Zero()) {
        m_tracker_state = TrackingBad;
    } else {
        // 80
        if (m_pnp_solver_result && reproj_error <= 3.0) {
            VLOG(5) << "tracker pnp inlier num success:" << m_pnp_inliers_num;
            m_tracker_state = TrackingGood;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                m_tracker_state, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        } else if (reproj_error <= 4.5) {
            VLOG(5) << "PNP fail but has enough inliers";
            m_tracker_state = TrackingUnreliable;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                m_tracker_state, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        } else {
            VLOG(5) << "PNP solve fail!";
            m_tracker_state = TrackingBad;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                m_tracker_state, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        }
        if (m_tracker_state == TrackingGood) {
            m_frame_Pre = m_frame_cur;
        } else {
            m_frame_Pre = std::make_shared<TrackerFrame>();
        }
    }
#endif
}

void PointCloudObjTracker::ResultRecord() {
    if (m_tracker_state != TrackingBad) {
        if (m_tracker_state == TrackingGood) {
            STATISTICS_UTILITY::StatsCollector pointCloudTrackingNum(
                "tracker good num");
            pointCloudTrackingNum.IncrementOne();
        }
    }
}

void PointCloudObjTracker::ProcessPoseSolverInliers(
    const std::vector<int> &inliers_3d) {
    m_projection_matches2dTo3d_inlier.clear();
    m_opticalFlow_matches2dTo3d_inlier.clear();
    m_pnp_inliers_opticalFlow_num = 0;
    m_pnp_inliers_projection_num = 0;
    if (inliers_3d.empty()) {
        return;
    }

    for (const auto &it : inliers_3d) {
        if (it >= m_frame_cur->m_opticalflow_matches2dto3d.size()) {
            auto matche3d = m_frame_cur->m_projection_matches3d_vec.at(
                it - m_frame_cur->m_opticalflow_matches2dto3d.size());
            m_projection_matches2dTo3d_inlier.insert(
                {matche3d.first, matche3d.second});
        } else {
            auto matche3d = m_frame_cur->m_opticalflow_matches3d_vec.at(it);
            m_opticalFlow_matches2dTo3d_inlier.insert(
                {matche3d.first, matche3d.second});
        }
    }

    m_pnp_inliers_opticalFlow_num = m_opticalFlow_matches2dTo3d_inlier.size();
    m_pnp_inliers_projection_num = m_projection_matches2dTo3d_inlier.size();

    std::vector<cv::Point2d> opticalflow_point2ds_tmp =
        m_frame_cur->m_opticalflow_point2ds;

    m_frame_cur->m_opticalflow_point2ds.clear();
    m_frame_cur->m_opticalflow_matches2dto3d.clear();

    int point2d_index = 0;
    for (const auto &it : m_opticalFlow_matches2dTo3d_inlier) {
        m_frame_cur->m_opticalflow_point2ds.emplace_back(
            opticalflow_point2ds_tmp[it.first]);
        m_frame_cur->m_opticalflow_matches2dto3d[point2d_index] = it.second;
        point2d_index++;
    }
    for (const auto &it : m_projection_matches2dTo3d_inlier) {
        m_frame_cur->m_opticalflow_point2ds.emplace_back(
            m_projection_points2d_cur[it.first]);
        m_frame_cur->m_opticalflow_matches2dto3d[point2d_index] = it.second;
        point2d_index++;
    }

    STATISTICS_UTILITY::StatsCollector stats_projInlier(
        "trakcer projection inlier num");
    stats_projInlier.AddSample(m_projection_matches2dTo3d_inlier.size());
    STATISTICS_UTILITY::StatsCollector stats_optInlier(
        "tracker opticalflow inlier num");
    stats_optInlier.AddSample(m_opticalFlow_matches2dTo3d_inlier.size());
}

void PointCloudObjTracker::DrawTextInfo(const cv::Mat &img, cv::Mat &img_txt) {
    std::vector<cv::KeyPoint> projectionMatch;
    for (const auto &it : m_projection_matches2dTo3d_cur) {
        projectionMatch.emplace_back(m_frame_cur->m_kpts[it.first]);
    }
    std::string match_txt =
        "proj kp match num:" + std::to_string(projectionMatch.size()) + "| ";
    std::string inlier_txt =
        "proj kp inliers num: " + std::to_string(m_pnp_inliers_projection_num) +
        "| ";
    std::string reproj_txt = "reproj error: " + std::to_string(reproj_error);

    std::vector<cv::KeyPoint> opticalFlowMatch;
    std::vector<cv::KeyPoint> opticalFlowInliers;
    for (const auto &it : m_frame_cur->m_opticalflow_matches2dto3d) {
        opticalFlowMatch.emplace_back(
            cv::KeyPoint(m_frame_cur->m_opticalflow_point2ds[it.first], 1.0f));
    }

    for (const auto &it : m_opticalFlow_matches2dTo3d_inlier) {
        opticalFlowInliers.emplace_back(
            cv::KeyPoint(m_frame_cur->m_opticalflow_point2ds[it.first], 1.0));
    }

    std::string opMatch_txt =
        "opt kp match num:" + std::to_string(opticalFlowMatch.size()) + "| ";
    std::string opInlier_txt =
        "opt kp inliers num: " + std::to_string(m_pnp_inliers_opticalFlow_num) +
        "| ";

    std::string trackingStateString;
    if (m_tracker_state == TrackingGood) {
        trackingStateString = "Good";
    } else if (m_tracker_state == TrackingUnreliable) {
        trackingStateString = "Unreliable";
    } else {
        trackingStateString = "Bad";
    }

    std::stringstream s;
    s << match_txt;
    s << inlier_txt;
    s << reproj_txt;
    std::stringstream s1;
    s1 << opMatch_txt;
    s1 << opInlier_txt;
    s1 << trackingStateString;

    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    img_txt = cv::Mat(img.rows + textSize.height + 23, img.cols, img.type());
    img.copyTo(img_txt.rowRange(0, img.rows).colRange(0, img.cols));
    img_txt.rowRange(img.rows, img_txt.rows) =
        cv::Mat::zeros(textSize.height + 23, img.cols, img.type());
    cv::putText(
        img_txt, s.str(), cv::Point(5, img_txt.rows - 17),
        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);

    cv::putText(
        img_txt, s1.str(), cv::Point(5, img_txt.rows - 5),
        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void PointCloudObjTracker::ShowTrackerResult() {
    std::vector<Eigen::Vector3d> mapPointBoundingBox;
    mapPointBoundingBox.reserve(8);
    ObjTrackerCommon::GetPointCloudBoundingBox(mObj, mapPointBoundingBox);

    cv::Mat showResult;
    cv::Mat imageCur = m_frame_cur->m_raw_image;
    cv::cvtColor(imageCur, showResult, cv::COLOR_GRAY2BGR);
    std::vector<cv::Point2d> boxProjResult;
    for (int i = 0; i < mapPointBoundingBox.size(); i++) {
        const Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();
        Eigen::Vector3d p = K * (Rco_cur_ * mapPointBoundingBox[i] + tco_cur_);
        cv::Point2d pResult;
        pResult.x = p(0) / p(2);
        pResult.y = p(1) / p(2);
        boxProjResult.emplace_back(pResult);
    }

    cv::Scalar corner_color = cv::Scalar(0, 255, 255);
    cv::Scalar edge_color;
    if (m_tracker_state == TrackingGood) {
        edge_color = cv::Scalar(224, 24, 255);
    } else {
        edge_color = cv::Scalar(0, 0, 0);
    }

    for (int i = 0; i < boxProjResult.size(); i++) {
        if (i < 4)
            cv::drawMarker(showResult, boxProjResult[i], corner_color);
        else
            cv::drawMarker(showResult, boxProjResult[i], corner_color);
    }

    ObjTrackerCommon::DrawBoundingBox(showResult, boxProjResult, edge_color);

    // yellow circle projected inliers
    std::vector<cv::KeyPoint> projectionInliers;
    for (const auto &it : m_projection_matches2dTo3d_inlier) {
        projectionInliers.emplace_back(m_frame_cur->m_kpts[it.first]);
    }
    cv::drawKeypoints(
        showResult, projectionInliers, showResult, cv::Scalar(0, 255, 255));

    // red circle opticalflow inliers
    std::vector<cv::KeyPoint> opticalFlowInliers;
    for (const auto &it : m_opticalFlow_matches2dTo3d_inlier) {
        opticalFlowInliers.emplace_back(
            cv::KeyPoint(m_frame_cur->m_opticalflow_point2ds[it.first], 1.0));
    }
    cv::drawKeypoints(
        showResult, opticalFlowInliers, showResult, cv::Scalar(0, 0, 255));

    cv::Mat img_text;
    DrawTextInfo(showResult, img_text);
    GlobalOcvViewer::UpdateView("Tracker Result", img_text);
}

float PointCloudObjTracker::ComputeAverageReProjError(
    const std::vector<int> &inliers_3d) {
    if (m_projection_matches2dTo3d_inlier.size() +
            m_opticalFlow_matches2dTo3d_inlier.size() ==
        0) {
        return -1;
    }

    float average_error = 0;
    for (auto matches : m_projection_matches2dTo3d_inlier) {
        auto mappoint = mObj->GetPointClouds()[matches.second];
        auto keypoint = m_frame_cur->m_kpts[matches.first];

        auto proj = CameraIntrinsic::GetInstance().GetEigenK() *
                    (Rco_cur_ * mappoint->GetPose() + tco_cur_);
        Eigen::Vector2d proj_2d =
            Eigen::Vector2d(proj(0) / proj(2), proj(1) / proj(2));

        float dis = sqrt(
            (proj_2d(0) - keypoint.pt.x) * (proj_2d(0) - keypoint.pt.x) +
            (proj_2d(1) - keypoint.pt.y) * (proj_2d(1) - keypoint.pt.y));

        average_error += dis;
    }

    for (auto matches : m_opticalFlow_matches2dTo3d_inlier) {
        auto mappoint = mObj->GetPointClouds()[matches.second];
        auto keypoint = m_frame_cur->m_opticalflow_point2ds[matches.first];

        auto proj = CameraIntrinsic::GetInstance().GetEigenK() *
                    (Rco_cur_ * mappoint->GetPose() + tco_cur_);
        Eigen::Vector2d proj_2d =
            Eigen::Vector2d(proj(0) / proj(2), proj(1) / proj(2));

        float dis = sqrt(
            (proj_2d(0) - keypoint.x) * (proj_2d(0) - keypoint.x) +
            (proj_2d(1) - keypoint.y) * (proj_2d(1) - keypoint.y));

        average_error += dis;
    }

    average_error = average_error / (m_opticalFlow_matches2dTo3d_inlier.size() +
                                     m_projection_matches2dTo3d_inlier.size());

    return average_error;
}

void PointCloudObjTracker::Process(
    const std::shared_ptr<ObjRecognition::FrameData> &frm) {
    if ((frm == nullptr || (frm != nullptr && frm->img.data == nullptr) ||
         mObj == nullptr)) {
        return;
    }

    // first tracking process when detection is good
    FrameIndex frmIndex;
    double timeStamp;
    ObjRecogState state;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    Eigen::Matrix3d Rwo;
    Eigen::Vector3d two;

    if (!m_first_detection_good) {
        mObj->GetPose(frmIndex, timeStamp, state, Rcw, tcw, Rwo, two);
        if (state != DetectionGood) {
            return;
        } else {
            m_first_detection_good = true;
        }
    }

    // STSLAMCommon::Timer timer("PointCloud tracker process");

    PreProcess(frm);

    PS::MatchSet3D totalMatchSet_3d = FindOpticalFlow3DMatch();

    PS::MatchSet3D matchset_3d = FindProjection3DMatch();
    totalMatchSet_3d.insert(
        totalMatchSet_3d.end(), matchset_3d.begin(), matchset_3d.end());

    std::vector<int> totalInliers_3d;
    PoseSolver(totalMatchSet_3d, {}, totalInliers_3d);

    ProcessPoseSolverInliers(totalInliers_3d);

    reproj_error = ComputeAverageReProjError(totalInliers_3d);

    PnPResultHandle();

    // VLOG(20) << "PointCloud tracker process time: " << timer.Stop();

    ShowTrackerResult();

    ResultRecord();

    // SetInfo();
}

void PointCloudObjTracker::Reset() {
    Clear();
    VLOG(10) << "PointCloudObjTracker::Reset";
}

void PointCloudObjTracker::Clear() {
    m_frame_Pre = std::make_shared<TrackerFrame>();
    m_match_points_num = 0;
    m_project_success_mappoint_num = 0;
    m_pnp_inliers_num = 0;
    m_tracker_state = TrackingBad;
    m_first_detection_good = false;

    VLOG(10) << "PointCloudObjTracker::Clear";
}

bool PointCloudObjTracker::Load(const long long &mem_size, const char *mem) {
    VLOG(30) << "PointCloudObjTracker::Load";
    return mObj->LoadPointCloud(mem_size, mem);
}

bool PointCloudObjTracker::Save(long long &mem_size, char **mem) {
    VLOG(30) << "PointCloudObjTracker::Save";
    return mObj->Save(mem_size, mem);
}

void PointCloudObjTracker::SetInfo() {

    m_info.clear();
    switch (m_tracker_state) {
    case TrackingGood:
        m_info += "tracker state: good";
        break;
    case TrackingBad:
        m_info += "tracker state: bad";
        break;
    case TrackingUnreliable:
        m_info += "tracker state: unreliable";
        break;
    default:
        m_info += "tracker state: unknow";
    }
    m_info += '\n';
    m_info += "tracker project MP num: " +
              std::to_string(m_project_success_mappoint_num) + '\n';
    m_info += "tracker match MP num: " + std::to_string(m_match_points_num) +
              " = " + std::to_string(m_match_points_projection_num) + " + " +
              std::to_string(m_match_points_opticalFlow_num) + '\n';
    m_info += "tracker pnp inliers num: " + std::to_string(m_pnp_inliers_num) +
              " = " + std::to_string(m_pnp_inliers_projection_num) + " + " +
              std::to_string(m_pnp_inliers_opticalFlow_num) + '\n';
}

int PointCloudObjTracker::GetInfo(std::string &info) {

    info += m_info;
    return 0;
}

} // namespace ObjRecognition
