#include <opencv2/imgproc.hpp>
#include <cv.hpp>
#include <glog/logging.h>
#include "Visualizer/GlobalImageViewer.h"
#include "Tracker/TrackerFrame.h"
#include "Tracker/TrackerPointCloud.h"
#include "Utility/Parameters.h"
#include "Utility/Camera.h"
#include "StatisticsResult/Statistics.h"
#include "StatisticsResult/Timer.h"
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

static PoseSolver::Point2D NormalizePoint2D(
    const cv::Point2d &pt, const float &fx, const float &fy, const float &cx,
    const float &cy) {
    PoseSolver::Point2D p2d;
    p2d(0) = (pt.x - cx) / fx;
    p2d(1) = (pt.y - cy) / fy;
    return std::move(p2d);
}

PoseSolver::MatchSet3D Generate3DMatch(
    const TrackerFrame::Ptr frame,
    const std::map<int, MapPointIndex> &matches2dTo3d,
    const std::vector<cv::Point2d> &keyPoints,
    const std::vector<Eigen::Vector3d> &pointClouds3dObj) {
    PoseSolver::MatchSet3D matchset_3d;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    float fx = static_cast<float>(Kcv.at<double>(0, 0));
    float fy = static_cast<float>(Kcv.at<double>(1, 1));
    float cx = static_cast<float>(Kcv.at<double>(0, 2));
    float cy = static_cast<float>(Kcv.at<double>(1, 2));

    frame->m_projection_matches3d_vec.clear();
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        PoseSolver::Point3D point_3d(
            pointClouds3dObj[iter->second](0),
            pointClouds3dObj[iter->second](1),
            pointClouds3dObj[iter->second](2));
        PoseSolver::Point2D point_2d =
            NormalizePoint2D(keyPoints[iter->first], fx, fy, cx, cy);
        PoseSolver::Match3D match_3d(point_3d, point_2d);
        matchset_3d.push_back(std::move(match_3d));
        frame->m_projection_matches3d_vec.emplace_back(
            iter->first, iter->second);
    }

    return std::move(matchset_3d);
}

PoseSolver::MatchSet3D OpticalFlowGenerate3DMatch(
    const std::shared_ptr<TrackerFrame> &frame,
    const std::vector<Eigen::Vector3d> pointClouds3dObj) {
    PoseSolver::MatchSet3D matchset_3d;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    auto fx = static_cast<float>(Kcv.at<double>(0, 0));
    auto fy = static_cast<float>(Kcv.at<double>(1, 1));
    auto cx = static_cast<float>(Kcv.at<double>(0, 2));
    auto cy = static_cast<float>(Kcv.at<double>(1, 2));

    // 2d pos, 3d pos
    frame->m_opticalflow_matches3d_vec.clear();
    for (const auto &it : frame->m_opticalflow_matches2dto3d) {
        PoseSolver::Point3D point_3d(
            pointClouds3dObj[it.second](0), pointClouds3dObj[it.second](1),
            pointClouds3dObj[it.second](2));
        PoseSolver::Point2D point_2d = NormalizePoint2D(
            frame->m_opticalflow_point2ds[it.first], fx, fy, cx, cy);
        PoseSolver::Match3D match_3d(point_3d, point_2d);
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
    GlobalOcvViewer::AddView("optical match", img_match);
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

PoseSolver::MatchSet3D PointCloudObjTracker::FindOpticalFlow3DMatch() {
    PoseSolver::MatchSet3D matchset_3d;
#ifdef USE_NO_OPTICALFLOW_FOR_TRACKER
    return matchset_3d;
#else
    if (m_frame_Pre->m_opticalflow_matches2dto3d.empty() ||
        m_frame_Pre->m_opticalflow_point2ds.empty()) {
        return matchset_3d;
    }

    TIMER_UTILITY::Timer timer;
    const std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    if (pointClouds.empty()) {
        LOG(ERROR) << "No mapPoints here!!!";
        return matchset_3d;
    }

    m_frame_cur->m_opticalflow_matches2dto3d.clear();
    m_frame_cur->m_opticalflow_point2ds.clear();

    FrameIndex frmIndex = m_frame_cur->m_frame_index;
    double timeStamp;
    ObjRecogState state;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    mObj->GetPose(frmIndex, timeStamp, state, Rcw, tcw, m_Rwo_cur, m_two_cur);
    const Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();

    std::vector<Eigen::Vector3d> mapPointsObj;
    std::vector<Eigen::Vector3d> mapPointsWorld;
    ObjTrackerCommon::GetMapPointPositions(
        pointClouds, m_Rwo_cur, m_two_cur, mapPointsObj, mapPointsWorld);

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

    STATISTICS_UTILITY::StatsCollector tracker_find_opticalflow_match(
        "Time: tracker find opticalflow match");
    tracker_find_opticalflow_match.AddSample(timer.Stop());
    return matchset_3d;
#endif
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

PoseSolver::MatchSet3D PointCloudObjTracker::FindProjection3DMatch() {

    PoseSolver::MatchSet3D matchset_3d;
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
    mObj->GetPose(frmIndex, timeStamp, state, Rcw, tcw, m_Rwo_cur, m_two_cur);

    std::vector<Eigen::Vector3d> mapPointsObj;
    std::vector<Eigen::Vector3d> mapPointsWorld;
    ObjTrackerCommon::GetMapPointPositions(
        pointClouds, m_Rwo_cur, m_two_cur, mapPointsObj, mapPointsWorld);

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
    TIMER_UTILITY::Timer timer;
    ObjTrackerCommon::ProjectSearch(
        mapPointsWorld, m_Rcw_cur, m_tcw_cur, projectFailState, projectPoints,
        false);
    m_project_success_mappoint_num = projectPoints.size();

    STATISTICS_UTILITY::StatsCollector stats_collector_project(
        "tracker mappoint project to image num");
    stats_collector_project.AddSample(m_project_success_mappoint_num);

    // 50
    /*if (m_project_success_mappoint_num <
        Parameters::GetInstance().kTrackerProjectSuccessNumTh) {
        return matchset_3d;
    }*/

    std::vector<bool> matchKeyPointsState;
    m_projection_matches2dTo3d_cur.clear();

#ifdef SUPERPOINT
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
    STATISTICS_UTILITY::StatsCollector tracker_projection_match(
        "Time: tracker find projection match");
    tracker_projection_match.AddSample(timer.Stop());

    m_match_points_projection_num = m_projection_matches2dTo3d_cur.size();
    STATISTICS_UTILITY::StatsCollector stats_projection_match(
        "tracker projection 2D-3D match num");
    stats_projection_match.AddSample(m_projection_matches2dTo3d_cur.size());

    matchset_3d = Generate3DMatch(
        m_frame_cur, m_projection_matches2dTo3d_cur, m_projection_points2d_cur,
        mapPointsObj);

    return matchset_3d;
}

bool PointCloudObjTracker::PoseSolver(
    const PoseSolver::MatchSet3D &matches_3d,
    const std::vector<PoseSolver::MatchSet2D> &matches_2d,
    std::vector<int> &inliers_3d) {

    m_pnp_solver_result = false;
    m_pnp_inliers_num = 0;
    // 40
    if (matches_3d.size() <
        Parameters::GetInstance().kTrackerMatchPointsNumTh) {
        VLOG(5) << "Not enough opticalFlow 3d mathces";
        return false;
    }

    PoseSolver::Options options;
    PoseSolver::Pose T;
    inliers_3d.clear();
    std::vector<std::vector<int>> inliers_2d;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();

    const float kPnpReprojectionError = 4.0;
    options.max_reproj_err =
        kPnpReprojectionError / (static_cast<float>(Kcv.at<double>(0, 0)));
    options.ransac_iterations = 100;
    options.ransac_confidence = 0.90;
    const int kPnpMinMatchesNum = 0;

    int kPnpMinInlierNum =
        Parameters::GetInstance().kTrackerPnPInliersGoodNumTh_PoseSolver;

    const double kPnpMinInlierRatio = 0.0;
    options.callbacks.emplace_back(PoseSolver::EarlyBreakBy3DInlierCounting(
        kPnpMinMatchesNum, kPnpMinInlierNum, kPnpMinInlierRatio));

    m_pnp_solver_result = PoseSolver::Ransac_Tracker(
        options, matches_3d, matches_2d, &T, &inliers_3d, &inliers_2d);

    Eigen::Matrix3d Rco = T.m_R.cast<double>();
    Eigen::Vector3d tco = T.m_t.cast<double>();

#ifdef MONO
    Eigen::Vector3d tco_slam_scale;
    if (mObj->GetScale() <= 1e-5) {
        tco_slam_scale = tco;
    } else {
        tco_slam_scale = tco / mObj->GetScale();
    }
#else
    Eigen::Vector3d tco_slam_scale = tco;
#endif

    Eigen::Matrix3d Row = Rco.transpose() * (m_Rcw_cur);
    Eigen::Vector3d tow = Rco.transpose() * (m_tcw_cur - tco_slam_scale);
    m_Rwo_cur = Row.transpose();
    m_two_cur = -m_Rwo_cur * tow;
    m_Rco_cur = Rco;
    m_tco_cur = tco_slam_scale;

    m_pnp_inliers_num = inliers_3d.size();
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
#ifdef USE_NO_METHOD_FOR_FUSE
#else
    m_Rwo_cur = Rwo_old;
    m_two_cur = Two_old;
#endif

#ifdef USE_INLIER
    int KTrackerPnPInliersGoodTh =
        Parameters::GetInstance().kTrackerPnPInliersGoodNumTh;
    int KTrackerPnPInliersUnreliableTh =
        Parameters::GetInstance().kTrackerPnPInliersUnreliableNumTh;

    if (m_Rwo_cur == Eigen::Matrix3d::Identity() &&
        m_two_cur == Eigen::Vector3d::Zero()) {
        m_tracker_state = TrackingBad;
    } else {
#ifdef OBJECT_BOX
#ifdef SUPERPOINT
#ifdef MONO
        int proj_success_num = 50;
#else
        int proj_success_num = m_match_points_projection_num * 0.15;
#endif
#else
        int proj_success_num = m_match_points_projection_num * 0.10;
#endif
#endif
#ifdef OBJECT_BAG
#ifdef SUPERPOINT
#ifdef MONO
        int proj_success_num = 40;
#else
        int proj_success_num = 70;
#endif
#else
        int proj_success_num = m_match_points_projection_num * 0.12;
#endif
#endif

#ifdef OBJECT_TOY
#ifdef SUPERPOINT
#ifdef MONO
        int proj_success_num = 40;
#else
        int proj_success_num = 70;
#endif
#else
        int proj_success_num = 70;
#endif
#endif

        if (m_pnp_solver_result &&
            m_pnp_inliers_num > KTrackerPnPInliersGoodTh &&
            m_pnp_inliers_projection_num > proj_success_num) {
            VLOG(5) << "tracker pnp inlier num success:" << m_pnp_inliers_num;
            m_tracker_state = TrackingGood;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_tracker_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
            mObj->SetPoseForFindSimilarKeyframe(m_Rcw_cur, m_tcw_cur);

            if (m_reproj_error >= 0) {
                // VLOG(0) << "tracker reproj error: " << m_reproj_error;
                STATISTICS_UTILITY::StatsCollector stats_collector_project(
                    "tracker inlier reproj error");
                stats_collector_project.AddSample(m_reproj_error);
            }

        } else if (m_pnp_inliers_num > KTrackerPnPInliersUnreliableTh) {
            VLOG(5) << "PNP fail but has enough inliers";
            m_tracker_state = TrackingUnreliable;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_tracker_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        } else {
            VLOG(5) << "PNP solve fail!";
            m_tracker_state = TrackingBad;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_tracker_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        }
        if (m_tracker_state == TrackingGood) {
            m_frame_Pre = m_frame_cur;
        } else {
            m_frame_Pre = std::make_shared<TrackerFrame>();
        }
    }
#endif

#ifdef USE_REPROJ
    if (m_Rwo_cur == Eigen::Matrix3d::Identity() &&
        m_two_cur == Eigen::Vector3d::Zero()) {
        m_tracker_state = TrackingBad;
    } else {
        // 80
        if (m_pnp_solver_result && m_reproj_error <= 3.0) {
            VLOG(5) << "tracker pnp inlier num success:" << m_pnp_inliers_num;
            m_tracker_state = TrackingGood;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_tracker_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        } else if (m_reproj_error <= 4.5) {
            VLOG(5) << "PNP fail but has enough inliers";
            m_tracker_state = TrackingUnreliable;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_tracker_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        } else {
            VLOG(5) << "PNP solve fail!";
            m_tracker_state = TrackingBad;
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_tracker_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        }
        if (m_tracker_state == TrackingGood) {
            m_frame_Pre = m_frame_cur;
        } else {
            m_frame_Pre = std::make_shared<TrackerFrame>();
        }
    }
#endif
} // namespace ObjRecognition

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

    m_opticalflow_point2ds_tmp = m_frame_cur->m_opticalflow_point2ds;

    m_frame_cur->m_opticalflow_point2ds.clear();
    m_frame_cur->m_opticalflow_matches2dto3d.clear();

    int point2d_index = 0;
    for (const auto &it : m_opticalFlow_matches2dTo3d_inlier) {
        m_frame_cur->m_opticalflow_point2ds.emplace_back(
            m_opticalflow_point2ds_tmp[it.first]);
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
    std::string reproj_txt = "reproj error: " + std::to_string(m_reproj_error);

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
        Eigen::Vector3d p =
            K * (m_Rco_cur * mapPointBoundingBox[i] + m_tco_cur);
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
    GlobalOcvViewer::AddView("Tracker Result", img_text);
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
                    (m_Rco_cur * mappoint->GetPose() + m_tco_cur);
        Eigen::Vector2d proj_2d =
            Eigen::Vector2d(proj(0) / proj(2), proj(1) / proj(2));

        float dis = sqrt(
            (proj_2d(0) - keypoint.pt.x) * (proj_2d(0) - keypoint.pt.x) +
            (proj_2d(1) - keypoint.pt.y) * (proj_2d(1) - keypoint.pt.y));

        average_error += dis;
    }

    for (auto matches : m_opticalFlow_matches2dTo3d_inlier) {
        auto mappoint = mObj->GetPointClouds()[matches.second];
        auto keypoint = m_opticalflow_point2ds_tmp[matches.first];

        auto proj = CameraIntrinsic::GetInstance().GetEigenK() *
                    (m_Rco_cur * mappoint->GetPose() + m_tco_cur);
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
    const std::shared_ptr<ObjRecognition::FrameForObjRecognition> &frm) {
    if ((frm == nullptr || (frm != nullptr && frm->m_img.data == nullptr) ||
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

    TIMER_UTILITY::Timer timer;
    m_frame_cur = std::make_shared<TrackerFrame>();
    m_frame_cur->m_frame_index = frm->m_frmIndex;
    m_frame_cur->m_raw_image = frm->m_img.clone();
    m_frame_cur->m_desp = frm->m_desp.clone();
    m_frame_cur->m_kpts = frm->m_kpts;

    m_Rcw_cur = frm->m_Rcw;
    m_tcw_cur = frm->m_tcw;

    PoseSolver::MatchSet3D totalMatchSet_3d = FindOpticalFlow3DMatch();

    PoseSolver::MatchSet3D matchset_3d = FindProjection3DMatch();
    totalMatchSet_3d.insert(
        totalMatchSet_3d.end(), matchset_3d.begin(), matchset_3d.end());

    std::vector<int> totalInliers_3d;
    TIMER_UTILITY::Timer timer_poseSolver;
    PoseSolver(totalMatchSet_3d, {}, totalInliers_3d);
    ProcessPoseSolverInliers(totalInliers_3d);
    STATISTICS_UTILITY::StatsCollector tracker_pose_solver(
        "Time: tracker pose solver");
    tracker_pose_solver.AddSample(timer_poseSolver.Stop());

    m_reproj_error = ComputeAverageReProjError(totalInliers_3d);

    PnPResultHandle();

    STATISTICS_UTILITY::StatsCollector tracker_process_time(
        "Time: tracker process single image");
    tracker_process_time.AddSample(timer.Stop());

    ShowTrackerResult();

    ResultRecord();
}

void PointCloudObjTracker::Reset() {
    Clear();
}

void PointCloudObjTracker::Clear() {
    m_frame_Pre = std::make_shared<TrackerFrame>();
    m_project_success_mappoint_num = 0;
    m_pnp_inliers_num = 0;
    m_tracker_state = TrackingBad;
    m_first_detection_good = false;
    m_scale = 0;
}

bool PointCloudObjTracker::Load(const long long &mem_size, const char *mem) {
    VLOG(30) << "PointCloudObjTracker::Load";
    return mObj->LoadPointCloud(mem_size, mem);
}

bool PointCloudObjTracker::Save(long long &mem_size, char **mem) {
    VLOG(30) << "PointCloudObjTracker::Save";
    return mObj->Save(mem_size, mem);
}
} // namespace ObjRecognition
