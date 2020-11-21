//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include <cv.hpp>
#include "Visualizer/GlobalImageViewer.h"
#include "ObjectRecognition/Utility/Utility.h"
#include "Detector/DetectorCommon.h"
#include "Utility/Camera.h"
#include "Utility/Statistics.h"
#include "Utility/Parameters.h"
#include "Utility/Timer.h"
#include "Detector/DetectorPointCloud.h"
#include "Optimizer/LBAOptimizer.h"
#include "mode.h"

namespace ObjRecognition {

PointCloudObjDetector::PointCloudObjDetector() {
    Clear();
}

PointCloudObjDetector::~PointCloudObjDetector() {
}

void PointCloudObjDetector::SetPointCloudObj(
    const std::shared_ptr<Object> &pObj) {
    CHECK_NOTNULL(pObj.get());
    mObj = pObj;
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
    for (int index = 0; index < kf_matches.size(); index++) {
        cv::Mat kf_raw_image = kf_matches.at(index)->GetRawImage();
        // keyframs selected from database using dbow2
        if (!kf_raw_image.empty()) {
            cv::Mat imshow;
            cv::drawMatches(
                frm->m_raw_image, frm->m_kpts, kf_raw_image,
                kf_matches.at(index)->GetKeyPoints(),
                frm->m_dmatches_2d.at(index), imshow);
            const std::string kf_match_result_name =
                "match 2D: " + std::to_string(index);
            cv::imwrite(
                "/home/zhangye/data1/test_objRecognition/" +
                    std::to_string(frm->m_frame_index) + "match.png",
                imshow);
            GlobalOcvViewer::UpdateView(kf_match_result_name, imshow);
        }
    }
}

std::vector<PS::MatchSet2D> Generate2DMatchesFromKeyFrame(
    const std::shared_ptr<DetectorFrame> &frm,
    const std::vector<KeyFrame::Ptr> &kf_matches) {

    auto fx = CameraIntrinsic::GetInstance().GetEigenK()(0, 0);
    auto fy = CameraIntrinsic::GetInstance().GetEigenK()(1, 1);
    auto cx = CameraIntrinsic::GetInstance().GetEigenK()(0, 2);
    auto cy = CameraIntrinsic::GetInstance().GetEigenK()(1, 2);

    int matches2d_count = 0;
    cv::Mat frmDesp = frm->m_desp;
    std::vector<PS::MatchSet2D> matchs_2ds;

    if (kf_matches.empty())
        return matchs_2ds;

    TIMER_UTILITY::Timer timer_getmatch;
    frm->m_dmatches_2d.clear();
    for (int index = 0; index < kf_matches.size(); index++) {
        CHECK_NOTNULL(kf_matches.at(index).get());
        PS::MatchSet2D matches;
        std::vector<cv::DMatch> dmatches;
        cv::Mat kfDesp = kf_matches.at(index)->GetDesciriptor();

#ifdef SUPERPOINT
        ObjDetectionCommon::FindMatchByKNN_SuperPoint_Homography(
            frm->m_kpts, kf_matches.at(index)->GetKeyPoints(), frmDesp, kfDesp,
            dmatches);
#else
        const float ratio_threshold = 0.70;
        ObjDetectionCommon::FindMatchByKNN_Homography(
            frm->m_kpts, kf_matches.at(index)->GetKeyPoints(), frmDesp, kfDesp,
            dmatches, ratio_threshold);
#endif

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
    STATISTICS_UTILITY::StatsCollector detector_2d_match_time(
        "Time: detector find 2d match");
    detector_2d_match_time.AddSample(timer_getmatch.Stop());

    STATISTICS_UTILITY::StatsCollector detector_2d_matches(
        "detector 2D-2D matches num");
    detector_2d_matches.AddSample(matches2d_count);
    // MatchKeyFramesShow(frm, kf_matches);
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

std::vector<PS::MatchSet2D> PointCloudObjDetector::Find2DMatches(
    const std::vector<KeyFrame::Ptr> &allKFs,
    std::vector<KeyFrame::Ptr> &kf_mathceds) {

#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
    // set same keyframe use pose_pre
    Eigen::Matrix3d Rcw_for_similar_keyframe;
    Eigen::Vector3d tcw_for_similar_keyframe;
    Eigen::Matrix3d Rwo_for_similar_keyframe;
    Eigen::Vector3d two_for_similar_keyframe;
    mObj->GetPoseForFindSimilarKeyframe(
        Rcw_for_similar_keyframe, tcw_for_similar_keyframe,
        Rwo_for_similar_keyframe, two_for_similar_keyframe);

    Eigen::Matrix4d Tcw;
    Tcw.setIdentity();
    Tcw.block<3, 3>(0, 0) = Rcw_for_similar_keyframe;
    Tcw.block<3, 1>(0, 3) = tcw_for_similar_keyframe;

    Eigen::Matrix4d Two = Eigen::Matrix4d::Identity();
    Two.block<3, 3>(0, 0) = Rwo_for_similar_keyframe;
    Two.block<3, 1>(0, 3) = two_for_similar_keyframe;

    Eigen::Matrix4d Tco = Tcw * Two;
    Eigen::Matrix3d Rco = Tco.block<3, 3>(0, 0);
    Eigen::Vector3d tco = Tco.block<3, 1>(0, 3);

    if (Rcw_for_similar_keyframe == Eigen::Matrix3d::Identity() &&
        Rwo_for_similar_keyframe == Eigen::Matrix3d::Identity()) {

    } else {
        KeyFrame::Ptr keyframe_best;
        KeyFrame::Ptr keyframe_better;

        float best_angle = 360.0;
        float best_dist = 100;

        float better_angle = 360.0;
        float better_dist = 100;

        for (const auto &keyframe : mObj->GetKeyFrames()) {
            Eigen::Vector3d tcw_keyframe;
            Eigen::Matrix3d Rcw_keyframe;
            keyframe->GetPose(Rcw_keyframe, tcw_keyframe);

            Eigen::Matrix3d delta_R = Rcw_keyframe.inverse() * Rco;
            float angle =
                acos((delta_R.diagonal().sum() - 1) / 2.0) * (180 / 3.14);

            Eigen::Vector3d delta_t = tco - tcw_keyframe;
            float distance = sqrt(
                delta_t(0) * delta_t(0) + delta_t(1) * delta_t(1) +
                delta_t(2) * delta_t(2));

            //            if (angle < 35 && distance < 0.38) {
            //                kf_mathceds.emplace_back(keyframe);
            //            }

            if (angle < best_angle && distance < best_dist) {
                better_angle = best_angle;
                better_dist = best_dist;
                keyframe_better = keyframe_best;

                best_angle = angle;
                best_dist = distance;
                keyframe_best = keyframe;
            } else if (angle < better_angle && distance < better_dist) {
                better_angle = angle;
                better_dist = distance;
                keyframe_better = keyframe;
            }
        }

        if (kf_mathceds.size() != 2) {
            if (kf_mathceds.empty()) {
                kf_mathceds.emplace_back(keyframe_best);
                kf_mathceds.emplace_back(keyframe_better);
            } else {
                kf_mathceds.emplace_back(keyframe_best);
            }
        }
    }

#else
    TIMER_UTILITY::Timer timer;
    kf_mathceds = mObj->FrameQueryMap(m_frame_cur);
    STATISTICS_UTILITY::StatsCollector detector_find_2d_match_time(
        "Time: detector query from map");
    detector_find_2d_match_time.AddSample(timer.Stop());
#endif
    std::vector<PS::MatchSet2D> matches_2ds;

    if (kf_mathceds.empty()) {
        LOG(ERROR) << "first frame has no candidate similar keyframes";
        return matches_2ds;
    }

    std::set<int> kf_matches_id;
    for (const KeyFrame::Ptr &kf : kf_mathceds) {
        kf_matches_id.insert(kf->GetID());
    }
    // GlobalKeyFrameMatchViewer::SetMatchedKeyFrames(kf_matches_id);
    matches_2ds = Generate2DMatchesFromKeyFrame(m_frame_cur, kf_mathceds);
    return matches_2ds;
}

PS::MatchSet3D PointCloudObjDetector::Find3DMatchByConnection(
    const std::vector<KeyFrame::Ptr> &kf_mathceds_by_pose) {
    PS::MatchSet3D matches_3d;

#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
    std::vector<KeyFrame::Ptr> kf_mathceds = kf_mathceds_by_pose;
#else
    std::vector<KeyFrame::Ptr> kf_mathceds = kf_mathceds_by_pose;
#endif

    if (kf_mathceds.empty()) {
        // no match, choose first two keyframe
        kf_mathceds.emplace_back(mObj->GetKeyFrames()[0]);
        kf_mathceds.emplace_back(mObj->GetKeyFrames()[1]);
    }

    std::set<int> associated_keyframe_ids;
    std::set<MapPoint::Ptr> associated_mappoints;
    std::set<MapPointIndex> associated_mappoints_ids;
    for (const KeyFrame::Ptr &kf : kf_mathceds) {
        auto connect_kf_ids = kf->connect_kfs;
        connect_kf_ids.emplace_back(kf->GetID());
        VLOG(0) << "single image connected keyframes:"
                << kf->connect_kfs.size();
        for (auto connect_kf_id : connect_kf_ids) {
            associated_keyframe_ids.insert(connect_kf_id);
            if (mObj->m_mp_keyframes.find(connect_kf_id) !=
                mObj->m_mp_keyframes.end()) {
                auto keyframe = mObj->m_mp_keyframes[connect_kf_id];
                auto connected_mappoints_id = keyframe->connect_mappoints;
                for (auto mappoint_id : connected_mappoints_id) {
                    if (mObj->m_pointclouds_map.find(mappoint_id) ==
                        mObj->m_pointclouds_map.end()) {
                        // not in the boundingbox
                        continue;
                    }
                    associated_mappoints.insert(
                        mObj->m_pointclouds_map[mappoint_id]);
                    associated_mappoints_ids.insert(mappoint_id);
                }
            } else {
                VLOG(0) << "the keyframe is not exist";
            }
        }
    }

    std::vector<MapPoint::Ptr> associated_mappoints_vector;
    for (const auto &mappoint : associated_mappoints) {
        associated_mappoints_vector.emplace_back(mappoint);
    }
    VLOG(0) << "associated mappoints num: "
            << associated_mappoints_vector.size();
    VLOG(0) << "associated keyframe num: " << associated_keyframe_ids.size();

    if (associated_mappoints_vector.empty()) {
        LOG(ERROR) << "there is no connected computed mappoints";
        return matches_3d;
    }

    // associated_mappoints
    mObj->SetAssociatedMapPointsByConnection(associated_mappoints_ids);
    mObj->SetAssociatedKeyFrames(associated_keyframe_ids);

    cv::Mat pcDesp;

    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    auto fx = static_cast<float>(Kcv.at<double>(0, 0));
    auto fy = static_cast<float>(Kcv.at<double>(1, 1));
    auto cx = static_cast<float>(Kcv.at<double>(0, 2));
    auto cy = static_cast<float>(Kcv.at<double>(1, 2));

    pcDesp = ObjDetectionCommon::GetPointCloudDespByConnection(
        associated_mappoints_vector);

    cv::Mat frmDesp;
    frmDesp = m_frame_cur->m_desp;

    std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    std::map<int, MapPointIndex> matches2dTo3d;
    matches2dTo3d.clear();
    knn_match_num_ = -1;

    std::vector<cv::DMatch> goodMatches;
#ifdef SUPERPOINT
    // TODO(zhangye): use only norm2 distance for superpoint match?
    TIMER_UTILITY::Timer timer;
    ObjDetectionCommon::FindMatchByKNN_SuperPoint(frmDesp, pcDesp, goodMatches);
    STATISTICS_UTILITY::StatsCollector detector_find_3d_by_connection_time(
        "Time: detector find 3d match by connection");
    detector_find_3d_by_connection_time.AddSample(timer.Stop());
#else
    const float ratio_threshold = 0.70;
    ObjDetectionCommon::FindMatchByKNN(
        frmDesp, pcDesp, goodMatches, ratio_threshold);
#endif

    for (int i = 0; i < goodMatches.size(); i++) {
        matches2dTo3d.insert(std::pair<int, MapPointIndex>(
            goodMatches[i].queryIdx, goodMatches[i].trainIdx));
    }

    knn_match_num_ = goodMatches.size();
    VLOG(5) << "detection 2D-3D match: " << goodMatches.size();

    STATISTICS_UTILITY::StatsCollector stats_collector_knn(
        "detector 2D-3D matches num");
    stats_collector_knn.AddSample(knn_match_num_);

    // 8
    const int kGoodMatchNumTh =
        Parameters::GetInstance().kDetectorKNNMatchNumTh;
    if (knn_match_num_ < kGoodMatchNumTh) {
        VLOG(5) << "The number of KNN 2D-3D match is small";
        return matches_3d;
    }

    std::vector<cv::Point3f> ptsObj;
    std::vector<cv::Point2f> ptsImg;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        ptsImg.emplace_back(cv::Point2f(keyPoints[iter->first].pt));
        cv::Point3f mapPoint = TypeConverter::Eigen2CVPoint(
            associated_mappoints_vector[iter->second]->GetPose());
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

    m_frame_cur->m_matches_3d_byconnection.clear();
    m_frame_cur->m_matches_3d_byconnection = matches2dTo3d;
    return matches_3d;
}

PS::MatchSet3D PointCloudObjDetector::Find3DMatch() {
    TIMER_UTILITY::Timer timer;
    PS::MatchSet3D matches_3d;
    cv::Mat pcDesp;

    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    auto fx = static_cast<float>(Kcv.at<double>(0, 0));
    auto fy = static_cast<float>(Kcv.at<double>(1, 1));
    auto cx = static_cast<float>(Kcv.at<double>(0, 2));
    auto cy = static_cast<float>(Kcv.at<double>(1, 2));

    pcDesp = ObjDetectionCommon::GetPointCloudDesp(mObj);

    cv::Mat frmDesp;
    frmDesp = m_frame_cur->m_desp;

    std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    std::map<int, MapPointIndex> matches2dTo3d;
    matches2dTo3d.clear();
    knn_match_num_ = -1;

    std::vector<cv::DMatch> goodMatches;
#ifdef SUPERPOINT
    // TODO(zhangye): use only norm2 distance for superpoint match?
    ObjDetectionCommon::FindMatchByKNN_SuperPoint(frmDesp, pcDesp, goodMatches);
#else
    const float ratio_threshold = 0.70;
    ObjDetectionCommon::FindMatchByKNN(
        frmDesp, pcDesp, goodMatches, ratio_threshold);
#endif

    for (int i = 0; i < goodMatches.size(); i++) {
        matches2dTo3d.insert(std::pair<int, MapPointIndex>(
            goodMatches[i].queryIdx, goodMatches[i].trainIdx));
    }

    knn_match_num_ = goodMatches.size();
    VLOG(5) << "detection 2D-3D match: " << goodMatches.size();

    STATISTICS_UTILITY::StatsCollector stats_collector_knn(
        "detector 2D-3D matches num");
    stats_collector_knn.AddSample(knn_match_num_);

    STATISTICS_UTILITY::StatsCollector detector_find_3d_time(
        "Time: detector find 3d match");
    detector_find_3d_time.AddSample(timer.Stop());

    // 8
    const int kGoodMatchNumTh =
        Parameters::GetInstance().kDetectorKNNMatchNumTh;
    if (knn_match_num_ < kGoodMatchNumTh) {
        VLOG(5) << "The number of KNN 2D-3D match is small";
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
    if (inliers_3d.empty()) {
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
    PS::Options options;
    PS::Pose T;
    pnp_inliers_num_ = 0;
    pnp_inliers_3d_num_ = 0;
    pnp_inliers_2d_num_ = 0;

    const float kPnpReprojectionError = 6.5;
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
    // 50
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

    //    VLOG(0) << "detectionPoseSolver 3d matches size: " <<
    //    matches_3d.size(); VLOG(0) << "detectionPoseSolver 3d inliers size: "
    //    << inliers_3d.size(); for (int i = 0; i < matches_2d.size(); i++) {
    //        VLOG(0) << "detectionPoseSolver 2d matches size: "
    //                << matches_2d.at(i).size();
    //        VLOG(0) << "detectionPoseSolver 2d inliers size: "
    //                << inliers_2d.at(i).size();
    //    }

    Eigen::Matrix3d Rco = T.m_R.cast<double>();
    Eigen::Vector3d tco = T.m_t.cast<double>();
    Rco_cur_ = Rco;
    tco_cur_ = tco;

    // PoseOptimize(inliers_3d);

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

    VLOG(5) << "detection inlier num 2d+3d: " << pnp_inliers_num_;
    STATISTICS_UTILITY::StatsCollector stats_collector_pnp_3d(
        "detector PnP 3d inlier num");
    stats_collector_pnp_3d.AddSample(pnp_inliers_3d_num_);
    STATISTICS_UTILITY::StatsCollector stats_collector_pnp_2d(
        "detector PnP 2d inlier num");
    stats_collector_pnp_2d.AddSample(pnp_inliers_2d_num_);

    return pnp_solver_result_;
}

void PointCloudObjDetector::PnPResultHandle() {
#ifdef USE_INLIER
#ifdef OBJ_WITH_KF
    // 50
    const int kDetectorPnPInliersThGood =
        Parameters::GetInstance().kDetectorPnPInliersGoodWithKFNumTh;
    // 20
    const int kDetectorPnPInliersThUnreliable =
        Parameters::GetInstance().kDetectorPnPInliersUnreliableWithKFNumTh;
#else
    const int kDetectorPnPInliersThGood =
        Parameters::GetInstance().kDetectorPnPInliersGoodNumTh;
    const int kDetectorPnPInliersThUnreliable =
        Parameters::GetInstance().kDetectorPnPInliersUnreliableNumTh;
#endif
#ifdef SUPERPOINT
    int proj_success_num = 40;
#else
    int proj_success_num = knn_match_num_ * 0.5;
#endif

    if (pnp_solver_result_ && pnp_inliers_num_ >= kDetectorPnPInliersThGood &&
        pnp_inliers_3d_num_ >= proj_success_num) {
        detect_state_ = DetectionGood;
        mObj->SetPoseForFindSimilarKeyframe(
            Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        mObj->SetPose(
            m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
            detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);

        // evaluation method
        if (reproj_error >= 0) {
            STATISTICS_UTILITY::StatsCollector stats_collector_project(
                "detector inlier reproj error");
            stats_collector_project.AddSample(reproj_error);
        }
    } else {
        if (pnp_inliers_num_ >= kDetectorPnPInliersThUnreliable) {
            VLOG(5) << "detector PnP fail but has enough inliers";
            detect_state_ = DetectionUnreliable;
            if (!has_good_result) {
                mObj->SetPoseForFindSimilarKeyframe(
                    Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
            }
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        } else {
            VLOG(5) << "detector PnP solve fail!";
            detect_state_ = DetectionBad;
            if (!has_good_result) {
                mObj->SetPoseForFindSimilarKeyframe(
                    Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
            }
            mObj->SetPose(
                m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
                detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
        }
    }
#endif

#ifdef USE_REPROJ
    if (pnp_solver_result_ && reproj_error <= 3.0) {
        detect_state_ = DetectionGood;
        mObj->SetPose(
            m_frame_cur->m_frame_index, m_frame_cur->m_time_stamp,
            detect_state_, Rcw_cur_, tcw_cur_, Rwo_cur_, two_cur_);
    } else {
        if (pnp_inliers_num_ <= 4.5) {
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
#endif
}

void PointCloudObjDetector::DrawTextInfo(const cv::Mat &img, cv::Mat &img_txt) {
    // 2d-3d暴力匹配上的关键点
#ifdef USE_CONNECT_FOR_DETECTOR
    std::string match_txt =
        "3dmatch num:" +
        std::to_string(m_frame_cur->m_matches_3d_byconnection.size()) + "| ";
#else
    std::string match_txt =
        "3dmatch num:" + std::to_string(m_frame_cur->m_matches_3d.size()) +
        "| ";
#endif

    //经过2d-3d, 2d-2d联合求解pose过后inlier的三维点对应的keypoint
    std::string inlier_txt =
        "3d|2d inliers num: " + std::to_string(pnp_inliers_3d_num_) + "|" +
        std::to_string(pnp_inliers_2d_num_) + "| ";
    std::string reproj_txt = "reproj error: " + std::to_string(reproj_error);

    std::string detectionStateString;
    if (detect_state_ == DetectionGood) {
        detectionStateString = "Good";

    } else if (detect_state_ == DetectionUnreliable) {
        detectionStateString = "Unreliable";
    } else {
        detectionStateString = "Bad";
    }

    std::stringstream s;
    s << match_txt;
    s << inlier_txt;
    s << reproj_txt;
    s << detectionStateString;

    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    img_txt = cv::Mat(img.rows + textSize.height + 10, img.cols, img.type());
    img.copyTo(img_txt.rowRange(0, img.rows).colRange(0, img.cols));
    img_txt.rowRange(img.rows, img_txt.rows) =
        cv::Mat::zeros(textSize.height + 10, img.cols, img.type());
    cv::putText(
        img_txt, s.str(), cv::Point(5, img_txt.rows - 5),
        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void PointCloudObjDetector::ShowDetectResult() {
    const Eigen::Isometry3f Tco =
        ObjDetectionCommon::GetTMatrix(Rco_cur_, tco_cur_);
    const std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    if (pointClouds.empty()) {
        LOG(ERROR) << "No pointclouds model here!";
    }
    const std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    const cv::Mat &cameraMatrix = CameraIntrinsic::GetInstance().GetCVK();
    std::vector<cv::Point3f> point3f;
    std::vector<cv::Point2f> point2f;
    std::vector<cv::Point2f> imagePoint2f;
    Eigen::Vector3f tempPoint3f;
    Eigen::Vector3f tempCameraPoint3f;
    cv::Mat showResult;
    cv::Mat imageCur = m_frame_cur->m_raw_image.clone();
    cv::cvtColor(imageCur, showResult, cv::COLOR_GRAY2BGR);

    std::vector<Eigen::Vector3d> mapPointBoundingBox;
    mapPointBoundingBox.reserve(8);
    ObjDetectionCommon::GetPointCloudBoundingBox(mObj, mapPointBoundingBox);
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
    if (detect_state_ == DetectionGood) {
        edge_color = cv::Scalar(224, 24, 255);
    } else {
        edge_color = cv::Scalar(0, 0, 0);
    }

    for (int i = 0; i < boxProjResult.size(); i++) {
        if (i < 4)
            cv::drawMarker(showResult, boxProjResult[i], corner_color);
        else
            cv::drawMarker(showResult, boxProjResult[i], corner_color);
        ObjDetectionCommon::DrawBoundingBox(
            showResult, boxProjResult, edge_color);
    }

    std::vector<cv::KeyPoint> matcheskeyPointsShow;
    // std::vector<MapPointIndex> mapPointId;

    for (auto iter = m_frame_cur->m_matches_3d.begin();
         iter != m_frame_cur->m_matches_3d.end(); iter++) {
        matcheskeyPointsShow.emplace_back(keyPoints[iter->first]);
        // mapPointId.emplace_back(iter->second);
    }

    // red circle  2d-3d暴力匹配上的关键点
    drawKeypoints(
        showResult, matcheskeyPointsShow, showResult, cv::Scalar(0, 0, 255));

    std::vector<cv::KeyPoint> matcheskeyPointsInliersShow;
    for (auto iter = m_frame_cur->m_matches2dto3d_inliers.begin();
         iter != m_frame_cur->m_matches2dto3d_inliers.end(); iter++) {
        matcheskeyPointsInliersShow.emplace_back(keyPoints[iter->first]);
    }

    // yellow circle 经过2d-3d,
    // 2d-2d联合求解pose过后inlier的三维点对应的keypoint
    drawKeypoints(
        showResult, matcheskeyPointsInliersShow, showResult,
        cv::Scalar(0, 255, 255));

    // 2d-3d暴力匹配上的地图点 white points
    // GlobalPointCloudMatchViewer::SetMatchedMapPoint(mapPointId);
    cv::Mat img_text;
    DrawTextInfo(showResult, img_text);
    GlobalOcvViewer::UpdateView("Detector Result", img_text);
}

float PointCloudObjDetector::ComputeAverageReProjError(
    const std::vector<int> &inliers_3d) {
    m_frame_cur->m_matches2dto3d_inliers.clear();
#ifdef USE_CONNECT_FOR_DETECTOR
    GetMatch2dTo3dInliers(
        m_frame_cur->m_matches_3d_byconnection, inliers_3d,
        m_frame_cur->m_matches2dto3d_inliers);
#else
    GetMatch2dTo3dInliers(
        m_frame_cur->m_matches_3d, inliers_3d,
        m_frame_cur->m_matches2dto3d_inliers);
#endif

    float average_error = -1;
    if (!m_frame_cur->m_matches2dto3d_inliers.empty()) {
        average_error = 0.0;
        for (auto matches : m_frame_cur->m_matches2dto3d_inliers) {
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
        average_error /= (m_frame_cur->m_matches2dto3d_inliers.size());
    }
    return average_error;
}

// detect algorithm start
void PointCloudObjDetector::Process(
    const std::shared_ptr<ObjRecognition::FrameData> &frm) {
    if (frm == nullptr || (frm != nullptr && frm->img.data == nullptr) ||
        mObj == nullptr) {
        return;
    }

    TIMER_UTILITY::Timer timer;
    PreProcess(frm);

    std::vector<PS::MatchSet2D> matchset_2d;
#ifdef USE_OLNY_SCAN_MAPPOINT
#else
#ifdef OBJ_WITH_KF
    auto allKFs = mObj->GetKeyFrames();
    std::vector<KeyFrame::Ptr> kf_mathceds;
    matchset_2d = Find2DMatches(allKFs, kf_mathceds);
#endif
#endif

#ifdef USE_CONNECT_FOR_DETECTOR
    PS::MatchSet3D matchset_3d;
    if (kf_mathceds.empty()) {
        VLOG(0) << "use without connection";
        matchset_3d = Find3DMatch();
    } else {
        matchset_3d = Find3DMatchByConnection(kf_mathceds);
        if (matchset_3d.empty()) {
            matchset_3d = Find3DMatch();
            VLOG(0) << "use without connection";
        } else {
            VLOG(0) << "use connection with" << matchset_3d.size();
        }
    }
#else
    PS::MatchSet3D matchset_3d = Find3DMatch();
#endif

    std::vector<int> inliers_3d;
    std::vector<std::vector<int>> inliers_2d;
    TIMER_UTILITY::Timer timer_poseSolver;
    PoseSolver(matchset_3d, matchset_2d, inliers_3d, inliers_2d);
    STATISTICS_UTILITY::StatsCollector detector_pose_solver(
        "Time: detector pose solver");
    detector_pose_solver.AddSample(timer_poseSolver.Stop());

    reproj_error = ComputeAverageReProjError(inliers_3d);

    PnPResultHandle();

    STATISTICS_UTILITY::StatsCollector detector_process_time(
        "Time: detector process single image");
    detector_process_time.AddSample(timer.Stop());

    ShowDetectResult();

    ResultRecord();

    // SetInfo();
}

void PointCloudObjDetector::Reset() {
    Clear();
    VLOG(10) << "PointCloudObjDetector::Reset";
}

void PointCloudObjDetector::Clear() {
    knn_match_num_ = 0;
    pnp_inliers_num_ = 0;
    detect_state_ = DetectionBad;
    has_good_result = false;
    VLOG(10) << "PointCloudObjDetector::Clear";
}

bool PointCloudObjDetector::Load(const long long &mem_size, const char *mem) {
    VLOG(10) << "PointCloudObjDetector::Load";
    return mObj->LoadPointCloud(mem_size, mem);
}

bool PointCloudObjDetector::Save(long long &mem_size, char **mem) {
    VLOG(10) << "PointCloudObjDetector::Save";
    return mObj->Save(mem_size, mem);
}

void PointCloudObjDetector::ResultRecord() {
    if (detect_state_ == DetectionGood) {
        Eigen::Matrix3d Rcw;
        Eigen::Vector3d tcw;
        m_frame_cur->GetCameraPose(Rcw, tcw);
        STATISTICS_UTILITY::StatsCollector pointCloudDetectionNum(
            "detector good num");
        pointCloudDetectionNum.IncrementOne();
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
