#include <glog/logging.h>
#include <cv.hpp>
#include "Visualizer/GlobalImageViewer.h"
#include "Detector/DetectorCommon.h"
#include "Utility/Camera.h"
#include "StatisticsResult/Statistics.h"
#include "Utility/Parameters.h"
#include "StatisticsResult/Timer.h"
#include "Detector/DetectorPointCloud.h"
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
    m_obj = pObj;
}

void PointCloudObjDetector::SetVoc(
    const std::shared_ptr<DBoW3::Vocabulary> &pVoc) {
    CHECK_NOTNULL(pVoc.get());
    m_voc = pVoc;
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
            GlobalOcvViewer::AddView(kf_match_result_name, imshow);
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
    const std::shared_ptr<ObjRecognition::FrameForObjRecognition> &frm) {
    m_frame_cur = std::make_shared<DetectorFrame>();
    m_frame_cur->m_frame_index = frm->m_frmIndex;
    m_frame_cur->m_raw_image = frm->m_img.clone();
    m_frame_cur->m_desp = frm->m_desp.clone();
    m_frame_cur->m_kpts = frm->m_kpts;
    m_Rcw_cur = frm->m_Rcw;
    m_tcw_cur = frm->m_tcw;
}

std::vector<PS::MatchSet2D> PointCloudObjDetector::Find2DMatches(
    const std::vector<KeyFrame::Ptr> &allKFs,
    std::vector<KeyFrame::Ptr> &kf_mathceds) {

#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
    Eigen::Matrix3d Rcw_for_similar_keyframe;
    Eigen::Vector3d tcw_for_similar_keyframe;
    m_obj->GetPoseForFindSimilarKeyframe(
        Rcw_for_similar_keyframe, tcw_for_similar_keyframe);

    Eigen::Matrix3d Rco = m_Rco_cur;
    Eigen::Vector3d tco = m_tco_cur;

    if (Rcw_for_similar_keyframe == Eigen::Matrix3d::Identity()) {

    } else {
        KeyFrame::Ptr keyframe_best = m_obj->GetKeyFrames()[0];
        KeyFrame::Ptr keyframe_better = m_obj->GetKeyFrames()[1];

        float best_angle = 360.0;
        float best_dist = 100;

        float better_angle = 360.0;
        float better_dist = 100;

        for (const auto &keyframe : m_obj->GetKeyFrames()) {
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
    kf_mathceds = m_obj->FrameQueryMap(m_frame_cur);
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

    TIMER_UTILITY::Timer timer;
    if (kf_mathceds.empty()) {
        // no match, choose first two keyframe
        kf_mathceds.emplace_back(m_obj->GetKeyFrames()[0]);
        kf_mathceds.emplace_back(m_obj->GetKeyFrames()[1]);
    }

    std::set<int> associated_keyframe_ids;
    std::set<MapPoint::Ptr> associated_mappoints;
    std::set<MapPointIndex> associated_mappoints_ids;
    for (const KeyFrame::Ptr &kf : kf_mathceds) {
        auto connect_kf_ids = kf->m_connect_kfs;
        connect_kf_ids.emplace_back(kf->GetID());
        // VLOG(0) << "single image connected keyframes:"
        //<< kf->m_connect_kfs.size();
        for (auto connect_kf_id : connect_kf_ids) {
            associated_keyframe_ids.insert(connect_kf_id);
            if (m_obj->m_mp_keyframes.find(connect_kf_id) !=
                m_obj->m_mp_keyframes.end()) {
                auto keyframe = m_obj->m_mp_keyframes[connect_kf_id];
                auto connected_mappoints_id = keyframe->m_connect_mappoints;
                for (auto mappoint_id : connected_mappoints_id) {
                    if (m_obj->m_pointclouds_map.find(mappoint_id) ==
                        m_obj->m_pointclouds_map.end()) {
                        // not in the boundingbox
                        continue;
                    }
                    associated_mappoints.insert(
                        m_obj->m_pointclouds_map[mappoint_id]);
                    associated_mappoints_ids.insert(mappoint_id);
                }
            } else {
                VLOG(0) << "the keyframe is not exist";
            }
        }
    }

    m_associated_mappoints_vector.clear();
    for (const auto &mappoint : associated_mappoints) {
        m_associated_mappoints_vector.emplace_back(mappoint);
    }
    VLOG(4) << "associated mappoints num: "
            << m_associated_mappoints_vector.size();
    VLOG(4) << "associated keyframe num: " << associated_keyframe_ids.size();

    if (m_associated_mappoints_vector.empty()) {
        LOG(ERROR) << "there is no connected computed mappoints";
        return matches_3d;
    }

    // associated_mappoints
    m_obj->SetAssociatedMapPointsByConnection(associated_mappoints_ids);
    m_obj->SetAssociatedKeyFrames(associated_keyframe_ids);

    cv::Mat pcDesp;

    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();
    auto fx = static_cast<float>(Kcv.at<double>(0, 0));
    auto fy = static_cast<float>(Kcv.at<double>(1, 1));
    auto cx = static_cast<float>(Kcv.at<double>(0, 2));
    auto cy = static_cast<float>(Kcv.at<double>(1, 2));

    pcDesp = ObjDetectionCommon::GetPointCloudDespByConnection(
        m_associated_mappoints_vector);

    cv::Mat frmDesp;
    frmDesp = m_frame_cur->m_desp;

    std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    std::map<int, MapPointIndex> matches2dTo3d;
    matches2dTo3d.clear();
    m_knn_match_num = -1;

    std::vector<cv::DMatch> goodMatches;
#ifdef SUPERPOINT
    // TODO(zhangye): use only norm2 distance for superpoint match?

    ObjDetectionCommon::FindMatchByKNN_SuperPoint(frmDesp, pcDesp, goodMatches);

#else
    const float ratio_threshold = 0.70;
    ObjDetectionCommon::FindMatchByKNN(
        frmDesp, pcDesp, goodMatches, ratio_threshold);
#endif

    STATISTICS_UTILITY::StatsCollector detector_find_3d_by_connection_time(
        "Time: detector find 3d match by connection");
    detector_find_3d_by_connection_time.AddSample(timer.Stop());

    for (int i = 0; i < goodMatches.size(); i++) {
        matches2dTo3d.insert(std::pair<int, MapPointIndex>(
            goodMatches[i].queryIdx, goodMatches[i].trainIdx));
    }

    m_knn_match_num = goodMatches.size();
    VLOG(5) << "detection 2D-3D match: " << goodMatches.size();

    STATISTICS_UTILITY::StatsCollector stats_collector_knn(
        "detector 2D-3D matches num");
    stats_collector_knn.AddSample(m_knn_match_num);

    // 8
    const int kGoodMatchNumTh =
        Parameters::GetInstance().kDetectorKNNMatchNumTh;
    if (m_knn_match_num < kGoodMatchNumTh) {
        VLOG(5) << "The number of KNN 2D-3D match is small";
        return matches_3d;
    }

    std::vector<cv::Point3f> ptsObj;
    std::vector<cv::Point2f> ptsImg;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        ptsImg.emplace_back(cv::Point2f(keyPoints[iter->first].pt));
        cv::Point3f mapPoint = cv::Point3f(
            m_associated_mappoints_vector[iter->second]->GetPose()(0),
            m_associated_mappoints_vector[iter->second]->GetPose()(1),
            m_associated_mappoints_vector[iter->second]->GetPose()(2));
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

    pcDesp = ObjDetectionCommon::GetPointCloudDesp(m_obj);

    cv::Mat frmDesp;
    frmDesp = m_frame_cur->m_desp;

    std::vector<MapPoint::Ptr> pointClouds = m_obj->GetPointClouds();
    std::vector<cv::KeyPoint> keyPoints = m_frame_cur->m_kpts;
    std::map<int, MapPointIndex> matches2dTo3d;
    matches2dTo3d.clear();
    m_knn_match_num = -1;

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

    m_knn_match_num = goodMatches.size();
    VLOG(5) << "detection 2D-3D match: " << goodMatches.size();

    STATISTICS_UTILITY::StatsCollector stats_collector_knn(
        "detector 2D-3D matches num");
    stats_collector_knn.AddSample(m_knn_match_num);

    STATISTICS_UTILITY::StatsCollector detector_find_3d_time(
        "Time: detector find 3d match");
    detector_find_3d_time.AddSample(timer.Stop());

    // 8
    const int kGoodMatchNumTh =
        Parameters::GetInstance().kDetectorKNNMatchNumTh;
    if (m_knn_match_num < kGoodMatchNumTh) {
        VLOG(5) << "The number of KNN 2D-3D match is small";
        return matches_3d;
    }

    std::vector<cv::Point3f> ptsObj;
    std::vector<cv::Point2f> ptsImg;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        ptsImg.emplace_back(cv::Point2f(keyPoints[iter->first].pt));
        cv::Point3f mapPoint = cv::Point3f(
            pointClouds[iter->second]->GetPose()(0),
            pointClouds[iter->second]->GetPose()(1),
            pointClouds[iter->second]->GetPose()(2));
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

bool PointCloudObjDetector::PoseSolver(
    const PS::MatchSet3D &matches_3d,
    const std::vector<PS::MatchSet2D> &matches_2d, std::vector<int> &inliers_3d,
    std::vector<std::vector<int>> &inliers_2d) {
    PS::Options options;
    PS::Pose T;
    m_pnp_inliers_num = 0;
    m_pnp_inliers_3d_num = 0;
    m_pnp_inliers_2d_num = 0;

    const float kPnpReprojectionError = 4.0;
    const cv::Mat Kcv = CameraIntrinsic::GetInstance().GetCVK();

    options.focal_length = static_cast<float>(Kcv.at<double>(0, 0));
    options.max_reproj_err = kPnpReprojectionError / options.focal_length;
    options.enable_2d_solver = true;
    options.enable_3d_solver = true;
    options.ransac_iterations = 100;
    options.ransac_confidence = 0.85;
    options.prefer_pure_2d_solver = false;
    options.try_refine_translation_before_optimization_for_2d_only_matches =
        true;
    const int kPnpMinMatchesNum = 0;

#ifdef OBJ_WITH_KF
    int kPnpMinInlierNum =
        Parameters::GetInstance().kDetectorPnPInliersGoodWithKFNumTh_PoseSolver;
#else
    int kPnpMinInlierNum =
        Parameters::GetInstance().kDetectorPnPInliersGoodNumTh;
#endif

#ifdef SUPERPOINT
#ifdef OBJECT_TOY
    kPnpMinInlierNum = m_knn_match_num * 0.13;
#endif

#endif
    const double kPnpMinInlierRatio = 0.0;
    options.callbacks.emplace_back(PS::EarlyBreakBy3DInlierCounting(
        kPnpMinMatchesNum, kPnpMinInlierNum, kPnpMinInlierRatio));
    options.CheckValidity();

    if (matches_3d.empty()) {
        m_pnp_solver_result = false;
    }
    m_pnp_solver_result = PS::Ransac(
        options, matches_3d, matches_2d, &T, &inliers_3d, &inliers_2d);

    Eigen::Matrix3d Rco = T.m_R.cast<double>();
    Eigen::Vector3d tco = T.m_t.cast<double>();
    m_Rco_cur = Rco;
    m_tco_cur = tco;
#ifdef MONO
    if (m_obj->GetScale() <= 1e-5) {
        m_tco_cur_scale = tco;
    } else {
        m_tco_cur_scale = tco / m_obj->GetScale();
    }
#else
    m_tco_cur_scale = tco;
#endif

    // LBA??

    Eigen::Matrix3d Row = Rco.transpose() * (m_Rcw_cur);
    Eigen::Vector3d Tow = Rco.transpose() * (m_tcw_cur - m_tco_cur_scale);
    m_Rwo_cur = Row.transpose();
    m_two_cur = -m_Rwo_cur * Tow;

    m_pnp_inliers_num = inliers_3d.size();
    m_pnp_inliers_3d_num = m_pnp_inliers_num;
    for (const auto &it : inliers_2d) {
        m_pnp_inliers_num += it.size();
        m_pnp_inliers_2d_num += it.size();
    }

    VLOG(5) << "detection inlier num 2d+3d: " << m_pnp_inliers_num;
    STATISTICS_UTILITY::StatsCollector stats_collector_pnp_3d(
        "detector PnP 3d inlier num");
    stats_collector_pnp_3d.AddSample(m_pnp_inliers_3d_num);
    STATISTICS_UTILITY::StatsCollector stats_collector_pnp_2d(
        "detector PnP 2d inlier num");
    stats_collector_pnp_2d.AddSample(m_pnp_inliers_2d_num);

    return m_pnp_solver_result;
}

void PointCloudObjDetector::PnPResultHandle() {
#ifdef USE_INLIER
#ifdef OBJ_WITH_KF
    // 50
    int kDetectorPnPInliersThGood =
        Parameters::GetInstance().kDetectorPnPInliersGoodWithKFNumTh;
    // 20
    int kDetectorPnPInliersThUnreliable =
        Parameters::GetInstance().kDetectorPnPInliersUnreliableWithKFNumTh;
#else
    const int kDetectorPnPInliersThGood =
        Parameters::GetInstance().kDetectorPnPInliersGoodNumTh;
    const int kDetectorPnPInliersThUnreliable =
        Parameters::GetInstance().kDetectorPnPInliersUnreliableNumTh;
#endif

#ifdef OBJECT_BOX
#ifdef SUPERPOINT
#ifdef MONO
    int proj_success_num = 50;
#else
    int proj_success_num = 80;
#endif
#else
    int proj_success_num = m_knn_match_num * 0.2;
#endif
#endif

#ifdef OBJECT_BAG
#ifdef SUPERPOINT
#ifdef MONO
    int proj_success_num = 30;
#else
    int proj_success_num = 60;
#endif
#else
    int proj_success_num = m_knn_match_num * 0.2;
#endif
#endif

#ifdef OBJECT_TOY
#ifdef SUPERPOINT
#ifdef MONO
    int proj_success_num = 40;
#else
    int proj_success_num = m_knn_match_num * 0.2;
#endif
#else
    int proj_success_num = m_knn_match_num * 0.2;
#endif
#endif

#ifdef USE_OLNY_SCAN_MAPPOINT
    kDetectorPnPInliersThGood = 20;
    kDetectorPnPInliersThUnreliable = 10;
#endif

#ifdef MONO
    VLOG(0) << "m_scale" << m_scale_num << " " << m_obj->GetScale();
#endif

    if (m_pnp_solver_result && m_pnp_inliers_num >= kDetectorPnPInliersThGood &&
        m_pnp_inliers_3d_num >= proj_success_num) {
        m_detect_state = DetectionGood;
#ifdef MONO
        if (m_scale_num <= 100) {
            if (m_last_detectionGood_R == Eigen::Matrix3d::Identity()) {
                m_last_detectionGood_R = m_Rco_cur;
                m_last_detectionGood_t = m_tco_cur;
                m_last_slam_R = m_Rcw_cur;
                m_last_slam_t = m_tcw_cur;
            } else {
                Eigen::Matrix4d last_detection = Eigen::Matrix4d::Identity();
                last_detection.block<3, 3>(0, 0) =
                    m_last_detectionGood_R.normalized();
                last_detection.block<3, 1>(0, 3) = m_last_detectionGood_t;

                Eigen::Matrix4d new_detection = Eigen::Matrix4d::Identity();
                new_detection.block<3, 3>(0, 0) = m_Rco_cur.normalized();
                new_detection.block<3, 1>(0, 3) = m_tco_cur;

                Eigen::Matrix4d delta_scan =
                    new_detection * last_detection.inverse();

                Eigen::Matrix4d last_slam = Eigen::Matrix4d::Identity();
                last_slam.block<3, 3>(0, 0) = m_last_slam_R.normalized();
                last_slam.block<3, 1>(0, 3) = m_last_slam_t;

                Eigen::Matrix4d new_slam = Eigen::Matrix4d::Identity();
                new_slam.block<3, 3>(0, 0) = m_Rcw_cur.normalized();
                new_slam.block<3, 1>(0, 3) = m_tcw_cur;

                Eigen::Matrix4d delta_slam = new_slam * last_slam.inverse();

                VLOG(0) << "delta scan: " << delta_scan;
                VLOG(0) << "delta slam: " << delta_slam;

                double scale_scan_slam = delta_scan.block<3, 1>(0, 3).norm() /
                                         delta_slam.block<3, 1>(0, 3).norm();

                if (m_scale_num) {
                    m_obj->SetScale(1.0);
                }
                m_scale = m_scale + scale_scan_slam;
                m_scale_num++;
            }
        }
#endif
        // TODO(zhangye) check!
        m_obj->SetPoseForFindSimilarKeyframe(m_Rcw_cur, m_tcw_cur);
        m_obj->SetPose(
            m_frame_cur->m_frame_index, m_detect_state, m_Rcw_cur, m_tcw_cur,
            m_Rwo_cur, m_two_cur);

        // evaluation method
        if (m_reproj_error >= 0) {
            STATISTICS_UTILITY::StatsCollector stats_collector_project(
                "detector inlier reproj error");
            stats_collector_project.AddSample(m_reproj_error);
        }
    } else {
        if (m_pnp_inliers_num >= kDetectorPnPInliersThUnreliable) {
            VLOG(5) << "detector PnP fail but has enough inliers";
            m_detect_state = DetectionUnreliable;
            if (!m_has_good_result) {
                m_obj->SetPoseForFindSimilarKeyframe(m_Rcw_cur, m_tcw_cur);
            }
            m_obj->SetPose(
                m_frame_cur->m_frame_index, m_detect_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        } else {
            VLOG(5) << "detector PnP solve fail!";
            m_detect_state = DetectionBad;
            if (!m_has_good_result) {
                m_obj->SetPoseForFindSimilarKeyframe(m_Rcw_cur, m_tcw_cur);
            }
            m_obj->SetPose(
                m_frame_cur->m_frame_index, m_detect_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        }
    }
#endif

#ifdef USE_REPROJ
    if (m_pnp_solver_result && m_reproj_error <= 3.0) {
        m_detect_state = DetectionGood;
        m_obj->SetPose(
            m_frame_cur->m_frame_index, m_detect_state, m_Rcw_cur, m_tcw_cur,
            m_Rwo_cur, m_two_cur);
    } else {
        if (m_pnp_inliers_num <= 4.5) {
            VLOG(5) << "detector PnP fail but has enough inliers";
            m_detect_state = DetectionUnreliable;
            m_obj->SetPose(
                m_frame_cur->m_frame_index, m_detect_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
        } else {
            VLOG(5) << "detector PnP solve fail!";
            m_detect_state = DetectionBad;
            m_obj->SetPose(
                m_frame_cur->m_frame_index, m_detect_state, m_Rcw_cur,
                m_tcw_cur, m_Rwo_cur, m_two_cur);
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
        "3d|2d inliers num: " + std::to_string(m_pnp_inliers_3d_num) + "|" +
        std::to_string(m_pnp_inliers_2d_num) + "| ";
    std::string reproj_txt = "reproj error: " + std::to_string(m_reproj_error);

    std::string detectionStateString;
    if (m_detect_state == DetectionGood) {
        detectionStateString = "Good";

    } else if (m_detect_state == DetectionUnreliable) {
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
    const std::vector<MapPoint::Ptr> pointClouds = m_obj->GetPointClouds();
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
    ObjDetectionCommon::GetPointCloudBoundingBox(m_obj, mapPointBoundingBox);
    std::vector<cv::Point2d> boxProjResult;
    for (int i = 0; i < mapPointBoundingBox.size(); i++) {
        const Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();
        Eigen::Vector3d p =
            K * (m_Rco_cur * mapPointBoundingBox[i] + m_tco_cur_scale);
        cv::Point2d pResult;
        pResult.x = p(0) / p(2);
        pResult.y = p(1) / p(2);
        boxProjResult.emplace_back(pResult);
    }

    cv::Scalar corner_color = cv::Scalar(0, 255, 255);
    cv::Scalar edge_color;
    if (m_detect_state == DetectionGood) {
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
    cv::Mat img_text;
    DrawTextInfo(showResult, img_text);
    GlobalOcvViewer::AddView("Detector Result", img_text);
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
#ifdef USE_CONNECT_FOR_DETECTOR

            auto mappoint = m_associated_mappoints_vector[matches.second];
#else
            auto mappoint = m_obj->GetPointClouds()[matches.second];
#endif

            auto keypoint = m_frame_cur->m_kpts[matches.first];

            auto proj = CameraIntrinsic::GetInstance().GetEigenK() *
                        (m_Rco_cur * mappoint->GetPose() + m_tco_cur_scale);

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
    const std::shared_ptr<ObjRecognition::FrameForObjRecognition> &frm) {
    if (frm == nullptr || (frm != nullptr && frm->m_img.data == nullptr) ||
        m_obj == nullptr) {
        return;
    }

    TIMER_UTILITY::Timer timer;
    PreProcess(frm);

    std::vector<PS::MatchSet2D> matchset_2d;
    std::vector<KeyFrame::Ptr> kf_mathceds;
#ifdef USE_OLNY_SCAN_MAPPOINT
#else
#ifdef OBJ_WITH_KF
    auto allKFs = m_obj->GetKeyFrames();
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

    m_reproj_error = ComputeAverageReProjError(inliers_3d);

    PnPResultHandle();

    STATISTICS_UTILITY::StatsCollector detector_process_time(
        "Time: detector process single image");
    detector_process_time.AddSample(timer.Stop());

    ShowDetectResult();

    ResultRecord();
}

void PointCloudObjDetector::Reset() {
    Clear();
}

void PointCloudObjDetector::Clear() {
    m_knn_match_num = 0;
    m_pnp_inliers_num = 0;
    m_detect_state = DetectionBad;
    m_has_good_result = false;
    m_scale = 0.0;
    m_scale_num = 0;
}

bool PointCloudObjDetector::Load(const long long &mem_size, const char *mem) {
    return m_obj->LoadPointCloud(mem_size, mem);
}

bool PointCloudObjDetector::Save(long long &mem_size, char **mem) {
    return m_obj->Save(mem_size, mem);
}

void PointCloudObjDetector::ResultRecord() {
    if (m_detect_state == DetectionGood) {
        STATISTICS_UTILITY::StatsCollector pointCloudDetectionNum(
            "detector good num");
        pointCloudDetectionNum.IncrementOne();
    }
}
} // namespace ObjRecognition
