//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_TRACKERPOINTCLOUD_H
#define ORB_SLAM3_TRACKERPOINTCLOUD_H
#include "Struct/PointCloudObject.h"
#include "PoseSolver/PoseSolver.h"
#include "Utility/RecognitionBase.h"
#include "Tracker/TrackerFrame.h"
namespace ObjRecognition {

class PointCloudObjTracker : public RecognitionBase {
public:
    PointCloudObjTracker();
    ~PointCloudObjTracker();

    void SetPointCloudObj(const std::shared_ptr<Object> &pointCloudPtr);
    void Process(const std::shared_ptr<FrameData> &frm);
    void Reset();
    void Clear();
    bool Load(const long long &mem_size, const char *mem);
    bool Save(long long &mem_size, char **mem);
    void SetInfo();
    int GetInfo(std::string &info);

private:
    void PreProcess(const std::shared_ptr<FrameData> &frm);

    PS::MatchSet3D FindProjection3DMatch();
    void PnPResultHandle();
    void ResultRecord();

    PS::MatchSet3D FindOpticalFlow3DMatch();
    bool PoseSolver(
        const PS::MatchSet3D &matches_3d,
        const std::vector<PS::MatchSet2D> &matches_2d,
        std::vector<int> &inliers_3d);
    void ProcessPoseSolverInliers(const std::vector<int> &inliers_3d);
    void ShowProjectedPointsAndMatchingKeyPoints(
        std::vector<Eigen::Vector2d> &projectPoints,
        std::vector<bool> &matchKeyPointsState);
    void OpticalFlowRejectWithF(
        std::vector<cv::Point2d> &ptsPre,
        std::vector<MapPointIndex> &mapPointIndexes);
    void RemoveOpticalFlow3dMatchOutliers(
        const std::vector<uchar> &status,
        const std::vector<cv::Point2f> &points_cur);
    void ShowOpticalFlowpoints(
        const std::vector<cv::Point2d> &opticalFlowKeyPointsPre);
    void ShowTrackerResult();
    void DrawTextInfo(const cv::Mat &img, cv::Mat &img_txt);
    float ComputeAverageReProjError(const std::vector<int> &inliers_3d);

private:
    std::shared_ptr<Object> mObj;
    std::shared_ptr<TrackerFrame> m_frame_cur;
    std::shared_ptr<TrackerFrame> m_frame_Pre;
    bool m_pnp_solver_result = false;
    int m_project_success_mappoint_num = 0;
    int m_match_points_num = 0;
    int m_match_points_projection_num = 0;
    int m_match_points_opticalFlow_num = 0;
    int m_pnp_inliers_num = 0;
    int m_pnp_inliers_projection_num = 0;
    int m_pnp_inliers_opticalFlow_num = 0;
    ObjRecogState m_tracker_state;

    Eigen::Matrix3d Rcw_cur_;
    Eigen::Vector3d tcw_cur_;
    Eigen::Matrix3d Rwo_cur_;
    Eigen::Vector3d two_cur_;
    Eigen::Matrix3d Rco_cur_;
    Eigen::Vector3d tco_cur_;

    std::map<int, MapPointIndex> m_projection_matches2dTo3d_cur;
    std::vector<cv::Point2d> m_projection_points2d_cur;
    std::map<int, MapPointIndex> m_projection_matches2dTo3d_inlier;
    std::map<int, MapPointIndex> m_opticalFlow_matches2dTo3d_inlier;

    std::string m_info;
    bool m_first_detection_good;

    float reproj_error;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_TRACKERPOINTCLOUD_H
