#ifndef ORB_SLAM3_TRACKERPOINTCLOUD_H
#define ORB_SLAM3_TRACKERPOINTCLOUD_H
#include "Struct/PointCloudObject.h"
#include "PoseSolver/PoseSolver.h"
#include "include/ObjectRecognition/ObjectRecognitionThread/RecognitionBase.h"
#include "Tracker/TrackerFrame.h"
namespace ObjRecognition {

class PointCloudObjTracker : public RecognitionBase {
public:
    PointCloudObjTracker();
    ~PointCloudObjTracker();

    void SetPointCloudObj(const std::shared_ptr<Object> &pointCloudPtr);
    void Process(const std::shared_ptr<FrameForObjRecognition> &frm);
    void Reset();
    void Clear();
    bool Load(const long long &mem_size, const char *mem);
    bool Save(long long &mem_size, char **mem);

private:
    PoseSolver::MatchSet3D FindProjection3DMatch();
    void PnPResultHandle();
    void ResultRecord();

    PoseSolver::MatchSet3D FindOpticalFlow3DMatch();
    bool PoseSolver(
        const PoseSolver::MatchSet3D &matches_3d,
        const std::vector<PoseSolver::MatchSet2D> &matches_2d,
        std::vector<int> &inliers_3d);
    void ProcessPoseSolverInliers(const std::vector<int> &inliers_3d);
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
    int m_match_points_projection_num = 0;
    int m_match_points_opticalFlow_num = 0;
    int m_pnp_inliers_num = 0;
    int m_pnp_inliers_projection_num = 0;
    int m_pnp_inliers_opticalFlow_num = 0;
    ObjRecogState m_tracker_state;

    Eigen::Matrix3d m_Rcw_cur;
    Eigen::Vector3d m_tcw_cur;
    Eigen::Matrix3d m_Rwo_cur;
    Eigen::Vector3d m_two_cur;
    Eigen::Matrix3d m_Rco_cur;
    Eigen::Vector3d m_tco_cur;

    std::map<int, MapPointIndex> m_projection_matches2dTo3d_cur;
    std::vector<cv::Point2d> m_projection_points2d_cur;
    std::map<int, MapPointIndex> m_projection_matches2dTo3d_inlier;
    std::map<int, MapPointIndex> m_opticalFlow_matches2dTo3d_inlier;

    bool m_first_detection_good;

    float m_reproj_error;
    std::vector<cv::Point2d> m_opticalflow_point2ds_tmp;

    double m_scale;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_TRACKERPOINTCLOUD_H
