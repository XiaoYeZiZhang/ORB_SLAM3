//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_LBAOPTIMIZER_H
#define ORB_SLAM3_LBAOPTIMIZER_H
#define SLIGHT_ANGLE 0.008726646
#define POSE_SIZE 7
#define LANDMARK_SIZE 3
#include "Struct/PointCloudObject.h"
typedef long unsigned int MapPointIndex;
namespace ObjRecognition {
class LBAOptimizer {
public:
    LBAOptimizer();
    void Reset();
    bool PoseCeresOptimization(
        const std::vector<cv::KeyPoint> &keyPoints,
        const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
        const std::map<int, MapPointIndex> &matches2dTo3d,
        const Eigen::Matrix3d &K, std::vector<Eigen::Matrix3d> &Rcos,
        std::vector<Eigen::Vector3d> &Tcos);

protected:
    bool OptimizeVO(
        const std::vector<cv::KeyPoint> &keyPoints,
        const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
        const std::map<int, MapPointIndex> &matches2dTo3d,
        const Eigen::Matrix3d &K, std::vector<Eigen::Matrix3d> &Rcos,
        std::vector<Eigen::Vector3d> &Tcos);
    bool OptimizeVOLBA(
        const std::vector<cv::KeyPoint> &keyPoints,
        const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
        const std::map<int, MapPointIndex> &matches2dTo3d,
        const Eigen::Matrix3d &K, std::vector<Eigen::Matrix3d> &Rcos,
        std::vector<Eigen::Vector3d> &Tcos);
    void PrepareVODataForCeres(
        const std::vector<cv::KeyPoint> &keyPoints,
        const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
        const std::map<int, MapPointIndex> &matches2dTo3d,
        const Eigen::Matrix3d &K, std::vector<Eigen::Matrix3d> &Rcos,
        std::vector<Eigen::Vector3d> &Tcos);
    void WriteVOBackParams(
        std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos);

    struct LBADebugError {
        /// reprojection error for all m_solved_frames
        double reprojection_error{DBL_MAX};
        /// reprojection error for latest frame
        double reprojection_error_latest_frm{DBL_MAX};
    };
    LBADebugError CalculateLBADebugError(
        const std::vector<cv::KeyPoint> &keyPoints,
        const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
        const std::map<int, MapPointIndex> &matches2dTo3d,
        const Eigen::Matrix3d &K, std::vector<Eigen::Matrix3d> &Rcos,
        std::vector<Eigen::Vector3d> &Tcos);

private:
    /// ceres options
    double m_max_solve_time;
    std::vector<std::array<double, POSE_SIZE>> para_Pose_fix;
    std::vector<std::array<double, POSE_SIZE>> para_Pose;
    std::vector<std::array<double, LANDMARK_SIZE>> para_landmark_fix;
    /// BA optimize parameters of update pose
    int m_optimize_camera_size;
    int m_optimize_count;
    int m_fixed_camera_size;
    double covariance_xx[6 * 6];
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_LBAOPTIMIZER_H
