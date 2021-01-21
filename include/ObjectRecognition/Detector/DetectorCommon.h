#ifndef ORB_SLAM3_DETECTORCOMMON_H
#define ORB_SLAM3_DETECTORCOMMON_H
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <memory>
#include "Frame.h"
#include "PointCloudObject.h"

namespace ObjRecognition {
namespace ObjDetectionCommon {
cv::Mat GetPointCloudDesp(const std::shared_ptr<Object> &pc);

cv::Mat GetPointCloudDespByConnection(
    const std::vector<MapPoint::Ptr> &associated_mappoints);

void DrawBoundingBox(
    const cv::Mat &showResult, std::vector<cv::Point2d> &boxProjResult,
    cv::Scalar &color);

void GetPointCloudBoundingBox(
    const std::shared_ptr<Object> &obj,
    std::vector<Eigen::Vector3d> &mapPointBoundingBox);

void FindMatchByKNN_Homography(
    const std::vector<cv::KeyPoint> &frmKeypoints,
    const std::vector<cv::KeyPoint> &kfKeyPoints, const cv::Mat &frmDesp,
    const cv::Mat &pcDesp, std::vector<cv::DMatch> &goodMatches,
    float ratio_threshold);

void FindMatchByKNN(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches, float ratio_threshold);

void FindMatchByKNN_SuperPoint_Homography(
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &frmDesp,
    const cv::Mat &pcDesp, std::vector<cv::DMatch> &goodMatches);

void FindMatchByKNN_SuperPoint(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches);
} // namespace ObjDetectionCommon
} // namespace ObjRecognition

#endif // ORB_SLAM3_DETECTORCOMMON_H
