//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_TRACKERCOMMON_H
#define ORB_SLAM3_TRACKERCOMMON_H
#include <vector>
#include <opencv2/core/types.hpp>
#include "Struct/PointCloudObject.h"
#include "Struct/Frame.h"

namespace ObjRecognition {
namespace ObjTrackerCommon {
void KeyPointsToPoints(
    const std::vector<cv::KeyPoint> &kPts, std::vector<cv::Point2d> &pts);

void GetCoordsInBorder(
    Eigen::Vector2d &pt, const int &xMin, const int &yMin, const int &xMax,
    const int &yMax);

bool InBorder(
    const Eigen::Vector2d &pt, const int &xMin, const int &yMin,
    const int &xMax, const int &yMax);

void Project(
    const std::vector<Eigen::Vector3d> &pointCloudsWorld,
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &Tcw,
    std::vector<bool> &projectFailState,
    std::vector<Eigen::Vector2d> &projectPoints, bool isBox);
void GetMapPointPositions(
    const std::vector<MapPoint::Ptr> &pointClouds, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &Two, std::vector<Eigen::Vector3d> &mapPointsObj,
    std::vector<Eigen::Vector3d> &mapPointsWorld);
void GetFeaturesInArea(
    const Eigen::Vector2d &point, const int &width, const int &height,
    const std::vector<cv::KeyPoint> &keyPoints, std::vector<int> &vIndices);

int SearchByProjection(
    const std::vector<Eigen::Vector2d> &projectPoints,
    const std::vector<MapPoint::Ptr> &pointClouds,
    const std::vector<bool> &projectFailState,
    const std::vector<cv::KeyPoint> &keyPoints, const cv::Mat &descriptors,
    std::vector<bool> &matchKeyPointsState,
    std::map<int, MapPointIndex> &matches2dTo3d);

bool SolvePnP(
    const std::map<int, MapPointIndex> &matches2dTo3d,
    const std::vector<cv::Point2d> &keyPoints,
    const std::vector<Eigen::Vector3d> &pointClouds3dObj,
    const Eigen::Matrix3d &initialRco, const Eigen::Vector3d &initialTco,
    const cv::Mat &Kcv, Eigen::Matrix3d &resultRco, Eigen::Vector3d &resultTco,
    std::map<int, MapPointIndex> &matches2dTo3dNew, int &inlierNum);

void DrawBoundingBox(
    const cv::Mat &showResult, std::vector<cv::Point2d> &boxProjResult,
    cv::Scalar &color);

void ExtractKeyPointsAndDes(
    const std::shared_ptr<ObjRecognition::FrameData> &frm,
    std::vector<cv::KeyPoint> &imgKeyPoints, cv::Mat &imgDescriptor);

void GetPointCloudBoundingBox(
    const std::shared_ptr<Object> &obj,
    std::vector<Eigen::Vector3d> &mapPointBoundingBox);
} // namespace ObjTrackerCommon
} // namespace ObjRecognition
#endif // ORB_SLAM3_TRACKERCOMMON_H
