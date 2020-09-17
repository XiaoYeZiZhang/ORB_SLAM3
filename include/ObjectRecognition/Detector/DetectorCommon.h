//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_DETECTORCOMMON_H
#define ORB_SLAM3_DETECTORCOMMON_H
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <memory>
#include "Struct/Frame.h"
#include "Struct/PointCloudObject.h"

namespace ObjRecognition {
namespace ObjDetectionCommon {
cv::Mat GetPointCloudDesp(const std::shared_ptr<Object> &pc);
void GetBoxPoint(
    const std::shared_ptr<Object> &mObj,
    std::vector<Eigen::Vector3d> &pointBoxs);
void FindMatchByKNN(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches);
std::vector<cv::Mat> ToDescriptorVector(const cv::Mat &Descriptors);
/*void FindMatchByBow(
    const cv::Mat &pcDesp, const cv::Mat &frmDesp, DBoW3::Vocabulary *&voc,
    std::map<int, MapPointIndex> &matches2dTo3d);*/
Eigen::Isometry3f
GetTMatrix(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
void DrawBox(
    cv::Mat &imgRGB, const Eigen::Isometry3f &T,
    const std::vector<Eigen::Vector3d> &pointBoxs);
void ShowDetectResult(
    const std::shared_ptr<ObjRecognition::FrameData> &frm,
    const std::shared_ptr<Object> &mObj, const Eigen::Isometry3f &T,
    const ObjRecogState &detectState,
    const std::map<int, MapPointIndex> &matches2dTo3d);
void GetMaskKeypointAndDesp(
    const cv::Mat &image,
    const std::shared_ptr<ObjRecognition::FrameData> &frm);
} // namespace ObjDetectionCommon
} // namespace ObjRecognition

#endif // ORB_SLAM3_DETECTORCOMMON_H
