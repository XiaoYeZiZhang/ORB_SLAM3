//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_FRAMEBASE_H
#define ORB_SLAM3_FRAMEBASE_H
#include <Eigen/Core>
#include <mutex>
#include <opencv2/core/mat.hpp>
namespace ObjRecognition {
class FrameBase {
public:
    void SetCameraPose(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);
    void GetCameraPose(Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);

    void SetObjectPose(const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two);
    void GetObjectPose(Eigen::Matrix3d &Rwo, Eigen::Vector3d &two);

    void SetObjectPoseInCamemra(
        const Eigen::Matrix3d &Rco, const Eigen::Vector3d &tco);
    void GetObjectPoseInCamera(Eigen::Matrix3d &Rco, Eigen::Vector3d &tco);

public:
    int m_frame_index = 0;
    double m_time_stamp = 0;
    cv::Mat m_raw_image;
    cv::Mat m_desp;
    std::vector<cv::KeyPoint> m_kpts;

private:
    Eigen::Matrix3d m_Rcw;
    Eigen::Vector3d m_tcw;
    Eigen::Matrix3d m_Rco;
    Eigen::Vector3d m_tco;
    Eigen::Matrix3d m_Rwo;
    Eigen::Vector3d m_two;

    std::mutex m_mutex_pose;

}; // class FrameBase

} // namespace ObjRecognition
#endif // ORB_SLAM3_FRAMEBASE_H
