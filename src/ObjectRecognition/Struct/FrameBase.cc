//
// Created by zhangye on 2020/9/16.
//
#include "Struct/FrameBase.h"
namespace ObjRecognition {

void FrameBase::SetCameraPose(
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw) {
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    m_Rcw = Rcw;
    m_tcw = tcw;
}

void FrameBase::GetCameraPose(Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw) {
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    Rcw = m_Rcw;
    tcw = m_tcw;
}

void FrameBase::SetObjectPose(
    const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two) {
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    m_Rwo = Rwo;
    m_two = two;
}

void FrameBase::GetObjectPose(Eigen::Matrix3d &Rwo, Eigen::Vector3d &two) {
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    Rwo = m_Rwo;
    two = m_two;
}

void FrameBase::SetObjectPoseInCamemra(
    const Eigen::Matrix3d &Rco, const Eigen::Vector3d &tco) {
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    m_Rco = Rco;
    m_tco = tco;
}

void FrameBase::GetObjectPoseInCamera(
    Eigen::Matrix3d &Rco, Eigen::Vector3d &tco) {
    std::unique_lock<std::mutex> lock(m_mutex_pose);
    Rco = m_Rco;
    tco = m_tco;
}

} // namespace ObjRecognition