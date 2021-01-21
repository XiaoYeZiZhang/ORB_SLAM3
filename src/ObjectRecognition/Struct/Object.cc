#include "Struct/Object.h"
#include <glog/logging.h>
#include "mode.h"

namespace ObjRecognition {

void ObjStateStruct::GetData(
    Eigen::Matrix3d &Rwo, Eigen::Matrix3d &Rcw, Eigen::Vector3d &two,
    Eigen::Vector3d &tcw, ObjRecogState &State, FrameIndex &frmIndex) {

    std::lock_guard<std::mutex> lck(m_stateMutex);
    Rwo = m_Rwo;
    Rcw = m_Rcw;
    two = m_two;
    tcw = m_tcw;
    State = m_state;
    frmIndex = m_frmIndex;
}

void ObjStateStruct::SetData(
    const Eigen::Matrix3d &Rwo, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &two, const Eigen::Vector3d &tcw,
    const ObjRecogState &State, const FrameIndex &FrmIndex) {
    std::lock_guard<std::mutex> lck(m_stateMutex);
    m_Rwo = Rwo;
    m_Rcw = Rcw;
    m_two = two;
    m_tcw = tcw;
    m_state = State;
    m_frmIndex = FrmIndex;
}

ObjRecogState ObjStateStruct::GetState() {
    std::lock_guard<std::mutex> lck(m_stateMutex);
    return m_state;
}

void ObjStateStruct::Reset(const ObjRecogState &state) {
    std::lock_guard<std::mutex> lck(m_stateMutex);
    m_Rwo = Eigen::Matrix3d::Identity();
    m_Rcw = Eigen::Matrix3d::Identity();
    m_two = Eigen::Vector3d::Zero();
    m_tcw = Eigen::Vector3d::Zero();
    m_state = state;
    m_frmIndex = -1;
}

ObjectBase::ObjectBase(int id) : m_Id(id) {
    m_tracker_state.Reset(TrackingBad);
    m_detector_state.Reset(DetectionBad);

    m_tracking_bad_voting_count = 0;
    m_detection_bad_voting_count = 0;
    m_boundingbox.reserve(8);
    m_scale = 0.0;
}

void ObjectBase::TrackingStateSetPose(
    const ObjRecogState &trackerState, const FrameIndex &frmIndex,
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
    const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two) {

    if (trackerState == TrackingGood) {
        m_tracking_bad_voting_count = 0;
        m_tracker_state.SetData(Rwo, Rcw, two, tcw, TrackingGood, frmIndex);
    } else if (
        trackerState == TrackingUnreliable &&
        m_tracker_state.GetState() != TrackingGood) {
        m_tracker_state.SetData(
            Rwo, Rcw, two, tcw, TrackingUnreliable, frmIndex);
    } else {
        if (m_tracking_bad_voting_count >= 4) {
            m_tracker_state.SetData(Rwo, Rcw, two, tcw, trackerState, frmIndex);
            m_tracking_bad_voting_count = 0;
        } else {
            if (trackerState == TrackingUnreliable)
                m_tracking_bad_voting_count += 1;
            if (trackerState == TrackingBad)
                m_tracking_bad_voting_count += 2;
        }
    }
}

void ObjectBase::DetectionStateSetPose(
    const ObjRecogState &detectionState, const FrameIndex &frmIndex,
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
    const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two) {

    if (detectionState == DetectionGood) {
        m_detection_bad_voting_count = 0;
        m_detector_state.SetData(Rwo, Rcw, two, tcw, DetectionGood, frmIndex);
    } else if (
        detectionState == DetectionUnreliable &&
        m_detector_state.GetState() != DetectionGood) {
        m_detector_state.SetData(
            Rwo, Rcw, two, tcw, DetectionUnreliable, frmIndex);
    } else {
        if (m_detection_bad_voting_count >= 4) {
            m_detector_state.SetData(
                Rwo, Rcw, two, tcw, detectionState, frmIndex);
            m_detection_bad_voting_count = 0;
        } else {
            if (detectionState == DetectionUnreliable)
                m_detection_bad_voting_count++;
            if (detectionState == DetectionBad)
                m_detection_bad_voting_count += 2;
        }
    }
}

void ObjectBase::SetPoseForFindSimilarKeyframe(
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw) {
    m_Rcw_for_similar_keyframe = Rcw;
    m_tcw_for_similar_keyframe = tcw;
}

void ObjectBase::GetPoseForFindSimilarKeyframe(
    Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw) {
    Rcw = m_Rcw_for_similar_keyframe;
    tcw = m_tcw_for_similar_keyframe;
}

void ObjectBase::SetPose(
    const FrameIndex &frmIndex, const ObjRecogState &state,
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
    const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two) {
    std::lock_guard<std::mutex> lck(m_pose_mutex);

#ifdef USE_NO_METHOD_FOR_FUSE
    if (state == TrackingGood || state == TrackingBad ||
        state == TrackingUnreliable) {
        m_tracker_state.SetData(Rwo, Rcw, two, tcw, state, frmIndex);
    } else {
        m_detector_state.SetData(Rwo, Rcw, two, tcw, state, frmIndex);
    }
#else
    if (state == TrackingGood || state == TrackingBad ||
        state == TrackingUnreliable) {
        TrackingStateSetPose(state, frmIndex, Rcw, tcw, Rwo, two);
    } else {
        DetectionStateSetPose(state, frmIndex, Rcw, tcw, Rwo, two);
    }
#endif
}

void ObjectBase::GetPose(
    FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
    Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, Eigen::Matrix3d &Rwo,
    Eigen::Vector3d &two) {
    std::lock_guard<std::mutex> lck(m_pose_mutex);

#ifdef USE_NO_METHOD_FOR_FUSE
    if (m_tracker_state.GetState() != TrackingBad) {
        m_tracker_state.GetData(Rwo, Rcw, two, tcw, state, frmIndex);
    } else {
        m_detector_state.GetData(Rwo, Rcw, two, tcw, state, frmIndex);
    }

#else
    if (m_tracker_state.GetState() == TrackingGood ||
        (m_tracker_state.GetState() == TrackingUnreliable &&
         m_detector_state.GetState() != DetectionGood)) {
        m_tracker_state.GetData(Rwo, Rcw, two, tcw, state, frmIndex);
    } else if (
        m_detector_state.GetState() == DetectionGood ||
        m_detector_state.GetState() == DetectionUnreliable) {
        m_detector_state.GetData(Rwo, Rcw, two, tcw, state, frmIndex);
        VLOG(5) << "Get Detector Pose: " << Rwo;
    } else {
        m_detector_state.GetData(Rwo, Rcw, two, tcw, state, frmIndex);
        VLOG(5) << "Get Detector Pose: " << Rwo;
        state = DetectionBad;
    }
#endif
}

void ObjectBase::SetScale(const double &scale) {
    m_scale = scale;
}

double ObjectBase::GetScale() {
    return m_scale;
}

std::vector<Eigen::Vector3d> ObjectBase::GetBoundingBox() {
    std::lock_guard<std::mutex> lck(m_boundingbox_mutex);
    return m_boundingbox;
}

void ObjectBase::Reset() {
    m_detector_state.Reset(ObjRecogState::DetectionBad);
    m_tracker_state.Reset(ObjRecogState::TrackingBad);
    m_tracking_bad_voting_count = 0;
    m_detection_bad_voting_count = 0;
}

} // namespace ObjRecognition
