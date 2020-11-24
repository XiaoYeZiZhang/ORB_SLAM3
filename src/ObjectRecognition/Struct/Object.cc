//
// Created by zhangye on 2020/9/16.
//

#include "Struct/Object.h"
#include <glog/logging.h>
#include "mode.h"

namespace ObjRecognition {

void ObjStateStruct::GetData(
    Eigen::Matrix3d &Rwo, Eigen::Matrix3d &Rcw, Eigen::Vector3d &two,
    Eigen::Vector3d &tcw, ObjRecogState &State, FrameIndex &frmIndex,
    double &TimeStamp) {

    std::lock_guard<std::mutex> lck(mStateMutex);

    Rwo = mRwo;
    Rcw = mRcw;
    two = mtwo;
    tcw = mtcw;
    State = mState;
    frmIndex = mFrmIndex;
    TimeStamp = mTimeStamp;
}

void ObjStateStruct::SetData(
    const Eigen::Matrix3d &Rwo, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &two, const Eigen::Vector3d &tcw,
    const ObjRecogState &State, const FrameIndex &FrmIndex,
    const double &TimeStamp) {

    std::lock_guard<std::mutex> lck(mStateMutex);

    mRwo = Rwo;
    mRcw = Rcw;
    mtwo = two;
    mtcw = tcw;
    mState = State;
    mFrmIndex = FrmIndex;
    mTimeStamp = TimeStamp;
}

ObjRecogState ObjStateStruct::GetState() {
    std::lock_guard<std::mutex> lck(mStateMutex);
    return mState;
}

void ObjStateStruct::Reset(const ObjRecogState &state) {

    std::lock_guard<std::mutex> lck(mStateMutex);

    mRwo = Eigen::Matrix3d::Identity();
    mRcw = Eigen::Matrix3d::Identity();
    mtwo = Eigen::Vector3d::Zero();
    mtcw = Eigen::Vector3d::Zero();
    mState = state;
    mFrmIndex = -1;
    mTimeStamp = -1;
}

ObjectBase::ObjectBase(int id) : mId(id) {
    tracker_state_.Reset(TrackingBad);
    detector_state_.Reset(DetectionBad);

    tracking_bad_voting_count_ = 0;
    detection_bad_voting_count_ = 0;
    mvBoundingBox.reserve(8);
}

void ObjectBase::TrackingStateSetPose(
    const ObjRecogState &trackerState, const FrameIndex &frmIndex,
    const double &timeStamp, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &tcw, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &two) {

    if (trackerState == TrackingGood) {
        tracking_bad_voting_count_ = 0;
        tracker_state_.SetData(
            Rwo, Rcw, two, tcw, TrackingGood, frmIndex, timeStamp);
    } else if (
        trackerState == TrackingUnreliable &&
        tracker_state_.GetState() != TrackingGood) {
        tracker_state_.SetData(
            Rwo, Rcw, two, tcw, TrackingUnreliable, frmIndex, timeStamp);
    } else {
        if (tracking_bad_voting_count_ >= 4) {
            tracker_state_.SetData(
                Rwo, Rcw, two, tcw, trackerState, frmIndex, timeStamp);
            tracking_bad_voting_count_ = 0;
        } else {
            if (trackerState == TrackingUnreliable)
                tracking_bad_voting_count_ += 1;
            if (trackerState == TrackingBad)
                tracking_bad_voting_count_ += 2;
        }
    }
}

void ObjectBase::DetectionStateSetPose(
    const ObjRecogState &detectionState, const FrameIndex &frmIndex,
    const double &timeStamp, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &tcw, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &two) {

    if (detectionState == DetectionGood) {
        detection_bad_voting_count_ = 0;
        detector_state_.SetData(
            Rwo, Rcw, two, tcw, DetectionGood, frmIndex, timeStamp);
    } else if (
        detectionState == DetectionUnreliable &&
        detector_state_.GetState() != DetectionGood) {
        detector_state_.SetData(
            Rwo, Rcw, two, tcw, DetectionUnreliable, frmIndex, timeStamp);
    } else {
        if (detection_bad_voting_count_ >= 4) {
            detector_state_.SetData(
                Rwo, Rcw, two, tcw, detectionState, frmIndex, timeStamp);
            detection_bad_voting_count_ = 0;
        } else {
            if (detectionState == DetectionUnreliable)
                detection_bad_voting_count_++;
            if (detectionState == DetectionBad)
                detection_bad_voting_count_ += 2;
        }
    }
}

void ObjectBase::SetPoseForFindSimilarKeyframe(
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
    const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two) {
    Rcw_for_similar_keyframe = Rcw;
    tcw_for_similar_keyframe = tcw;
    Rwo_for_similar_keyframe = Rwo;
    two_for_similar_keyframe = two;
}

void ObjectBase::GetPoseForFindSimilarKeyframe(
    Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, Eigen::Matrix3d &Rwo,
    Eigen::Vector3d &two) {
    Rcw = Rcw_for_similar_keyframe;
    tcw = tcw_for_similar_keyframe;
    Rwo = Rwo_for_similar_keyframe;
    two = two_for_similar_keyframe;
}

void ObjectBase::SetPose(
    const FrameIndex &frmIndex, const double &timeStamp,
    const ObjRecogState &state, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &tcw, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &two) {
    std::lock_guard<std::mutex> lck(mPoseMutex);

#ifdef USE_NO_METHOD_FOR_FUSE
    if (state == TrackingGood || state == TrackingBad ||
        state == TrackingUnreliable) {
        tracker_state_.SetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
    } else {
        detector_state_.SetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
    }
#else
    if (state == TrackingGood || state == TrackingBad ||
        state == TrackingUnreliable) {
        TrackingStateSetPose(state, frmIndex, timeStamp, Rcw, tcw, Rwo, two);
    } else {
        DetectionStateSetPose(state, frmIndex, timeStamp, Rcw, tcw, Rwo, two);
    }
#endif
}

void ObjectBase::GetPose(
    FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
    Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, Eigen::Matrix3d &Rwo,
    Eigen::Vector3d &two) {
    std::lock_guard<std::mutex> lck(mPoseMutex);

#ifdef USE_NO_METHOD_FOR_FUSE
    if (tracker_state_.GetState() != TrackingBad) {
        tracker_state_.GetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
    } else {
        detector_state_.GetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
    }

#else
    if (tracker_state_.GetState() == TrackingGood ||
        (tracker_state_.GetState() == TrackingUnreliable &&
         detector_state_.GetState() != DetectionGood)) {
        tracker_state_.GetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
    } else if (
        detector_state_.GetState() == DetectionGood ||
        detector_state_.GetState() == DetectionUnreliable) {
        detector_state_.GetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
        VLOG(5) << "Get Detector Pose: " << Rwo;
    } else {
        detector_state_.GetData(Rwo, Rcw, two, tcw, state, frmIndex, timeStamp);
        VLOG(5) << "Get Detector Pose: " << Rwo;
        state = DetectionBad;
    }
#endif
}

std::vector<Eigen::Vector3d> ObjectBase::GetBoundingBox() {
    std::lock_guard<std::mutex> lck(mBoundingBoxMutex);
    return mvBoundingBox;
}

void ObjectBase::Reset() {
    detector_state_.Reset(ObjRecogState::DetectionBad);
    tracker_state_.Reset(ObjRecogState::TrackingBad);
    tracking_bad_voting_count_ = 0;
    detection_bad_voting_count_ = 0;
}

} // namespace ObjRecognition
