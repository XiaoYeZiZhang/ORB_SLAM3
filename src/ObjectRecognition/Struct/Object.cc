//
// Created by zhangye on 2020/9/16.
//

#include "Struct/Object.h"

namespace ObjRecognition {

void ObjStateStruct::GetData(
    Eigen::Matrix3d &Rwo, Eigen::Matrix3d &Rcw, Eigen::Vector3d &Two,
    Eigen::Vector3d &Tcw, ObjRecogState &State, FrameIndex &FrmIndex,
    double &TimeStamp) {

    std::lock_guard<std::mutex> lck(mStateMutex);

    Rwo = mRwo;
    Rcw = mRcw;
    Two = mTwo;
    Tcw = mTcw;
    State = mState;
    FrmIndex = mFrmIndex;
    TimeStamp = mTimeStamp;
}

void ObjStateStruct::SetData(
    const Eigen::Matrix3d &Rwo, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &Two, const Eigen::Vector3d &Tcw,
    const ObjRecogState &State, const FrameIndex &FrmIndex,
    const double &TimeStamp) {

    std::lock_guard<std::mutex> lck(mStateMutex);

    mRwo = Rwo;
    mRcw = Rcw;
    mTwo = Two;
    mTcw = Tcw;
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
    mTwo = Eigen::Vector3d::Zero();
    mTcw = Eigen::Vector3d::Zero();
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
    const Eigen::Vector3d &Tcw, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &Two) {

    int bad_voting_param = 2;
    int unreliable_voting_param = 1;

    unreliable_voting_param = 1;
    bad_voting_param = 2;

    if (trackerState == TrackingGood) {
        tracking_bad_voting_count_ = 0;
        tracker_state_.SetData(
            Rwo, Rcw, Two, Tcw, TrackingGood, frmIndex, timeStamp);
    } else if (
        trackerState == TrackingUnreliable &&
        tracker_state_.GetState() != TrackingGood) {
        tracker_state_.SetData(
            Rwo, Rcw, Two, Tcw, TrackingUnreliable, frmIndex, timeStamp);
    } else {
        if (tracking_bad_voting_count_ >= 4) {
            tracker_state_.SetData(
                Rwo, Rcw, Two, Tcw, trackerState, frmIndex, timeStamp);
            tracking_bad_voting_count_ = 0;
        } else {
            if (trackerState == TrackingUnreliable)
                tracking_bad_voting_count_ += unreliable_voting_param;
            if (trackerState == TrackingBad)
                tracking_bad_voting_count_ += bad_voting_param;
        }
    }
}

void ObjectBase::DetectionStateSetPose(
    const ObjRecogState &detectionState, const FrameIndex &frmIndex,
    const double &timeStamp, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &Tcw, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &Two) {

    if (detectionState == DetectionGood) {
        detection_bad_voting_count_ = 0;
        detector_state_.SetData(
            Rwo, Rcw, Two, Tcw, DetectionGood, frmIndex, timeStamp);
    } else if (
        detectionState == DetectionUnreliable &&
        detector_state_.GetState() != DetectionGood) {
        detector_state_.SetData(
            Rwo, Rcw, Two, Tcw, DetectionUnreliable, frmIndex, timeStamp);
    } else {
        if (detection_bad_voting_count_ >= 4) {
            detector_state_.SetData(
                Rwo, Rcw, Two, Tcw, detectionState, frmIndex, timeStamp);
            detection_bad_voting_count_ = 0;
        } else {
            if (detectionState == DetectionUnreliable)
                detection_bad_voting_count_++;
            if (detectionState == DetectionBad)
                detection_bad_voting_count_ += 2;
        }
    }
}

void ObjectBase::SetPose(
    const FrameIndex &frmIndex, const double &timeStamp,
    const ObjRecogState &state, const Eigen::Matrix3d &Rcw,
    const Eigen::Vector3d &Tcw, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &Two) {
    std::lock_guard<std::mutex> lck(mPoseMutex);
    if (state == TrackingGood || state == TrackingBad ||
        state == TrackingUnreliable) {
        TrackingStateSetPose(state, frmIndex, timeStamp, Rcw, Tcw, Rwo, Two);
    } else {
        DetectionStateSetPose(state, frmIndex, timeStamp, Rcw, Tcw, Rwo, Two);
    }
}

// TODO(xiarui): use the structrure to get pose
void ObjectBase::GetPose(
    FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
    Eigen::Matrix3d &Rcw, Eigen::Vector3d &Tcw, Eigen::Matrix3d &Rwo,
    Eigen::Vector3d &Two) {
    std::lock_guard<std::mutex> lck(mPoseMutex);

    if (tracker_state_.GetState() == TrackingGood ||
        (tracker_state_.GetState() == TrackingUnreliable &&
         detector_state_.GetState() != DetectionGood)) {
        tracker_state_.GetData(Rwo, Rcw, Two, Tcw, state, frmIndex, timeStamp);
    } else if (
        detector_state_.GetState() == DetectionGood ||
        detector_state_.GetState() == DetectionUnreliable) {
        detector_state_.GetData(Rwo, Rcw, Two, Tcw, state, frmIndex, timeStamp);
    } else {
        detector_state_.GetData(Rwo, Rcw, Two, Tcw, state, frmIndex, timeStamp);
        state = DetectionBad;
    }
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
