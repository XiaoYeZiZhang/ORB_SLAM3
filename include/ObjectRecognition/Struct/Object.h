//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_OBJECT_H
#define ORB_SLAM3_OBJECT_H
#include <Eigen/Core>
#include <mutex>
#include <vector>
typedef long unsigned int FrameIndex;
namespace ObjRecognition {

typedef enum ObjRecognitionState {
    DetectionGood = 0,
    DetectionUnreliable = 1,
    DetectionBad = 2,
    TrackingGood = 3,
    TrackingUnreliable = 4,
    TrackingBad = 5
} ObjRecogState;

class ObjStateStruct {
public:
    ObjStateStruct(
        const Eigen::Matrix3d &Rwo, const Eigen::Matrix3d &Rcw,
        const Eigen::Vector3d &Two, const Eigen::Vector3d &Tcw,
        const ObjRecogState &State, const FrameIndex &FrmIndex,
        const double &TimeStamp)
        : mRwo(Rwo), mRcw(Rcw), mtwo(Two), mtcw(Tcw), mState(State),
          mFrmIndex(FrmIndex), mTimeStamp(TimeStamp) {
    }

    ObjStateStruct() {
        mRwo = Eigen::Matrix3d::Identity();
        mRcw = Eigen::Matrix3d::Identity();
        mtwo = Eigen::Vector3d::Zero();
        mtcw = Eigen::Vector3d::Zero();
        mFrmIndex = -1;
        mTimeStamp = -1;
    }

    ~ObjStateStruct() {
    }

    void GetData(
        Eigen::Matrix3d &Rwo, Eigen::Matrix3d &Rcw, Eigen::Vector3d &two,
        Eigen::Vector3d &tcw, ObjRecogState &State, FrameIndex &frmIndex,
        double &TimeStamp);

    void SetData(
        const Eigen::Matrix3d &Rwo, const Eigen::Matrix3d &Rcw,
        const Eigen::Vector3d &two, const Eigen::Vector3d &tcw,
        const ObjRecogState &State, const FrameIndex &FrmIndex,
        const double &TimeStamp);

    ObjRecogState GetState();

    void Reset(const ObjRecogState &state);

private:
    Eigen::Matrix3d mRwo, mRcw;
    Eigen::Vector3d mtwo, mtcw;
    ObjRecogState mState;
    FrameIndex mFrmIndex;
    double mTimeStamp;

    std::mutex mStateMutex;
};

typedef struct ObjRecogResult {
    int num;
    int frame_index;
    double time_stamp;

    /// each digits corresponds to a object id
    int *id_buffer;
    /// each digits corresponds to a object state
    ///     0: state success
    ///     1: state lost
    ///     2: state stopped
    ///     10: state initializing
    int *state_buffer;
    /// object pose: Rwo two
    /// [0, 8] [9, 17] ... [9*n, 8+9*n]; every 9 digits is the rotation matrix
    /// (col major)
    float *R_obj_buffer;
    /// [0, 2] [3, 5] ... [3*n, 2+3*n]; every 3 digits is the translation vector
    float *t_obj_buffer;

    /// camera pose: Rcw tcw
    float R_camera[3][3];
    float t_camera[3];

    /// bounding box: SLAM world coordinate
    /// [0, 23] [24, 47] ... [24*n, 23+24*n]; every 24 digits is the bounding
    /// box vector
    float *bounding_box;

    int info_length;
    const char *info;

    std::vector<Eigen::Vector3d> pointCloud_pos;
} ObjRecogResult;

class ObjectBase {
public:
    ObjectBase(int id);

    virtual ~ObjectBase() {
    }

    int GetId() {
        return mId;
    }

    virtual bool Load(const long long &mem_size, const char *mem) = 0;
    virtual bool Save(long long &mem_size, char **mem) = 0;

    void TrackingStateSetPose(
        const ObjRecogState &trackerState, const FrameIndex &frmIndex,
        const double &timeStamp, const Eigen::Matrix3d &Rcw,
        const Eigen::Vector3d &tcw, const Eigen::Matrix3d &Rwo,
        const Eigen::Vector3d &two);

    void DetectionStateSetPose(
        const ObjRecogState &detectionState, const FrameIndex &frmIndex,
        const double &timeStamp, const Eigen::Matrix3d &Rcw,
        const Eigen::Vector3d &tcw, const Eigen::Matrix3d &Rwo,
        const Eigen::Vector3d &two);

    void SetPose(
        const FrameIndex &frmIndex, const double &timeStamp,
        const ObjRecogState &state, const Eigen::Matrix3d &Rcw,
        const Eigen::Vector3d &tcw, const Eigen::Matrix3d &Rwo,
        const Eigen::Vector3d &two);

    void GetPose(
        FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
        Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, Eigen::Matrix3d &Rwo,
        Eigen::Vector3d &two);

    std::vector<Eigen::Vector3d> GetBoundingBox();

    void Reset();

protected:
    int mId;
    ObjStateStruct tracker_state_;
    ObjStateStruct detector_state_;
    int tracking_bad_voting_count_;
    int detection_bad_voting_count_;
    std::vector<Eigen::Vector3d> mvBoundingBox;
    Eigen::Vector3d m_box_scale;

    std::mutex mPoseMutex;
    std::mutex mBoundingBoxMutex;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_OBJECT_H
