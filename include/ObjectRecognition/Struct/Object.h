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
    ObjStateStruct() {
        m_Rwo = Eigen::Matrix3d::Identity();
        m_Rcw = Eigen::Matrix3d::Identity();
        m_two = Eigen::Vector3d::Zero();
        m_tcw = Eigen::Vector3d::Zero();
        m_frmIndex = -1;
    }

    ~ObjStateStruct() {
    }

    void GetData(
        Eigen::Matrix3d &Rwo, Eigen::Matrix3d &Rcw, Eigen::Vector3d &two,
        Eigen::Vector3d &tcw, ObjRecogState &State, FrameIndex &frmIndex);

    void SetData(
        const Eigen::Matrix3d &Rwo, const Eigen::Matrix3d &Rcw,
        const Eigen::Vector3d &two, const Eigen::Vector3d &tcw,
        const ObjRecogState &State, const FrameIndex &FrmIndex);

    ObjRecogState GetState();

    void Reset(const ObjRecogState &state);

private:
    Eigen::Matrix3d m_Rwo, m_Rcw;
    Eigen::Vector3d m_two, m_tcw;
    ObjRecogState m_state;
    FrameIndex m_frmIndex;

    std::mutex m_stateMutex;
};

typedef struct ObjRecogResult {
    int num;
    int *state_buffer;
    float *R_obj_buffer;
    float *t_obj_buffer;
    float R_camera[3][3];
    float t_camera[3];
    float *bounding_box;
    std::vector<Eigen::Vector3d> pointCloud_pos;
} ObjRecogResult;

class ObjectBase {
public:
    ObjectBase(int id);
    virtual ~ObjectBase() {
    }
    int GetId() {
        return m_Id;
    }
    virtual bool Load(const long long &mem_size, const char *mem) = 0;
    virtual bool Save(long long &mem_size, char **mem) = 0;

    void TrackingStateSetPose(
        const ObjRecogState &trackerState, const FrameIndex &frmIndex,
        const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
        const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two);

    void DetectionStateSetPose(
        const ObjRecogState &detectionState, const FrameIndex &frmIndex,
        const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
        const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two);

    void SetPose(
        const FrameIndex &frmIndex, const ObjRecogState &state,
        const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw,
        const Eigen::Matrix3d &Rwo, const Eigen::Vector3d &two);

    void SetScale(const double &scale);
    double GetScale();

    void SetPoseForFindSimilarKeyframe(
        const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);

    void
    GetPoseForFindSimilarKeyframe(Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);

    void GetPose(
        FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
        Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw, Eigen::Matrix3d &Rwo,
        Eigen::Vector3d &two);

    std::vector<Eigen::Vector3d> GetBoundingBox();

    void Reset();

protected:
    int m_Id;
    ObjStateStruct m_tracker_state;
    ObjStateStruct m_detector_state;
    int m_tracking_bad_voting_count;
    int m_detection_bad_voting_count;
    std::vector<Eigen::Vector3d> m_boundingbox;
    Eigen::Vector3d m_box_scale;

    std::mutex m_pose_mutex;
    std::mutex m_boundingbox_mutex;

    Eigen::Matrix3d m_Rcw_for_similar_keyframe;
    Eigen::Vector3d m_tcw_for_similar_keyframe;

    double m_scale;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_OBJECT_H
