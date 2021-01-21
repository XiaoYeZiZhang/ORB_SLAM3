#ifndef ORB_SLAM3_POINTCLOUDOBJECT_H
#define ORB_SLAM3_POINTCLOUDOBJECT_H
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include "DetectorFrame.h"
#include "Object.h"
#include "DBow3/src/DBoW3.h"

typedef long unsigned int KeyFrameIndex;
namespace ObjRecognition {
#define OBJ_WITH_KF
#define USE_INLIER
#define INVALID_FRAME_INDEX -1

typedef long unsigned int FrameIndex;
typedef long unsigned int MapPointIndex;
class MapPoint {
public:
    typedef std::shared_ptr<MapPoint> Ptr;
    bool Load(const int &descriptor_len, long long &mem_pos, const char *mem);
    bool Save(int &mem_size, char **mem);

    cv::Mat &GetDescriptor();
    Eigen::Vector3d &GetPose();
    std::vector<std::pair<FrameIndex, int>> &GetObservations();
    void SetPose(const Eigen::Vector3d &pose);
    MapPointIndex &GetIndex();
    MapPointIndex GetID() const {
        return m_Id;
    }

private:
    std::mutex m_pose_mutex;
    MapPointIndex m_Id;
    Eigen::Vector3d m_world_pos;
    std::pair<FrameIndex, cv::Mat> m_desps;
    std::vector<std::pair<FrameIndex, int>> m_observations;
};

class KeyFrame {
public:
    typedef std::shared_ptr<KeyFrame> Ptr;
    int GetID() const;
    void
    ReadFromMemory(int descriptor_len, long long &mem_pos, const char *mem);
    void SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    void GetPose(Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);
    cv::Mat &GetRawImage();
    cv::Mat &GetDesciriptor();
    std::vector<cv::KeyPoint> &GetKeyPoints();
    bool ComputeBowFeatures();
    DBoW3::FeatureVector &GetBowFeatVec();
    DBoW3::BowVector &GetBowVec();

    std::vector<long unsigned int> m_connect_kfs;
    std::vector<long unsigned int> m_connect_mappoints;

private:
    KeyFrameIndex mnId = -1;
    cv::Mat mImage;
    std::vector<cv::KeyPoint> mvKeypoints;
    cv::Mat mDescriptors;

    std::shared_ptr<DBoW3::Vocabulary> m_voc;
    std::vector<cv::Mat> mvDesp;
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;
    std::vector<DBoW3::NodeId> mvNodeIds;
    bool mbBowValid;

    Eigen::Matrix3d m_Rcw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d m_tcw = Eigen::Vector3d::Zero();

    long unsigned int m_connect_kfs_num;
    long unsigned int m_connect_mappoints_num;
};

typedef std::vector<MapPoint::Ptr> PointModel;
class Object : public ObjectBase {
public:
    Object(int id);
    virtual ~Object();
    bool Load(const long long &mem_size, const char *mem) {
        return true;
    };
    bool LoadPointCloud(const long long &mem_size, const char *mem);
    void SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    bool Save(long long &mem_size, char **mem);
    std::vector<MapPoint::Ptr> &GetPointClouds();
    std::vector<KeyFrame::Ptr> &GetKeyFrames();
    void SetDatabase(const std::shared_ptr<DBoW3::Database> &database);
    std::shared_ptr<DBoW3::Database> &GetDatabase();
    size_t GetPointCloudsNum();
    int GetKeyFrameIndexByEntryId(int entry_id);
    std::shared_ptr<KeyFrame> GetKeyFrameByIndex(const int &nKFID);
    std::shared_ptr<KeyFrame> GetMatchFrameFromMap(
        const DBoW3::QueryResults &dbowRet, const int &retIndex);
    std::vector<KeyFrame::Ptr>
    FrameQueryMap(const std::shared_ptr<ObjRecognition::DetectorFrame> &frm);
    void AddKeyFrames2Database(
        const std::vector<ObjRecognition::KeyFrame::Ptr> &kfs);

public:
    std::map<MapPointIndex, MapPoint::Ptr> m_pointclouds_map;
    std::map<FrameIndex, std::shared_ptr<KeyFrame>> m_mp_keyframes;

    void SetAssociatedMapPointsByConnection(
        const std::set<long unsigned int> &associated_mappoints_id);
    void SetAssociatedKeyFrames(const std::set<int> &associated_keyframes_id);
    std::set<long unsigned int> m_associated_mappoints_id;
    std::set<int> m_associated_keyframes_id;

private:
    std::vector<MapPoint::Ptr> m_pointclouds;
    std::vector<KeyFrame::Ptr> m_keyframes;
    std::shared_ptr<DBoW3::Vocabulary> m_voc;
    std::shared_ptr<DBoW3::Database> m_database;
    std::vector<FrameIndex> m_entry_to_keyframe_index;
    std::unordered_map<FrameIndex, int> m_keyframe_index_to_entry;
    int m_descriptor_len;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_POINTCLOUDOBJECT_H
