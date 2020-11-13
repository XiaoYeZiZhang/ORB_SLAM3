//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_POINTCLOUDOBJECT_H
#define ORB_SLAM3_POINTCLOUDOBJECT_H
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include "Detector/DetectorFrame.h"
#include "Object.h"
#include "DBow3/src/DBoW3.h"

typedef long unsigned int KeyFrameIndex;
namespace ObjRecognition {
#define OBJ_WITH_KF
#define USE_INLIER
//#define USE_REPROJ
#define INVALID_FRAME_INDEX -1

enum ObjModelVersion { NormalModel = 0, ModelWithImage = 1 };
typedef long unsigned int FrameIndex;
typedef long unsigned int MapPointIndex;
class MapPoint {
public:
    typedef std::shared_ptr<MapPoint> Ptr;
    bool Load(long long &mem_pos, const char *mem);
    bool Save(int &mem_size, char **mem);
    bool IsBad();

    cv::Mat &GetDescriptor();
    Eigen::Vector3d &GetPose();
    std::vector<std::pair<FrameIndex, cv::Mat>> &GetMultiDescriptor();
    std::vector<std::pair<FrameIndex, int>> &GetObservations();
    void SetPose(const Eigen::Vector3d &pose);
    MapPointIndex &GetIndex();
    static int GetMemSize() {
        static int perSize = sizeof(MapPointIndex) + sizeof(double) * 3 +
                             sizeof(char) * 32 + sizeof(int) * 2;
        return perSize;
    }
    const std::string GetInfo();
    MapPointIndex GetID() {
        return mnId;
    }

private:
    std::mutex mMutexPos;

    MapPointIndex mnId;
    Eigen::Vector3d mWorldPos;
    std::vector<std::pair<FrameIndex, cv::Mat>> m_multi_desps;

    std::vector<std::pair<FrameIndex, int>> m_observations;

    int mnVisible;
    int mnFound;
};

class KeyFrame {
public:
    typedef std::shared_ptr<KeyFrame> Ptr;

    int GetID();
    void ReadFromMemory(long long &mem_pos, const char *mem);
    void SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    void GetPose(Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);
    cv::Mat &GetRawImage();

    cv::Mat &GetDesciriptor();
    std::vector<cv::KeyPoint> &GetKeyPoints();
    bool ComputeBowFeatures();

    DBoW3::FeatureVector &GetBowFeatVec();
    DBoW3::BowVector &GetBowVec();

    std::vector<long unsigned int> connect_kfs;
    std::vector<long unsigned int> connect_mappoints;

private:
    int mnVersion = 0;
    KeyFrameIndex mnId = -1;

    cv::Mat mImage;
    std::vector<cv::KeyPoint> mvKeypoints;
    cv::Mat mDescriptors;

    /// bow feature
    std::shared_ptr<DBoW3::Vocabulary> voc_;
    std::vector<cv::Mat> mvDesp;
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;
    std::vector<DBoW3::NodeId> mvNodeIds;
    bool mbBowValid;

    Eigen::Matrix3d mRcw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d mtcw = Eigen::Vector3d::Zero();

    long unsigned int connect_kfs_num;
    long unsigned int connect_mappoints_num;

    int mImgWidth = 0;
    int mImgHeight = 0;
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
    bool LoadMesh(const int &mem_size, const char *mem);
    void SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    bool Save(long long &mem_size, char **mem);
    std::vector<MapPoint::Ptr> &GetPointClouds();
    std::vector<KeyFrame::Ptr> &GetKeyFrames();
    void SetDatabase(const std::shared_ptr<DBoW3::Database> &database);
    std::shared_ptr<DBoW3::Database> &GetDatabase();
    size_t GetPointCloudsNum();
    size_t GetKeyFramesNum();
    int GetKeyFrameIndexByEntryId(int entry_id);
    std::shared_ptr<KeyFrame> GetKeyFrameByIndex(const int &nKFID);
    std::shared_ptr<KeyFrame> GetMatchFrameFromMap(
        const DBoW3::QueryResults &dbowRet, const int &retIndex);
    std::vector<KeyFrame::Ptr>
    FrameQueryMap(const std::shared_ptr<ObjRecognition::DetectorFrame> &frm);
    void AddKeyFrames2Database(
        const std::vector<ObjRecognition::KeyFrame::Ptr> &kfs);
    std::shared_ptr<DBoW3::Vocabulary> &GetVocabulary();
    std::map<MapPointIndex, MapPoint::Ptr> m_pointclouds_map;
    std::map<FrameIndex, std::shared_ptr<KeyFrame>> m_mp_keyframes;

private:
    std::string m_version = "V0.0.0.0";
    double m_timestamp;
    std::vector<MapPoint::Ptr> m_pointclouds;

    std::vector<KeyFrame::Ptr> m_keyframes;
    std::shared_ptr<DBoW3::Vocabulary> m_voc;
    std::shared_ptr<DBoW3::Database> m_database;

    std::vector<FrameIndex> m_entry_to_keyframe_index;
    std::unordered_map<FrameIndex, int> m_keyframe_index_to_entry;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_POINTCLOUDOBJECT_H
