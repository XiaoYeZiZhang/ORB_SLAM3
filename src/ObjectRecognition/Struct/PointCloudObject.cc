//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include "ORBSLAM3/Converter.h"
#include "Utility/Camera.h"
#include "Struct/PointCloudObject.h"
namespace ObjRecognition {

template <class T1, class T2>
void GetDataFromMem(
    T1 *dst_mem, T2 *src_mem, const int mem_size, unsigned int &pos) {
    memcpy(dst_mem, src_mem, mem_size);
    pos += mem_size;
}

bool MapPoint::Load(unsigned int &mem_pos, const char *mem) {
    GetDataFromMem(&mnId, mem + mem_pos, sizeof(mnId), mem_pos);
    Eigen::Vector3d posTmp;

    // TODO(zhangye): the mappoint position, OPENGL Coords??? Need to change???
    // (Tco) SLAM Coords
    GetDataFromMem(&posTmp(0), ((mem) + mem_pos), sizeof(double), mem_pos);
    GetDataFromMem(&posTmp(1), ((mem) + mem_pos), sizeof(double), mem_pos);
    GetDataFromMem(&posTmp(2), ((mem) + mem_pos), sizeof(double), mem_pos);

    /// pointcloud model is OpenGL coordiante, turn them to the
    /// SLAM coordinate
    /*mWorldPos(0) = posTmp(0);
    mWorldPos(1) = -posTmp(2);
    mWorldPos(2) = posTmp(1);*/

    mWorldPos(0) = posTmp(0);
    mWorldPos(1) = posTmp(1);
    mWorldPos(2) = posTmp(2);
    int desps_size;
    GetDataFromMem(&desps_size, ((mem) + mem_pos), sizeof(int), mem_pos);

    for (int i = 0; i < desps_size; i++) {
        FrameIndex index;
        GetDataFromMem(&index, ((mem) + mem_pos), sizeof(FrameIndex), mem_pos);
        cv::Mat desp = cv::Mat::zeros(32, 1, CV_8U);
        GetDataFromMem(
            desp.data, ((mem) + mem_pos), 32 * sizeof(uchar), mem_pos);
        m_multi_desps.push_back({index, desp});
    }

    unsigned int ref_kfs_id_size;
    GetDataFromMem(
        &ref_kfs_id_size, ((mem) + mem_pos), sizeof(unsigned int), mem_pos);

    for (int i = 0; i < ref_kfs_id_size; i++) {
        FrameIndex kf_id;
        GetDataFromMem(&kf_id, ((mem) + mem_pos), sizeof(FrameIndex), mem_pos);
        m_reference_kf_ids.emplace_back(kf_id);
    }

    GetDataFromMem(&mnVisible, ((mem) + mem_pos), sizeof(mnVisible), mem_pos);
    GetDataFromMem(&mnFound, ((mem) + mem_pos), sizeof(mnFound), mem_pos);
    VLOG(0) << "MapPoint Data: " << mnId << ", " << m_multi_desps.size() << ", "
            << m_reference_kf_ids.size() << ", " << mnVisible << ", "
            << mnFound;
    return true;
}

bool MapPoint::Save(int &mem_size, char **mem) {
    VLOG(20) << "MapPoint::Save";
    return true;
}

bool MapPoint::IsBad() {
    VLOG(30) << "MapPoint::IsBad";
    return false;
}

cv::Mat &MapPoint::GetDescriptor() {
    return m_multi_desps.at(0).second;
}

std::vector<std::pair<FrameIndex, cv::Mat>> &MapPoint::GetMultiDescriptor() {
    return m_multi_desps;
}

std::vector<FrameIndex> &MapPoint::GetRefKeyFrameID() {
    return m_reference_kf_ids;
}

Eigen::Vector3d &MapPoint::GetPose() {
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos;
}

void MapPoint::SetPose(const Eigen::Vector3d &pose) {
    std::unique_lock<std::mutex> lock(mMutexPos);
    mWorldPos = pose;
}

MapPointIndex &MapPoint::GetIndex() {
    return mnId;
}

const std::string MapPoint::GetInfo() {
    std::string info;
    info += "MapPoint id: " + std::to_string(mnId);
    info += " Observation num: " + std::to_string(m_reference_kf_ids.size());
    return info;
}

template <class T1, class T2>
void PutDataToMem(
    T1 *dst_mem, T2 *src_mem, const unsigned int mem_size, unsigned int &pos) {
    memcpy(dst_mem, src_mem, mem_size);
    pos += mem_size;
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat>
UnpackORBFeatures(unsigned int &mem_cur, const char *mem) {
    unsigned int nKpts = 0;
    PutDataToMem(&(nKpts), mem + mem_cur, sizeof(nKpts), mem_cur);
    if (nKpts == 0) {
        return std::forward_as_tuple(std::vector<cv::KeyPoint>(), cv::Mat());
    }
    std::vector<cv::KeyPoint> vKpts(nKpts, cv::KeyPoint(0, 0, 0));
    for (auto &kpt : vKpts) {
        PutDataToMem(&(kpt.pt.x), mem + mem_cur, sizeof(kpt.pt.x), mem_cur);
        PutDataToMem(&(kpt.pt.y), mem + mem_cur, sizeof(kpt.pt.y), mem_cur);
        PutDataToMem(&(kpt.size), mem + mem_cur, sizeof(kpt.size), mem_cur);
        PutDataToMem(&(kpt.angle), mem + mem_cur, sizeof(kpt.angle), mem_cur);
        PutDataToMem(
            &(kpt.response), mem + mem_cur, sizeof(kpt.response), mem_cur);
        PutDataToMem(&(kpt.octave), mem + mem_cur, sizeof(kpt.octave), mem_cur);
        PutDataToMem(
            &(kpt.class_id), mem + mem_cur, sizeof(kpt.class_id), mem_cur);
    }

    cv::Mat temp_desp(nKpts, 32, CV_8UC1, (void *)(mem + mem_cur));
    cv::Mat desp = temp_desp.clone();
    mem_cur += desp.rows * desp.cols * sizeof(uchar);
    return std::forward_as_tuple(std::move(vKpts), std::move(desp));
}

void UnPackCamCWFromMem(
    unsigned int &mem_pos, const char *mem, Eigen::Vector3d &Tcw,
    Eigen::Matrix3d &Rcw) {
    PutDataToMem(&Tcw(0), mem + mem_pos, sizeof(double), mem_pos);
    PutDataToMem(&Tcw(1), mem + mem_pos, sizeof(double), mem_pos);
    PutDataToMem(&Tcw(2), mem + mem_pos, sizeof(double), mem_pos);

    Eigen::Quaterniond QR;
    PutDataToMem(&(QR.w()), mem + mem_pos, sizeof(double), mem_pos);
    PutDataToMem(&(QR.x()), mem + mem_pos, sizeof(double), mem_pos);
    PutDataToMem(&(QR.y()), mem + mem_pos, sizeof(double), mem_pos);
    PutDataToMem(&(QR.z()), mem + mem_pos, sizeof(double), mem_pos);
    Rcw = QR;
}

void KeyFrame::ReadFromMemory(unsigned int &mem_pos, const char *mem) {
    PutDataToMem(&mnId, mem + mem_pos, sizeof(mnId), mem_pos);
    std::tie(mvKeypoints, mDescriptors) = UnpackORBFeatures(mem_pos, mem);
    // TODO(zhangye): check keyframe pose, Tcw under OPENGL?? need to change to
    // SLAM COords?
    UnPackCamCWFromMem(mem_pos, mem, mtcw, mRcw);

    // KF pose in the model file is under the OpenGL coordinate,
    // convert the coordinate to the SLAM coordinate
    /*Eigen::Matrix3d Rgl2slam = Eigen::Matrix3d::Zero();
    Rgl2slam(0, 0) = 1;
    Rgl2slam(1, 2) = -1;
    Rgl2slam(2, 1) = 1;
    mRcw = mRcw * Rgl2slam.transpose();*/

    {
        /// upload image ???
        int imgWidth = ObjRecognition::CameraIntrinsic::GetInstance().Width();
        int imgHeight = ObjRecognition::CameraIntrinsic::GetInstance().Height();
        cv::Mat tempImg(imgHeight, imgWidth, CV_8UC1, (void *)(mem + mem_pos));
        mem_pos += sizeof(char) * imgHeight * imgWidth;
        mImage = tempImg.clone();
    }
}

void KeyFrame::SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc) {
    CHECK_NOTNULL(voc.get());
    voc_ = voc;
}

cv::Mat &KeyFrame::GetRawImage() {
    return mImage;
}

void KeyFrame::GetPose(Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw) {
    Rcw = mRcw;
    tcw = mtcw;
}

cv::Mat &KeyFrame::GetDesciriptor() {
    return mDescriptors;
}

std::vector<cv::KeyPoint> &KeyFrame::GetKeyPoints() {
    return mvKeypoints;
}

bool KeyFrame::ComputeBowFeatures() {
    //  std::lock_guard<std::mutex> lck(mMutexBowFeature);
    if (!mbBowValid && !mDescriptors.empty()) {
        for (int i = 0; i < mDescriptors.rows; ++i) {
            mvDesp.push_back(mDescriptors.row(i));
        }
        voc_->transform(mvDesp, mBowVec, mFeatVec, mvNodeIds, 4);
        VLOG(5) << "KeyFrame::ComputeBowFeatures: mvDesp.size = "
                << mvDesp.size() << " , NodeIds.size = " << mvNodeIds.size();
        mbBowValid = true;
    }
    return !mDescriptors.empty();
}

DBoW3::FeatureVector &KeyFrame::GetBowFeatVec() {
    return mFeatVec;
}

DBoW3::BowVector &KeyFrame::GetBowVec() {
    return mBowVec;
}

int KeyFrame::GetID() {
    return mnId;
}

Object::Object(int id) : ObjectBase(id) {
    VLOG(30) << "PointCloudObject: create";
}

Object::~Object() {
    VLOG(30) << "~PointCloudObject";
}

bool Object::LoadPointCloud(const int &mem_size, const char *mem) {
    int mapPointNum;
    unsigned int mem_pos = 0;
    VLOG(3) << "model mem size: " << mem_size;

    char version_str[sizeof(m_version)];
    GetDataFromMem(version_str, mem + mem_pos, m_version.size(), mem_pos);
    GetDataFromMem(&m_timestamp, mem + mem_pos, sizeof(m_timestamp), mem_pos);
    m_version = std::string(version_str, m_version.size());

    int kf_img_width, kf_img_height;
    GetDataFromMem(&kf_img_width, mem + mem_pos, sizeof(kf_img_width), mem_pos);
    GetDataFromMem(
        &kf_img_height, mem + mem_pos, sizeof(kf_img_height), mem_pos);

    double fx, fy, cx, cy;
    GetDataFromMem(&fx, mem + mem_pos, sizeof(fx), mem_pos);
    GetDataFromMem(&fy, mem + mem_pos, sizeof(fy), mem_pos);
    GetDataFromMem(&cx, mem + mem_pos, sizeof(cx), mem_pos);
    GetDataFromMem(&cy, mem + mem_pos, sizeof(cy), mem_pos);

    // TODO(zhangye): how to set the right boundingbox when scanning
    double bounding_box[24];
    GetDataFromMem(bounding_box, mem + mem_pos, 24 * sizeof(double), mem_pos);
    for (size_t index = 0; index < 8; index++) {
        mvBoundingBox.push_back(Eigen::Vector3d(
            bounding_box[index * 3], bounding_box[index * 3 + 1],
            bounding_box[index * 3 + 2]));
    }

    GetDataFromMem(
        m_box_scale.data(), mem + mem_pos, 3 * sizeof(double), mem_pos);

    GetDataFromMem(&mapPointNum, mem + mem_pos, sizeof(mapPointNum), mem_pos);
    VLOG(3) << "MapPoint num: " << mapPointNum;
    m_pointclouds.reserve(mapPointNum);

    for (int i = 0; i < mapPointNum; i++) {

        std::shared_ptr<MapPoint> mapPoint = std::make_shared<MapPoint>();
        if (mapPoint->Load(mem_pos, mem)) {
            m_pointclouds.push_back(mapPoint);
        } else {
            VLOG(0) << "PointCloudObject::Load fail!";
            return false;
        }
    }

    int keyFrameNum = 0;
    GetDataFromMem(&keyFrameNum, mem + mem_pos, sizeof(keyFrameNum), mem_pos);
    VLOG(3) << "KeyFrame num: " << keyFrameNum;
    m_keyframes.reserve(keyFrameNum);
    for (int i = 0; i < keyFrameNum; i++) {
        std::shared_ptr<KeyFrame> pKF = std::make_shared<KeyFrame>();
        pKF->ReadFromMemory(mem_pos, mem);
        m_keyframes.push_back(pKF);
    }

    if (mem_pos == mem_size) {
        return true;
    } else {
        LOG(ERROR) << "wrong pointCloud file format!";
        return false;
    }
}

void Object::AddKeyFrames2Database(
    const std::vector<ObjRecognition::KeyFrame::Ptr> &kfs) {

    for (const auto &itKF : kfs) {
        itKF->SetVocabulary(m_voc);
        itKF->ComputeBowFeatures();
        auto bowVec = std::move(itKF->GetBowVec());
        auto vFeatVec = std::move(itKF->GetBowFeatVec());
        m_database->add(bowVec, vFeatVec);
        ObjRecognition::FrameIndex kf_id = itKF->GetID();
        m_mp_keyframes.emplace(std::make_pair(kf_id, itKF));
        m_keyframe_index_to_entry.insert(
            std::make_pair(kf_id, m_entry_to_keyframe_index.size()));
        m_entry_to_keyframe_index.emplace_back(kf_id);

        VLOG(10) << "KF id: " << kf_id << " BoW vec size: " << bowVec.size();
    }

    // timer.Stop();
}

int Object::GetKeyFrameIndexByEntryId(int entry_id) {
    //    unique_lock<std::recursive_mutex> lck(mMutexMap);
    if (entry_id >= m_entry_to_keyframe_index.size())
        return INVALID_FRAME_INDEX;
    return m_entry_to_keyframe_index[entry_id];
}

std::shared_ptr<KeyFrame> Object::GetKeyFrameByIndex(const int &nKFID) {
    //    lock_guard<std::recursive_mutex> lck(mMutexMap);
    if (m_mp_keyframes.count(nKFID) > 0) {
        return m_mp_keyframes[nKFID];
    }
    return std::shared_ptr<KeyFrame>();
}

std::shared_ptr<KeyFrame> Object::GetMatchFrameFromMap(
    const DBoW3::QueryResults &dbowRet, const int &retIndex) {

    std::shared_ptr<KeyFrame> pMostMatchFrame;
    if (dbowRet.size() <= retIndex) {
        VLOG(10) << "2DMatch DBoW query results, ret.size = " << dbowRet.size()
                 << ", index = " << retIndex;
    } else {
        int entry_id = dbowRet[retIndex].Id;
        ObjRecognition::FrameIndex KFId = GetKeyFrameIndexByEntryId(entry_id);
        pMostMatchFrame = GetKeyFrameByIndex(KFId);
        if (pMostMatchFrame) {
            VLOG(10) << "2DMatch get match KF from the map, succeed to get id: "
                     << KFId;
        } else {
            VLOG(10) << "2DMatch get match KF from the map, fail to get id: "
                     << KFId;
        }
    }

    return pMostMatchFrame;
}

std::vector<KeyFrame::Ptr> Object::FrameQueryMap(
    const std::shared_ptr<ObjRecognition::DetectorFrame> &frm) {

    // STSLAMCommon::Timer timer("Frame query Map");
    std::vector<cv::Mat> desp;
    DBoW3::BowVector bow_vec;
    DBoW3::FeatureVector feat_vec;
    std::vector<DBoW3::NodeId> node_Ids;
    DBoW3::QueryResults dbow_ret;

    for (int i = 0; i < frm->m_desp.rows; ++i) {
        desp.push_back(frm->m_desp.row(i));
    }
    m_voc->transform(desp, bow_vec, feat_vec, node_Ids, 4);

    std::shared_ptr<DBoW3::Database> database = m_database;
    DBoW3::EntryId db_size = database->size();
    int max_id = static_cast<int>(db_size) - 0;
    database->query(bow_vec, dbow_ret, max_id);

    std::vector<std::shared_ptr<KeyFrame>> kf_matcheds;
    for (int index = 0; index < 2; index++) {

        std::shared_ptr<KeyFrame> kf_matched =
            GetMatchFrameFromMap(dbow_ret, index);
        if (kf_matched && kf_matched->GetKeyPoints().size() > 0) {
            kf_matcheds.push_back(kf_matched);
        }
    }

    // timer.Stop();

    return kf_matcheds;
}

void Object::SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc) {
    CHECK_NOTNULL(voc.get());
    m_voc = voc;
}

std::shared_ptr<DBoW3::Vocabulary> &Object::GetVocabulary() {
    return m_voc;
}

bool Object::Save(int &mem_size, char **mem) {
    VLOG(3) << "PointCloudObject::Save";
    return true;
}

std::vector<MapPoint::Ptr> &Object::GetPointClouds() {
    return m_pointclouds;
}

std::vector<KeyFrame::Ptr> &Object::GetKeyFrames() {
    return m_keyframes;
}

void Object::SetDatabase(const std::shared_ptr<DBoW3::Database> &database) {
    m_database = database;
}

std::shared_ptr<DBoW3::Database> &Object::GetDatabase() {
    return m_database;
}

size_t Object::GetPointCloudsNum() {
    return m_pointclouds.size();
}

size_t Object::GetKeyFramesNum() {
    return m_keyframes.size();
}

} // namespace ObjRecognition
