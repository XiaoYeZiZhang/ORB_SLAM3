//
// Created by zhangye on 2020/9/15.
//
//
#include <zlib.h>
#include <iostream>
#include "Utility/Camera.h"
#include "ObjectRecognitionSystem/ObjectRecognitionManager.h"
namespace ObjRecognitionExd {

ObjRecongManager &ObjRecongManager::Instance() {
    static ObjRecongManager instance;
    return instance;
}

ObjRecongManager::ObjRecongManager() {
    voc_ = std::make_shared<DBoW3::Vocabulary>();
    version_buffer = new char[version_.size() + 1];
    memcpy(version_buffer, version_.data(), version_.size());
    *(version_buffer + version_.size()) = '\0';

    info_buffer_ = new char[info_Buffer_maxsize_];

    obj_num_ = 1;
    obj_rotation_buffer_ = new float[obj_num_ * 9];
    obj_translation_buffer_ = new float[obj_num_ * 3];
    model_bounding_box_ = new float[obj_num_ * 24];
    obj_id_buffer_ = new int[obj_num_];
    obj_state_buffer_ = new int[obj_num_];

    Clear();
}

ObjRecongManager::~ObjRecongManager() {
    delete[] info_buffer_;
    delete[] version_buffer;
    delete[] obj_rotation_buffer_;
    delete[] obj_translation_buffer_;
    delete[] model_bounding_box_;
    delete[] obj_id_buffer_;
    delete[] obj_state_buffer_;
}

void ObjRecongManager::Clear() {
    VLOG(10) << "ObjRecong Manager Clear";
    info_buffer_length_ = 0;

    {
        std::unique_lock<std::mutex> lock(m_mutex_info_buffer);
        memset(info_buffer_, 0, info_Buffer_maxsize_);
    }
    object_map_.clear();
    objrecog_thread_.RequestReset();
    objrecog_thread_.WaitEndReset();
    objrecog_thread_.StartRunning();
}

int ObjRecongManager::Destroy() {
    VLOG(10) << "ObjRecong Manager Destroy";
    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);
    if (!IsInitializedState()) {
        return -1;
    }
    Clear();
    ToUninitializedState();
    objrecog_thread_.RequestStop();
    objrecog_thread_.WaitEndStop();
    return 0;
}

int ObjRecongManager::Reset() {
    VLOG(10) << "ObjRecong Manager Reset";
    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);
    if (!IsInitializedState()) {
        return -1;
    }

    Clear();
    return 0;
}

int ObjRecongManager::CreateWithConfig() {
    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);

    if (IsInitializedState()) {
        VLOG(0) << "ObjRecogManager isn't initialized";
        return -1;
    }

    int ret = Init();
    if (ret == 0) {
        ToInitializedState();
    } else {
        Clear();
    }

    return 0;
}

int read_compressed_voc(
    const char *zip_buffer, uint64_t len, DBoW3::Vocabulary *voc) {
    const char *ptr = zip_buffer;
    uint64_t compressed_len = len;
    uint64_t uncompressed_len;
    memcpy(&uncompressed_len, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);
    compressed_len -= sizeof(uint64_t);

    char *voc_buf = new char[uncompressed_len];

    z_stream infstream;
    infstream.zalloc = Z_NULL;
    infstream.zfree = Z_NULL;
    infstream.opaque = Z_NULL;

    infstream.avail_in = (uInt)compressed_len;
    infstream.next_in = (Bytef *)ptr;
    infstream.avail_out = (uInt)uncompressed_len;
    infstream.next_out = (Bytef *)voc_buf;

    inflateInit(&infstream);
    inflate(&infstream, Z_NO_FLUSH);
    inflateEnd(&infstream);

    int res = voc->LoadFromMemory(voc_buf, uncompressed_len);
    delete[] voc_buf;
    return res;
}

bool ObjRecongManager::LoadORBVoc(std::string &voc_path) {
    if (voc_.get()) {
        voc_.get()->load("/home/zhangye/Develope/ObjectRecognition_ORBSLAM3/"
                         "Vocabulary/ORBvoc.txt");
        return true;
    }
    return false;
}

int ObjRecongManager::LoadDic(char const *buffer, int buffer_len) {
    VLOG(10) << "ObjRecong Manager LoadDic";
    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);

    int res = -1;
    if (!voc_) {
        LOG(FATAL) << "Need to create vocabulary";
        return -1;
    }

    int voc_ok = read_compressed_voc(buffer, buffer_len, voc_.get());
    // VLOG(10) << "after load voc, cost " << timer.Stop();
    if (voc_ok == -1) {
        VLOG(0) << "BackEndSystem: voc parse failed.";
    } else if (voc_ok == -2) {
        VLOG(0) << "BackEndSystem: voc empty.";
    } else if (voc_ok == 0 && voc_.get() == nullptr) {
        VLOG(0) << "BackEndSystem: create Vocabulary failed.";
        res = -1;
    } else if (voc_ok == 0 && voc_.get() != nullptr) {
        VLOG(0) << "BackEndSystem: local load voc done.";
        res = 0;
    } else {
        VLOG(0) << "BackEndSystem: load voc unknow error.";
    }
    res = voc_ok;
    VLOG(0) << "Load dic success";

    return res;
}

int ObjRecongManager::LoadModel(
    const int id, const char *buffer, int buffer_len) {
    VLOG(10) << "ObjRecong Manager LoadModel";
    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);
    if (IsUninitializedState()) {
        return -1;
    }

    if (object_map_.find(id) != object_map_.end()) {
        LOG(ERROR) << "object id " << id << " already exist";
        return -1;
    }

    object_ = std::make_shared<ObjRecognition::Object>(id);
    object_map_.insert(
        std::pair<int, std::shared_ptr<ObjRecognition::Object>>(id, object_));
    if (!object_->LoadPointCloud(buffer_len, buffer)) {
        LOG(ERROR) << "Load PointCloud failed, not set model";
        return -1;
    }

    object_->SetVocabulary(voc_);
    objrecog_thread_.SetModel(object_);

    return 0;
}

int ObjRecongManager::Init() {
    objrecog_thread_.SetVocabulary(voc_);
    objrecog_thread_.Init();
    objrecog_thread_.StartThread("ObjRecongManager", 0x10);
    return 0;
}

int ObjRecongManager::Run(
    const ObjRecognition::ObjRecogFrameCallbackData &platform_frame) {
    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);

    int ret = -1;
    if (object_map_.size() <= 0) {
        VLOG(10) << "Objects' num is zero";
        return ret;
    }

    // Common::StatsCollector pointCloudFrameNum("Frame num");
    // pointCloudFrameNum.IncrementOne();

    VLOG(10) << "ObjRecogManager Run";

    std::shared_ptr<ObjRecognition::ObjRecogFrameCallbackData> frame =
        std::make_shared<ObjRecognition::ObjRecogFrameCallbackData>();

    frame->timestamp = platform_frame.timestamp;
    frame->id = platform_frame.id;
    frame->has_image = platform_frame.has_image;
    frame->feature_mem_size = platform_frame.feature_mem_size;
    std::memcpy(&frame->t, &platform_frame.t, 3 * sizeof(platform_frame.t[0]));
    for (int index = 0; index < 3; index++) {
        std::memcpy(
            &frame->R[index], &platform_frame.R[index],
            3 * sizeof(platform_frame.R[index][0]));
    }

    frame->img.width = 0;
    frame->img.height = 0;
    if (frame->has_image) {
        frame->img.width =
            ObjRecognition::CameraIntrinsic::GetInstance().Width();
        frame->img.height =
            ObjRecognition::CameraIntrinsic::GetInstance().Height();
        frame->img.data =
            new unsigned char[frame->img.height * frame->img.width];
        std::memcpy(
            frame->img.data, platform_frame.img.data,
            sizeof(char) * frame->img.height * frame->img.width);
    }

    objrecog_thread_.PushUnProcessedFrame(frame);
    ret = 0;

    return ret;
}

ObjRecognition::ObjRecogResult ObjRecongManager::GetObjRecognitionResult() {
    double timestamp = 0;
    ObjRecognition::FrameIndex frmIndex = -1;
    ObjRecognition::ObjRecogState state =
        ObjRecognition::ObjRecogState::TrackingBad;
    Eigen::Matrix3d R_camera = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_camera = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_obj = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_obj = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Row = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tow = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rwo = Eigen::Matrix3d::Identity();
    Eigen::Vector3d two = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rco = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tco = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rslam2gl = Eigen::Matrix3d::Zero();

    if (IsInitializedState()) {
        objrecog_thread_.GetResult(
            frmIndex, timestamp, state, R_camera, t_camera, Rwo, two);
    } else {
        VLOG(0) << "ObjRecogManager isn't initialized";
    }

    Rco = R_camera * Rwo;
    tco = R_camera * two + t_camera;
    // TODO(zhangye): check the coords
    Rslam2gl(0, 0) = 1;
    Rslam2gl(1, 2) = -1;
    Rslam2gl(2, 1) = 1;
    // Rco = Rco * Rslam2gl;

    Row = Rco.transpose() * (R_camera); // world -> obj
    tow = Rco.transpose() * (t_camera - tco);
    Rwo = Row.transpose();
    two = -Rwo * tow;

    Eigen::Matrix3f R_camera_f = R_camera.cast<float>();
    Eigen::Vector3f t_camera_f = t_camera.cast<float>();
    Eigen::Matrix3f R_obj_f = Rwo.cast<float>();
    Eigen::Vector3f t_obj_f = two.cast<float>();

    ObjRecognition::ObjRecogResult objrecog_result;
    objrecog_result.frame_index = frmIndex;
    objrecog_result.time_stamp = timestamp;

    obj_id_buffer_[0] = 0;
    obj_state_buffer_[0] = 0;
    if (state == ObjRecognition::ObjRecogState::TrackingGood) {
        obj_state_buffer_[0] = 0;
        objrecog_result.num = 1;
    } else {
        obj_state_buffer_[0] = -1;
        objrecog_result.num = 0;
    }

    std::memcpy(
        &objrecog_result.t_camera, t_camera_f.data(),
        3 * sizeof(t_camera_f[0]));
    std::memcpy(
        &objrecog_result.R_camera[0], Eigen::Vector3f(R_camera_f.row(0)).data(),
        3 * sizeof(t_camera_f[0]));
    std::memcpy(
        &objrecog_result.R_camera[1], Eigen::Vector3f(R_camera_f.row(1)).data(),
        3 * sizeof(t_camera_f[0]));
    std::memcpy(
        &objrecog_result.R_camera[2], Eigen::Vector3f(R_camera_f.row(2)).data(),
        3 * sizeof(t_camera_f[0]));

    std::memcpy(
        &obj_translation_buffer_[0], t_obj_f.data(), 3 * sizeof(t_obj_f[0]));
    std::memcpy(
        &obj_rotation_buffer_[0], Eigen::Vector3f(R_obj_f.col(0)).data(),
        3 * sizeof(t_obj_f[0]));
    std::memcpy(
        &obj_rotation_buffer_[3], Eigen::Vector3f(R_obj_f.col(1)).data(),
        3 * sizeof(t_obj_f[0]));
    std::memcpy(
        &obj_rotation_buffer_[6], Eigen::Vector3f(R_obj_f.col(2)).data(),
        3 * sizeof(t_obj_f[0]));

    std::vector<Eigen::Vector3d> bounding_box;
    if (object_ != nullptr) {
        bounding_box = object_->GetBoundingBox();
        for (size_t index = 0; index < bounding_box.size(); index++) {
            bounding_box.at(index) = Rwo * bounding_box.at(index) + two;

            model_bounding_box_[index * 3] = bounding_box.at(index)[0];
            model_bounding_box_[index * 3 + 1] = bounding_box.at(index)[1];
            model_bounding_box_[index * 3 + 2] = bounding_box.at(index)[2];
        }
    }

    if (bounding_box.empty()) {
        objrecog_result.num = 0;
    }

    VLOG(5) << "objrecog result num: " << objrecog_result.num;

    SetObjRecongInfo();

    objrecog_result.id_buffer = obj_id_buffer_;
    objrecog_result.state_buffer = obj_state_buffer_;
    objrecog_result.t_obj_buffer = obj_translation_buffer_;
    objrecog_result.R_obj_buffer = obj_rotation_buffer_;
    objrecog_result.bounding_box = model_bounding_box_;

    objrecog_result.info_length = info_buffer_length_;
    objrecog_result.info = info_buffer_;

    std::vector<Eigen::Vector3d> pointCloud_pos;
    for (auto pointcloud : object_->GetPointClouds()) {
        pointCloud_pos.emplace_back(pointcloud->GetPose());
    }

    objrecog_result.pointCloud_pos.clear();
    objrecog_result.pointCloud_pos = pointCloud_pos;
    return objrecog_result;
}

int ObjRecongManager::SetObjRecongInfo() {
    int ret = -1;
    info_buffer_length_ = 0;
    if (!IsInitializedState()) {
        return ret;
    }

    std::string info;
    objrecog_thread_.GetInfo(info);
    info = version_ + '\n' + info;

    {
        std::unique_lock<std::mutex> lock(m_mutex_info_buffer);
        std::memset(info_buffer_, 0, info_Buffer_maxsize_);
        std::memcpy(info_buffer_, info.data(), info.size());
    }
    info_buffer_length_ = info.size();

    ret = 0;

    return ret;
}

char *ObjRecongManager::GetVersion() {
    //    std::lock_guard<std::mutex> lck(mMutexForPublicAPI);
    return version_buffer;
}
} // namespace ObjRecognitionExd
