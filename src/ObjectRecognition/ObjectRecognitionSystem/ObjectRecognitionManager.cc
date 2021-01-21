#include <zlib.h>
#include <iostream>
#include "Utility/Camera.h"
#include "StatisticsResult/Statistics.h"
#include "ObjectRecognitionSystem/ObjectRecognitionManager.h"
#include "mode.h"
namespace ObjRecognitionExd {

ObjRecongManager &ObjRecongManager::Instance() {
    static ObjRecongManager instance;
    return instance;
}

ObjRecongManager::ObjRecongManager() {
    m_voc = std::make_shared<DBoW3::Vocabulary>();
    m_obj_rotation_buffer = new float[9];
    m_obj_translation_buffer = new float[3];
    m_model_bounding_box = new float[24];
    m_obj_state_buffer = new int[1];
    Clear();
}

ObjRecongManager::~ObjRecongManager() {
    delete[] m_obj_rotation_buffer;
    delete[] m_obj_translation_buffer;
    delete[] m_model_bounding_box;
    delete[] m_obj_state_buffer;
}

void ObjRecongManager::Clear() {
    m_objrecog_thread.RequestReset();
    m_objrecog_thread.WaitEndReset();
    m_objrecog_thread.StartThread();
}

int ObjRecongManager::Destroy() {
    std::lock_guard<std::mutex> lck(m_mutex);
    Clear();
    m_objrecog_thread.RequestStop();
    m_objrecog_thread.WaitEndStop();
    return 0;
}

int ObjRecongManager::Reset() {
    std::lock_guard<std::mutex> lck(m_mutex);
    Clear();
    return 0;
}

int ObjRecongManager::CreateWithConfig() {
    std::lock_guard<std::mutex> lck(m_mutex);
    int ret = Init();
    if (ret == 0) {
    } else {
        Clear();
    }
    return 0;
}

bool ObjRecongManager::LoadVoc(std::string &voc_path) {
    if (m_voc.get()) {
        m_voc.get()->load(voc_path);
        return true;
    }
    return false;
}

int ObjRecongManager::LoadModel(
    const int id, const char *buffer, long long buffer_len,
    std::shared_ptr<ObjRecognition::Object> &object) {
    std::lock_guard<std::mutex> lck(m_mutex);
    m_object = std::make_shared<ObjRecognition::Object>(id);
    if (!m_object->LoadPointCloud(buffer_len, buffer)) {
        LOG(ERROR) << "Load PointCloud failed, not set model";
        return -1;
    }
    object = m_object;

#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#else
    m_object->SetVocabulary(m_voc);
#endif
    m_objrecog_thread.SetModel(m_object);
    return 0;
}

int ObjRecongManager::Init() {
    m_objrecog_thread.SetVocabulary(m_voc);
    m_objrecog_thread.Init();
    if (m_objrecog_thread.StartThread()) {
        VLOG(0) << "create objRecognition thread failed";
        return -1;
    }
    return 0;
}

int ObjRecongManager::Run(const ObjRecognition::CallbackFrame &platform_frame) {
    std::lock_guard<std::mutex> lck(m_mutex);
    STATISTICS_UTILITY::StatsCollector pointCloudFrameNum("Image number");
    pointCloudFrameNum.IncrementOne();

    std::shared_ptr<ObjRecognition::CallbackFrame> frame =
        std::make_shared<ObjRecognition::CallbackFrame>();
    frame->id = platform_frame.id;
    std::memcpy(&frame->t, &platform_frame.t, 3 * sizeof(platform_frame.t[0]));
    for (int index = 0; index < 3; index++) {
        std::memcpy(
            &frame->R[index], &platform_frame.R[index],
            3 * sizeof(platform_frame.R[index][0]));
    }
    frame->width = 0;
    frame->height = 0;
    frame->width = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    frame->height = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    frame->data = new unsigned char[frame->height * frame->width];
    std::memcpy(
        frame->data, platform_frame.data,
        sizeof(char) * frame->height * frame->width);

    m_objrecog_thread.PushData(frame);
    return 0;
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

    m_objrecog_thread.GetResult(
        frmIndex, timestamp, state, R_camera, t_camera, Rwo, two);

    Rco = R_camera * Rwo;
    tco = R_camera * two + t_camera;
    Row = Rco.transpose() * (R_camera);
    tow = Rco.transpose() * (t_camera - tco);
    Rwo = Row.transpose();
    two = -Rwo * tow;

    Eigen::Matrix3f R_camera_f = R_camera.cast<float>();
    Eigen::Vector3f t_camera_f = t_camera.cast<float>();
    Eigen::Matrix3f R_obj_f = Rwo.cast<float>();
    Eigen::Vector3f t_obj_f = two.cast<float>();

    ObjRecognition::ObjRecogResult objrecog_result;

    m_obj_state_buffer[0] = 0;
    if (state == ObjRecognition::ObjRecogState::TrackingGood) {
        m_obj_state_buffer[0] = 0;
        objrecog_result.num = 1;
    } else {
        m_obj_state_buffer[0] = -1;
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
        &m_obj_translation_buffer[0], t_obj_f.data(), 3 * sizeof(t_obj_f[0]));
    std::memcpy(
        &m_obj_rotation_buffer[0], Eigen::Vector3f(R_obj_f.col(0)).data(),
        3 * sizeof(t_obj_f[0]));
    std::memcpy(
        &m_obj_rotation_buffer[3], Eigen::Vector3f(R_obj_f.col(1)).data(),
        3 * sizeof(t_obj_f[0]));
    std::memcpy(
        &m_obj_rotation_buffer[6], Eigen::Vector3f(R_obj_f.col(2)).data(),
        3 * sizeof(t_obj_f[0]));

    std::vector<Eigen::Vector3d> bounding_box;
    if (m_object != nullptr) {
        bounding_box = m_object->GetBoundingBox();
        for (size_t index = 0; index < bounding_box.size(); index++) {
            m_model_bounding_box[index * 3] = bounding_box.at(index)[0];
            m_model_bounding_box[index * 3 + 1] = bounding_box.at(index)[1];
            m_model_bounding_box[index * 3 + 2] = bounding_box.at(index)[2];
        }
    }

    if (bounding_box.empty()) {
        objrecog_result.num = 0;
    }

    objrecog_result.state_buffer = m_obj_state_buffer;
    objrecog_result.t_obj_buffer = m_obj_translation_buffer;
    objrecog_result.R_obj_buffer = m_obj_rotation_buffer;
    objrecog_result.bounding_box = m_model_bounding_box;

    std::vector<Eigen::Vector3d> pointCloud_pos;
    for (const auto &pointcloud : m_object->GetPointClouds()) {
        pointCloud_pos.emplace_back(pointcloud->GetPose());
    }

    objrecog_result.pointCloud_pos.clear();
    objrecog_result.pointCloud_pos = pointCloud_pos;
    return objrecog_result;
}
} // namespace ObjRecognitionExd
