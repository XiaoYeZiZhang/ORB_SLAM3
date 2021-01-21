#ifndef ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
#define ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
#include <memory>
#include <mutex>
#include "ObjectRecognitionSystem.h"
#include "Camera.h"
namespace ObjRecognitionExd {

class ObjRecongManager {
public:
    static ObjRecongManager &Instance();
    ~ObjRecongManager();
    int Destroy();
    int Reset();
    int Run(const ObjRecognition::CallbackFrame &platform_frame);
    int CreateWithConfig();
    bool LoadVoc(std::string &voc_path);
    int LoadModel(
        int id, const char *buffer, long long buffer_len,
        std::shared_ptr<ObjRecognition::Object> &object);
    ObjRecognition::ObjRecogResult GetObjRecognitionResult();

private:
    ObjRecongManager();
    void Clear();
    int Init();

private:
    std::mutex m_mutex;
    std::shared_ptr<ObjRecognition::Object> m_object;
    std::shared_ptr<DBoW3::Vocabulary> m_voc;
    ObjRecognition::ObjRecogThread m_objrecog_thread;

    int *m_obj_state_buffer;
    float *m_obj_rotation_buffer;
    float *m_obj_translation_buffer;
    float *m_model_bounding_box;
};

static void ObjRecogCallback(ObjRecognition::CallbackFrame *&callback_data) {
    ObjRecognition::CallbackFrame frame;
    frame.id = (callback_data)->id;
    memcpy(&frame.t, (callback_data)->t, 3 * sizeof((callback_data)->t[0]));
    memcpy(
        &frame.R[0], (callback_data)->R[0],
        3 * sizeof((callback_data)->R[0][0]));
    memcpy(
        &frame.R[1], (callback_data)->R[1],
        3 * sizeof((callback_data)->R[1][0]));
    memcpy(
        &frame.R[2], (callback_data)->R[2],
        3 * sizeof((callback_data)->R[2][0]));

    frame.width = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    frame.height = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    frame.data = new unsigned char[frame.height * frame.width];
    memcpy(
        frame.data, (callback_data)->data,
        sizeof(char) * frame.height * frame.width);
    ObjRecongManager::Instance().Run(frame);
}
} // namespace ObjRecognitionExd
#endif // ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
