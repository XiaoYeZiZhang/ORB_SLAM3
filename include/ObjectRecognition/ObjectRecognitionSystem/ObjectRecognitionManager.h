//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
#define ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
#include <memory>
#include <mutex>
#include "Interface/HandleBase.h"
#include "ObjectRecognitionSystem.h"
#include "Utility/Camera.h"
namespace ObjRecognitionExd {

class ObjRecongManager : public ObjRecognitionInterface::HandleBase {
public:
    static ObjRecongManager &Instance();
    ~ObjRecongManager();
    int Destroy();
    int Reset();

    int Run(const ObjRecognition::ObjRecogFrameCallbackData &platform_frame);
    int CreateWithConfig();
    int LoadDic(char const *buffer, int buffer_len);
    bool LoadVoc(std::string &voc_path);
    int LoadModel(
        const int id, const char *buffer, long long buffer_len,
        std::shared_ptr<ObjRecognition::Object> &object);

    ObjRecognition::ObjRecogResult GetObjRecognitionResult();
    int SetObjRecongInfo();
    char *GetVersion();

private:
    ObjRecongManager();
    void Clear();
    int Init();
    /*
    void ProcessFrame(const STPlatformFrame &platformFrame);
    void CheckDatasetDir();
    void CheckMapResultFileDir();
    bool SaveDatainAndroid(const STPlatformFrame &platformFrame);*/

private:
    std::mutex mMutexForPublicAPI;

    std::map<int, std::shared_ptr<ObjRecognition::Object>> object_map_;
    std::shared_ptr<ObjRecognition::Object> object_;
    std::shared_ptr<DBoW3::Vocabulary> voc_;
    ObjRecognition::ObjRecogThread objrecog_thread_;
    std::string version_ = "V1.0.1.0";
    char *version_buffer;

    const int info_Buffer_maxsize_ = 1024;
    int info_buffer_length_;
    char *info_buffer_;

    int obj_num_;
    int *obj_id_buffer_;
    int *obj_state_buffer_;
    float *obj_rotation_buffer_;
    float *obj_translation_buffer_;
    float *model_bounding_box_;

    std::mutex m_mutex_info_buffer;
};

static void
ObjRecogCallback(ObjRecognition::ObjRecogFrameCallbackData *&callback_data) {
    ObjRecognition::ObjRecogFrameCallbackData frame;
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

    frame.feature_mem_size = 0;

    frame.img.width = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    frame.img.height = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    frame.img.data = new unsigned char[frame.img.height * frame.img.width];
    memcpy(
        frame.img.data, (callback_data)->img.data,
        sizeof(char) * frame.img.height * frame.img.width);
    frame.has_image = true;
    frame.timestamp = (callback_data)->timestamp;

    ObjRecongManager::Instance().Run(frame);
}
} // namespace ObjRecognitionExd
#endif // ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
