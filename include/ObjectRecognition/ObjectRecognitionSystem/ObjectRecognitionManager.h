//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
#define ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
#include <memory>
#include <mutex>
#include "Interface/HandleBase.h"
#include "ObjectRecognitionSystem.h"
namespace ObjRecognitionExd {
class ObjRecongManager : public ObjRecognitionInterface::HandleBase {
public:
    static ObjRecongManager &Instance();
    ~ObjRecongManager();
    int Destroy();
    int Reset();

    int Run(const ObjRecognition::ObjRecogFrameCallbackData &platform_frame);
    int CreateWithConfig();
    /*int LoadDic(char const *buffer, int buffer_len);*/
    int LoadModel(const int id, const char *buffer, int buffer_len);

    ObjRecognition::ObjRecogResult GetObjRecognitionResult();
    int SetObjRecongInfo();
    /*char *GetVersion();*/

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
    // std::shared_ptr<DBoW3::Vocabulary> voc_;
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
} // namespace ObjRecognitionExd
#endif // ORB_SLAM3_OBJECTRECOGNITIONMANAGER_H
