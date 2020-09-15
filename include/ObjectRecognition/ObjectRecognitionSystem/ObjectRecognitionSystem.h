//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
#define ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
#include "include/ObjectRecognition/Utility/Thread/ThreadBase.h"
namespace ObjRecognition {

class ObjRecogThread : public Common::ThreadBase {

public:
    ObjRecogThread();

    int Init();

    /*int SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    int SetModel(const std::shared_ptr<Object> &object);

    void PushUnProcessedFrame(
        const std::shared_ptr<ObjRecogFrameCallbackData> &frame);

    void GetResult(
        FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
        Mat3d &R_cam, Vec3d &t_cam, Mat3d &R_obj, Vec3d &t_obj);*/

    // int GetInfo(std::string &info);

protected:
    enum { DATA_TYPE_UNPROCESSED_FRAME = 0 };
    int Reset();
    int Stop();
    int Process();

    // void SetInfo();

private:
    /*std::shared_ptr<DBoW3::Vocabulary> voc_;

    std::shared_ptr<STObjRecognition::Object> object_;

    STObjRecognition::DetectorThread detector_thread_;
    STObjRecognition::TrackerThread tracker_thread_;

    std::shared_ptr<STObjRecognition::PointCloudObjDetector>
        pointcloudobj_detector_;
    std::shared_ptr<STObjRecognition::PointCloudObjTracker>
        pointcloudobj_tracker_;

    std::shared_ptr<STObjRecognition::OpticalFlowObjDetector>
        opticalFlowobj_detector_;
    std::shared_ptr<STObjRecognition::OpticalFlowObjTracker>
        opticalFlowobj_tracker_;*/

    std::string info_;

    int frame_processed_num_ = 0;

    std::mutex mMutexInfoBuffer;

}; // namespace STObjRecognition
} // namespace ObjRecognition
#endif // ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
