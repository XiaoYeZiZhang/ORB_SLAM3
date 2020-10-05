//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
#define ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
#include "Tracker/TrackerPointCloud.h"
#include "Detector/DetectorPointCloud.h"
#include "Struct/PointCloudObject.h"
#include "Detector/DetectorThread.h"
#include "Tracker/TrackerThread.h"
#include "Utility/Thread/ThreadBase.h"
namespace ObjRecognition {

class ObjRecogThread : public Common::ThreadBase {

public:
    ObjRecogThread();

    int Init();

    int SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    int SetModel(const std::shared_ptr<Object> &object);

    void PushUnProcessedFrame(
        const std::shared_ptr<ObjRecogFrameCallbackData> &frame);

    void GetResult(
        FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
        Eigen::Matrix3d &R_cam, Eigen::Vector3d &t_cam, Eigen::Matrix3d &R_obj,
        Eigen::Vector3d &t_obj);

    int GetInfo(std::string &info);

protected:
    enum { DATA_TYPE_UNPROCESSED_FRAME = 0 };
    int Reset();
    int Stop();
    int Process();
    void SetInfo();

private:
    std::shared_ptr<DBoW3::Vocabulary> voc_;
    std::shared_ptr<ObjRecognition::Object> object_;

    ObjRecognition::DetectorThread detector_thread_;
    ObjRecognition::TrackerThread tracker_thread_;

    std::shared_ptr<ObjRecognition::PointCloudObjDetector>
        pointcloudobj_detector_;
    std::shared_ptr<ObjRecognition::PointCloudObjTracker>
        pointcloudobj_tracker_;

    std::string info_;

    int frame_processed_num_ = 0;

    std::mutex mMutexInfoBuffer;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
