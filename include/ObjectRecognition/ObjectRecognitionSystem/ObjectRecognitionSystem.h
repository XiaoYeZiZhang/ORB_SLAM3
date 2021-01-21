#ifndef ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
#define ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
#include "SPextractor.h"
#include "TrackerPointCloud.h"
#include "DetectorPointCloud.h"
#include "PointCloudObject.h"
#include "DetectorThread.h"
#include "TrackerThread.h"
#include "ThreadBase.h"
namespace ObjRecognition {

class ObjRecogThread : public ThreadBase<CallbackFrame> {

public:
    ObjRecogThread();
    ~ObjRecogThread() {
        delete m_SPextractor;
    }

    int Init();

    int SetVocabulary(const std::shared_ptr<DBoW3::Vocabulary> &voc);
    int SetModel(const std::shared_ptr<Object> &object);

    void GetResult(
        FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
        Eigen::Matrix3d &R_cam, Eigen::Vector3d &t_cam, Eigen::Matrix3d &R_obj,
        Eigen::Vector3d &t_obj);

protected:
    void Reset();
    void Stop();
    void Process();
    void GetNewestData();

private:
    long int last_processed_frame = -1;
    std::shared_ptr<CallbackFrame> m_curData;
    std::shared_ptr<DBoW3::Vocabulary> m_voc;
    std::shared_ptr<ObjRecognition::Object> m_object;

    ObjRecognition::DetectorThread m_detector_thread;
    ObjRecognition::TrackerThread m_tracker_thread;

    std::shared_ptr<ObjRecognition::PointCloudObjDetector>
        m_pointcloudobj_detector;
    std::shared_ptr<ObjRecognition::PointCloudObjTracker>
        m_pointcloudobj_tracker;
    ORB_SLAM3::SPextractor *m_SPextractor = NULL;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_OBJECTRECOGNITIONSYSTEM_H
