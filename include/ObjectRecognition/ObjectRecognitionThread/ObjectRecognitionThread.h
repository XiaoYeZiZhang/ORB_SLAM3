#ifndef ORB_SLAM3_OBJECTRECOGNITIONTHREAD_H
#define ORB_SLAM3_OBJECTRECOGNITIONTHREAD_H
#include "Camera.h"
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
class ObjRecongManager {
public:
    static ObjRecongManager &Instance();
    ~ObjRecongManager();
    int Run(const ObjRecognition::CallbackFrame &platform_frame);
    void SetThreadHandler(
        std::shared_ptr<ObjRecognition::ObjRecogThread> &thread_handler);

private:
    ObjRecongManager();

private:
    std::shared_ptr<ObjRecognition::ObjRecogThread> m_objrecog_thread;
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

    frame.width = CameraIntrinsic::GetInstance().Width();
    frame.height = CameraIntrinsic::GetInstance().Height();
    frame.data = new unsigned char[frame.height * frame.width];
    memcpy(
        frame.data, (callback_data)->data,
        sizeof(char) * frame.height * frame.width);

    ObjRecongManager::Instance().Run(frame);
}
} // namespace ObjRecognition
#endif // ORB_SLAM3_OBJECTRECOGNITIONTHREAD_H
