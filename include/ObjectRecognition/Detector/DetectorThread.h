#ifndef ORB_SLAM3_DETECTORTHREAD_H
#define ORB_SLAM3_DETECTORTHREAD_H
#include "RecognitionBase.h"
#include "ThreadBase.h"
#include "Frame.h"

namespace ObjRecognition {

class DetectorThread : public ThreadBase<FrameForObjRecognition> {
public:
    DetectorThread();
    ~DetectorThread();

    void SetDetector(const std::shared_ptr<RecognitionBase> &detector);

protected:
    void Process();
    void Stop();
    void Reset();
    void GetNewestData();

protected:
    std::shared_ptr<RecognitionBase> m_detector;
    std::shared_ptr<FrameForObjRecognition> m_curData;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_DETECTORTHREAD_H
