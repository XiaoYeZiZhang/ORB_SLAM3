#ifndef ORB_SLAM3_TRACKERTHREAD_H
#define ORB_SLAM3_TRACKERTHREAD_H
#include <memory>
#include <glog/logging.h>
#include "include/ObjectRecognition/ObjectRecognitionThread/RecognitionBase.h"
#include "Utility/ThreadBase.h"
#include "Struct/Frame.h"

namespace ObjRecognition {

class TrackerThread : public ThreadBase<FrameForObjRecognition> {
public:
    TrackerThread();
    ~TrackerThread();

    void SetTracker(const std::shared_ptr<RecognitionBase> &tracker);

protected:
    void Process();
    void Stop();
    void Reset();
    void GetNewestData();

protected:
    std::shared_ptr<RecognitionBase> m_tracker;
    std::shared_ptr<FrameForObjRecognition> m_curData;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_TRACKERTHREAD_H
