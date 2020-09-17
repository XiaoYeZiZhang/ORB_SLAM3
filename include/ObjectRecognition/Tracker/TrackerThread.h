//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_TRACKERTHREAD_H
#define ORB_SLAM3_TRACKERTHREAD_H
#include <memory>
#include <glog/logging.h>
#include "Utility/RecognitionBase.h"
#include "Utility/ThreadBase.h"
#include "Struct/Frame.h"

namespace ObjRecognition {

class TrackerThread : public ThreadBase<FrameData> {
public:
    TrackerThread();
    ~TrackerThread();

    void SetTracker(const std::shared_ptr<RecognitionBase> &tracker);

protected:
    void Process();
    void Stop();
    void Reset();
    void GetCurInputData();

protected:
    std::shared_ptr<RecognitionBase> mTracker;
    std::shared_ptr<FrameData> mCurData;
};
} // namespace ObjRecognition
#endif // ORB_SLAM3_TRACKERTHREAD_H
