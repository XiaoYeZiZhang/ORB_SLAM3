//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_DETECTORTHREAD_H
#define ORB_SLAM3_DETECTORTHREAD_H
#include "Utility/RecognitionBase.h"
#include "Utility/ThreadBase.h"
#include "Struct/Frame.h"

namespace ObjRecognition {

class DetectorThread : public ThreadBase<FrameData> {
public:
    DetectorThread();
    ~DetectorThread();

    void SetDetector(const std::shared_ptr<RecognitionBase> &detector);

protected:
    void Process();
    void Stop();
    void Reset();
    void GetCurInputData();

protected:
    std::shared_ptr<RecognitionBase> mDetector;
    std::shared_ptr<FrameData> mCurData;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_DETECTORTHREAD_H
