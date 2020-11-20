//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include "Detector/DetectorThread.h"

namespace ObjRecognition {
DetectorThread::DetectorThread() {
    VLOG(20) << "DetectorThread: create";
}

DetectorThread::~DetectorThread() {
    VLOG(20) << "~DetectorThread";
}

void DetectorThread::SetDetector(
    const std::shared_ptr<ObjRecognition::RecognitionBase> &detector) {
    mDetector = detector;
}

void DetectorThread::Process() {
    VLOG(20) << "DetectorThread::Process";
    mDetector->Process(mCurData);
}

void DetectorThread::Stop() {
    VLOG(20) << "DetectorThread::Stop";
    mDetector->Clear();
}

void DetectorThread::Reset() {
    VLOG(20) << "DetectorThread::Reset";
    mDetector->Reset();
}

void DetectorThread::GetCurInputData() {
    std::lock_guard<std::mutex> lck(mInputMutex);
    if (!mInputQueue.empty()) {
        VLOG(0) << "detector: queue size: " << mInputQueue.size();
        mCurData = mInputQueue.back();
        std::queue<std::shared_ptr<FrameData>> emptyQueue;
        std::swap(emptyQueue, mInputQueue);
    } else {
        mCurData = InputDataPtr();
    }
}
} // namespace ObjRecognition