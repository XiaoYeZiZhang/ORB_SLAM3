//
// Created by zhangye on 2020/9/16.
//
#include "Tracker/TrackerThread.h"
namespace ObjRecognition {
TrackerThread::TrackerThread() {
    VLOG(20) << "TrackerThread: create";
}

TrackerThread::~TrackerThread() {
    VLOG(20) << "~TrackerThread";
}

void TrackerThread::SetTracker(
    const std::shared_ptr<ObjRecognition::RecognitionBase> &tracker) {
    mTracker = tracker;
}

void TrackerThread::Process() {
    VLOG(20) << "TrackerThread::Process";
    mTracker->Process(mCurData);
}

void TrackerThread::Stop() {
    VLOG(20) << "TrackerThread::Stop";
    mTracker->Clear();
}

void TrackerThread::Reset() {
    VLOG(20) << "TrackerThread::Reset";
    mTracker->Reset();
}

void TrackerThread::GetCurInputData() {
    std::lock_guard<std::mutex> lck(mInputMutex);
    if (!mInputQueue.empty()) {
        VLOG(0) << "tracker: queue size: " << mInputQueue.size();
        mCurData = mInputQueue.back();
        std::queue<std::shared_ptr<FrameData>> emptyQueue;
        std::swap(emptyQueue, mInputQueue);
    } else {
        mCurData = InputDataPtr();
    }
}
} // namespace ObjRecognition