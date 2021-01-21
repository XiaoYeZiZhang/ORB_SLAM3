#include "Tracker/TrackerThread.h"
namespace ObjRecognition {
TrackerThread::TrackerThread() {
}

TrackerThread::~TrackerThread() {
}

void TrackerThread::SetTracker(
    const std::shared_ptr<ObjRecognition::RecognitionBase> &tracker) {
    m_tracker = tracker;
}

void TrackerThread::Process() {
    m_tracker->Process(m_curData);
}

void TrackerThread::Stop() {
    m_tracker->Clear();
}

void TrackerThread::Reset() {
    m_tracker->Reset();
}

void TrackerThread::GetNewestData() {
    std::lock_guard<std::mutex> lck(m_input_mutex);
    if (!m_input_queue.empty()) {
        m_curData = m_input_queue[0];
        m_input_queue.clear();
    } else {
        m_curData = InputDataPtr();
    }
}
} // namespace ObjRecognition