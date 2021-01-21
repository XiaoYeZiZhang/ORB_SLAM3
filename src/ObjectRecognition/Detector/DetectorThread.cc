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
    m_detector = detector;
}

void DetectorThread::Process() {
    VLOG(20) << "DetectorThread::Process";
    m_detector->Process(m_curData);
}

void DetectorThread::Stop() {
    VLOG(20) << "DetectorThread::Stop";
    m_detector->Clear();
}

void DetectorThread::Reset() {
    VLOG(20) << "DetectorThread::Reset";
    m_detector->Reset();
}

void DetectorThread::GetNewestData() {
    std::lock_guard<std::mutex> lck(m_input_mutex);
    if (!m_input_queue.empty()) {
        m_curData = m_input_queue[0];
        m_input_queue.clear();
    } else {
        m_curData = InputDataPtr();
    }
}
} // namespace ObjRecognition