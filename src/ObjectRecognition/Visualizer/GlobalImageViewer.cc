#include "Visualizer/GlobalImageViewer.h"
namespace ObjRecognition {

std::map<std::string, cv::Mat> GlobalOcvViewer::m_previews_view;
std::mutex GlobalOcvViewer::m_viewer_mutex;
std::set<int> GlobalKeyFrameMatchViewer::matched_keyFrames_id_;
void GlobalKeyFrameMatchViewer::GetMatchedKeyFrames(
    std::set<int> &matchedKeyFramesId) {
    matchedKeyFramesId.clear();
    matchedKeyFramesId = matched_keyFrames_id_;
}

void GlobalKeyFrameMatchViewer::SetMatchedKeyFrames(
    const std::set<int> &matchedKeyFramesId) {
    matched_keyFrames_id_.clear();
    matched_keyFrames_id_ = matchedKeyFramesId;
}

void GlobalOcvViewer::AddView(
    const std::string &window_name, cv::Mat &preview) {
    std::lock_guard<std::mutex> lk(m_viewer_mutex);
    m_previews_view[window_name] = preview.clone();
}

void GlobalOcvViewer::Draw() {
    m_viewer_mutex.lock();
    auto previews = m_previews_view;
    m_viewer_mutex.unlock();
    for (const auto &pair : previews) {
        cv::imshow(pair.first, pair.second);
    }
    cv::waitKey(1);
}
} // namespace ObjRecognition
