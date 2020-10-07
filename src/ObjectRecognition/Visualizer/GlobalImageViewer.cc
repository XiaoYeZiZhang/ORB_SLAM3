//
// Created by root on 2020/10/7.
//

#include "Visualizer/GlobalImageViewer.h"
namespace ObjRecognition {

std::map<std::string, cv::Mat> GlobalOcvViewer::s_previews;
std::mutex GlobalOcvViewer::s_mutex_viewer;
int GlobalOcvViewer::s_wait_time = 1;
std::vector<ObjRecognition::MapPointIndex>
    GlobalPointCloudMatchViewer::matched_mapPoints_;
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
void GlobalPointCloudMatchViewer::SetMatchedMapPoint(
    const std::vector<MapPointIndex> &matchedMapPoints) {
    matched_mapPoints_.clear();
    matched_mapPoints_ = matchedMapPoints;
}

void GlobalPointCloudMatchViewer::GetMatchedMapPoint(
    std::vector<MapPointIndex> &matchedMapPoints) {
    matchedMapPoints = matched_mapPoints_;
}

void GlobalPointCloudMatchViewer::DrawMatchedMapPoint(
    const std::vector<MapPoint::Ptr> &pointClouds, const Eigen::Isometry3f &T,
    const std::vector<MapPointIndex> &match3dId,
    std::vector<Eigen::Vector3f> &mapPoints) {
    for (int i = 0; i < match3dId.size(); i++) {
        Eigen::Vector3f p = pointClouds[match3dId[i]]->GetPose().cast<float>();
        p = T.inverse() * p;
        mapPoints.emplace_back(p);
    }
}

void GlobalOcvViewer::UpdateView(std::string window_name, cv::Mat &preview) {
    std::lock_guard<std::mutex> lk(s_mutex_viewer);
    s_previews[window_name] = preview.clone();
}

void GlobalOcvViewer::DrawAllView() {
    s_mutex_viewer.lock();
    auto previews = s_previews;
    s_mutex_viewer.unlock();
    for (const auto &pair : previews) {
        cv::imshow(pair.first, pair.second);
    }
    cv::waitKey(s_wait_time);
}

void GlobalOcvViewer::DrawSingleView(std::string window_name) {
    s_mutex_viewer.lock();
    auto previews = s_previews;
    s_mutex_viewer.unlock();
    auto it = previews.find(window_name);
    if (it != previews.end()) {
        cv::imshow(it->first, it->second);
        cv::waitKey(s_wait_time);
    } else {
        //        PRINT_W("[GlobalOcvViewer] DrawSingleView no preview matches
        //        %s",
        //                window_name.c_str());
    }
}

} // namespace ObjRecognition
