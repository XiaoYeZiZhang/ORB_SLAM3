#ifndef ORB_SLAM3_GLOBALIMAGEVIEWER_H
#define ORB_SLAM3_GLOBALIMAGEVIEWER_H
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include "Struct/PointCloudObject.h"

namespace ObjRecognition {
class GlobalKeyFrameMatchViewer {
public:
    static void GetMatchedKeyFrames(std::set<int> &matchedKeyFramesId);
    static void SetMatchedKeyFrames(const std::set<int> &matchedKeyFramesId);

private:
    static std::set<int> matched_keyFrames_id_;
};

class GlobalOcvViewer {
public:
    static void AddView(const std::string &window_name, cv::Mat &preview);
    static void Draw();

private:
    static std::map<std::string, cv::Mat> m_previews_view;
    static std::mutex m_viewer_mutex;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_GLOBALIMAGEVIEWER_H
