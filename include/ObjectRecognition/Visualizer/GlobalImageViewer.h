//
// Created by root on 2020/10/7.
//

#ifndef ORB_SLAM3_GLOBALIMAGEVIEWER_H
#define ORB_SLAM3_GLOBALIMAGEVIEWER_H
//
// Created by root on 2020/10/7.
//
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include "Struct/PointCloudObject.h"

namespace ObjRecognition {
class GlobalKeyFrameMatchViewer {
private:
    static std::set<int> matched_keyFrames_id_;

public:
    static void GetMatchedKeyFrames(std::set<int> &matchedKeyFramesId);
    static void SetMatchedKeyFrames(const std::set<int> &matchedKeyFramesId);
};
class GlobalPointCloudMatchViewer {
private:
    static std::vector<ObjRecognition::MapPointIndex> matched_mapPoints_;

public:
    static void SetMatchedMapPoint(
        const std::vector<ObjRecognition::MapPointIndex> &matchedMapPoints);
    static void GetMatchedMapPoint(
        std::vector<ObjRecognition::MapPointIndex> &matchedMapPoints);
    static void DrawMatchedMapPoint(
        const std::vector<MapPoint::Ptr> &pointClouds,
        const Eigen::Isometry3f &T,
        const std::vector<ObjRecognition::MapPointIndex> &match3dId,
        std::vector<Eigen::Vector3f> &mapPoints);
};
class GlobalOcvViewer {
private:
    static std::map<std::string, cv::Mat> s_previews;
    static std::mutex s_mutex_viewer;

public:
    static int s_wait_time;

    /// update view in other threads
    static void UpdateView(std::string window_name, cv::Mat &preview);

    /// draw all views in main thread
    static void DrawAllView();

    /// draw single view in main thread
    static void DrawSingleView(std::string window_name);
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_GLOBALIMAGEVIEWER_H
