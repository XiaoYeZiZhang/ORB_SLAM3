//
// Created by root on 2020/10/9.
//

#ifndef ORB_SLAM3_VIEWERCOMMON_H
#define ORB_SLAM3_VIEWERCOMMON_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
void AddTextToImage(
    const std::string &s, cv::Mat &im, const int r, const int g, const int b);
void PrintSLAMStatusForViewer(
    const int &status, const int &image_num, cv::Mat &img);
void PrintStatus(
    const int &status, const bool &bLocMode, const int mappoint_num,
    cv::Mat &im);
void LoadCameraPose(const cv::Mat &Tcw);
void DrawImageTexture(pangolin::GlTexture &imageTexture, cv::Mat &im);
#endif // ORB_SLAM3_VIEWERCOMMON_H
