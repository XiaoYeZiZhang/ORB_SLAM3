#ifndef STSLAM_FEATUREEXTRACTOR_H_
#define STSLAM_FEATUREEXTRACTOR_H_

#include <opencv2/imgproc.hpp>
#include <vector>

namespace SLAMCommon {

void ExtractNewFeatures(
    const cv::Mat &img, std::vector<cv::Point2f> &new_pts, int maxCorners,
    double qualityLevel = 0.01, double minDistance = 30,
    const cv::Mat &mask = cv::Mat());

/// calculate descriptor distance
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

// FAST
void ExtractFastFeatures(
    const cv::Mat &img, std::vector<cv::Point2f> &newPts, int maxCorners,
    const cv::Mat &mask = cv::Mat(), int threshold = 10,
    bool nonmaxSuppression = true);
} // namespace SLAMCommon
#endif