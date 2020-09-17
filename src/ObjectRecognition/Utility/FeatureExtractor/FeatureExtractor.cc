#include "FeatureExtractor.h"

#include <opencv2/features2d.hpp>

namespace SLAMCommon {

void ExtractNewFeatures(
    const cv::Mat &img, std::vector<cv::Point2f> &newPts, int maxCorners,
    double qualityLevel, double minDistance, const cv::Mat &mask) {
    cv::goodFeaturesToTrack(
        img, newPts, maxCorners, qualityLevel, minDistance, mask);
}

int HammingDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    //    switch (DescTrackerParam::FEATURE_TYPE) {
    //        case DescTrackerParam::ORB_SENSE:
    //            return HammingDistance(a, b);
    //        case DescTrackerParam::ORB_OPENCV:
    return HammingDistance(a, b);
    //         case DescTrackerParam::FREAK_OPENCV:
    //             return cv::norm(a, b, cv::NORM_HAMMING);
    //        default:
    //            throw std::runtime_error("unknown feature type");
    //    }
}

void ExtractFastFeatures(
    const cv::Mat &img, std::vector<cv::Point2f> &newPts, int maxCorners,
    const cv::Mat &mask, int threshold, bool nonmaxSuppression) {
    std::vector<cv::KeyPoint> newKps;
    cv::FAST(img, newKps, threshold, nonmaxSuppression);
    std::sort(
        newKps.begin(), newKps.end(),
        [](const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2) {
            return kpt1.response > kpt2.response;
        });

    for (int i = 0; i < newKps.size(); ++i) {
        if (maxCorners < 1)
            break;
        if (mask.at<uchar>(newKps[i].pt)) {
            --maxCorners;
            newPts.push_back(newKps[i].pt);
            cv::circle(mask, newKps[i].pt, 30, 1, -1);
        }
    }
}
} // namespace SLAMCommon