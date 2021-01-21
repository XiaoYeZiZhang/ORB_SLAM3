#ifndef ORB_SLAM3_SUPERPOINTMATCHER_H
#define ORB_SLAM3_SUPERPOINTMATCHER_H
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"
namespace ORB_SLAM3 {

class SuperPointMatcher {
public:
    explicit SuperPointMatcher(float nnratio = 0.6);
    static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    static int SearchByProjection(
        KeyFrame &CurrentKeyFrame, KeyFrame &LastKeyFrame, float th,
        bool bMono);
    static void FindMatchByBruteForce(
        const std::vector<cv::KeyPoint> &keypoints1,
        const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &frmDesp,
        const cv::Mat &pcDesp, std::vector<cv::DMatch> &goodMatches);
    int SearchByBoW(
        KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);
    int SearchForTriangulation(
        KeyFrame *pKF1, KeyFrame *pKF2, const cv::Mat &F12,
        std::vector<pair<size_t, size_t>> &vMatchedPairs);

public:
    static const float TH_LOW;
    static const float TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    bool CheckDistEpipolarLine(
        const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
        const KeyFrame *pKF);
    float mfNNratio;
};
} // namespace ORB_SLAM3
#endif // ORB_SLAM3_SUPERPOINTMATCHER_H
