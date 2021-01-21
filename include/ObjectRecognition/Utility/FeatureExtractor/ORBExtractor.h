#ifndef STSLAM_ORBEXTRACTOR_H_
#define STSLAM_ORBEXTRACTOR_H_

#include <vector>
#include <memory>
#include <opencv2/imgproc.hpp>

namespace SLAMCommon {
class ORBExtractor {
public:
    static const float factorPI;

public:
    ORBExtractor(
        int nfeatures, float scaleFactor, int nlevels, int iniThFAST,
        int minThFAST);
    virtual ~ORBExtractor();

    const int nfeatures;
    const int nlevels;
    const int iniThFAST;
    const int minThFAST;

    const std::vector<float> &ScaleFactor() const {
        return mvScaleFactor;
    }

    virtual int DetectKeyPoints(
        const cv::Mat &image, std::vector<cv::KeyPoint> &outKeyPoints,
        const cv::Mat &mask = cv::Mat());

    virtual int ComputeDescriptors(
        const cv::Mat &image, std::vector<cv::KeyPoint> &keyPoints,
        cv::Mat &outDescriptors);

    virtual int ComputeDescriptorsWithoutScale(
        const cv::Mat &image, std::vector<cv::KeyPoint> &keyPoints,
        cv::Mat &outDescriptors);

protected:
    int
    ComputePyramid(const cv::Mat &image, std::vector<cv::Mat> &outImagePyramid);

    virtual int DetectRawFastKeyPoints(
        const std::vector<cv::Mat> &imagePyramid,
        std::vector<std::vector<cv::KeyPoint>> &outAllKeyPoints);

    int CullKeyPoints(
        const std::vector<cv::Mat> &imagePyramid,
        std::vector<std::vector<cv::KeyPoint>> &inOutKeyPoints,
        const cv::Mat &mask);

    int PostProcessKeyPoints(
        const std::vector<cv::Mat> &imagePyramid,
        std::vector<std::vector<cv::KeyPoint>> &inOutKeyPoints,
        const cv::Mat &mask);

    std::vector<cv::KeyPoint> ScaleKeyPointsToRawImage(
        const std::vector<std::vector<cv::KeyPoint>> &keyPointsWithScale);

    std::vector<std::vector<cv::KeyPoint>>
    ScaleKeyPointsToImagePyramid(const std::vector<cv::KeyPoint> &keyPoints);

    virtual int DetectKeyPoints(
        const cv::Mat &image, std::vector<cv::Mat> &outImagePyramid,
        std::vector<std::vector<cv::KeyPoint>> &outKeyPointsWithScale,
        const cv::Mat &mask);

    virtual int ComputeDescriptors(
        const std::vector<cv::Mat> &imagePyramid,
        std::vector<std::vector<cv::KeyPoint>> &keyPointsWithScale,
        cv::Mat &outDescriptors, const cv::Mat *imagePyramidGaussData = nullptr,
        int imagePyramidGaussCount = 0);

protected:
    static const int kPatternSize = 512;
    static const int PATCH_SIZE = 31;
    static const int EDGE_THRESHOLD = 19;

    std::vector<int> umax;
    std::vector<cv::Point> pattern;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
    std::vector<int> mnFeaturesPerLevel;

    std::vector<cv::Mat> ImagePyramidCache(const cv::Mat &keyImage) const;
    void SetImagePyramidCache(
        const cv::Mat &keyImage, const std::vector<cv::Mat> &imagePyramid);

private:
    void *m_imagePyramidKey;
    std::vector<cv::Mat> m_imagePyramidCache;
};

std::vector<cv::KeyPoint> DistributeOctTree(
    const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
    const int &maxX, const int &minY, const int &maxY, const int &nFeatures,
    const int &level);

int ComputeKeyPointsOrientation(
    const cv::Mat &image, std::vector<cv::KeyPoint> &inOutKeypoints,
    const std::vector<int> &umax);

void ComputeOrbDescriptor(
    const cv::KeyPoint &kpt, const cv::Mat &img, const cv::Point *pattern,
    uchar *desc);

void KeyPointsFilterByPixelsMask(
    std::vector<cv::KeyPoint> &inOutKeyPoints, const cv::Mat &mask,
    int minBorderX, int minBorderY, float scaleFactor = 1.0f);
} // namespace SLAMCommon
#endif