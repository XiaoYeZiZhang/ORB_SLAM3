//
// Created by root on 2020/10/21.
//

#ifndef ORB_SLAM3_SPEXTRACTOR_H
#define ORB_SLAM3_SPEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv/cv.h>
#include <torch/torch.h>
#include "ORBSLAM3/SuperPoint.h"
#include <glog/logging.h>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

namespace ORB_SLAM3 {

class ExtractorNode_sp {
public:
    ExtractorNode_sp() : bNoMore(false) {
    }

    void DivideNode(
        ExtractorNode_sp &n1, ExtractorNode_sp &n2, ExtractorNode_sp &n3,
        ExtractorNode_sp &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode_sp>::iterator lit;
    bool bNoMore;
};

class SPextractor {
public:
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    SPextractor(
        int descriptor_len, int nfeatures, float scaleFactor, int nlevels,
        float iniThFAST, float minThFAST, bool is_use_cuda);

    ~SPextractor() {
    }

    // Compute the SP features and descriptors on an image.
    // SP are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator()(
        cv::InputArray image, const cv::Mat &mask,
        std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors);

    int inline GetLevels() {
        return nlevels;
    }

    float inline GetScaleFactor() {
        return scaleFactor;
    }

    std::vector<float> inline GetScaleFactors() {
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors() {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares() {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares() {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(
        std::vector<std::vector<cv::KeyPoint>> &allKeypoints, cv::Mat &_desc);
    void ComputeKeyPointsWithMask(
        std::vector<std::vector<cv::KeyPoint>> &allKeypoints, cv::Mat &_desc,
        const cv::Mat &mask);
    std::vector<cv::KeyPoint> DistributeOctTree(
        const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
        const int &maxX, const int &minY, const int &maxY, const int &nFeatures,
        const int &level);

    // void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >&
    // allKeypoints); std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    float iniThFAST;
    float minThFAST;
    int m_descriptor_len;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    torch::jit::script::Module traced_module_480_640;
    torch::jit::script::Module traced_module_400_533;
    torch::jit::script::Module traced_module_333_444;
    bool is_use_cuda;
};
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_SPEXTRACTOR_H
