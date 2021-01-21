#ifndef ORB_SLAM3_SPEXTRACTOR_H
#define ORB_SLAM3_SPEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv/cv.h>
#include <torch/torch.h>
#include "ORBSLAM3/SuperPoint.h"
#include <glog/logging.h>
#include "mode.h"

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
    SPextractor(
        int descriptor_len, int nfeatures, float scaleFactor, int nlevels,
        float iniThFAST, float minThFAST, bool is_use_cuda);

    ~SPextractor() {
    }

    void operator()(
        cv::InputArray image, const cv::Mat &mask,
        std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors);

    int inline GetLevels() {
        return m_levels;
    }

    float inline GetScaleFactor() {
        return m_scaleFactor;
    }

    std::vector<float> inline GetScaleFactors() {
        return m_scalefactor;
    }

    std::vector<float> inline GetInverseScaleFactors() {
        return m_invscalefactor;
    }

    std::vector<float> inline GetScaleSigmaSquares() {
        return m_level_sigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares() {
        return m_invlevel_sigma2;
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

    int m_features;
    double m_scaleFactor;
    int m_levels;
    float m_iniTh_FAST;
    float m_minTh_FAST;
    int m_descriptor_len;

    std::vector<int> m_features_perlevel;

    std::vector<float> m_scalefactor;
    std::vector<float> m_invscalefactor;
    std::vector<float> m_level_sigma2;
    std::vector<float> m_invlevel_sigma2;

    torch::jit::script::Module m_traced_module_480_640;
    torch::jit::script::Module m_traced_module_400_533;
    torch::jit::script::Module m_traced_module_333_444;

    torch::jit::script::Module m_traced_module_384_512;
    torch::jit::script::Module m_traced_module_320_427;
    torch::jit::script::Module m_traced_module_267_356;
    bool is_use_cuda;
};
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_SPEXTRACTOR_H
