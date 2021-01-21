#ifndef ORB_SLAM3_SUPERPOINT_H
#define ORB_SLAM3_SUPERPOINT_H
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "mode.h"
#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

namespace ORB_SLAM3 {

struct SuperPoint : torch::nn::Module {
    SuperPoint();
};

class SPDetector {
public:
    SPDetector(
        torch::jit::script::Module level1_module,
        torch::jit::script::Module level2_module,
        torch::jit::script::Module level3_module, bool is_mono = false);

    SPDetector() {
    }

    void detect(cv::Mat &image, int level, bool cuda);
    void getKeyPoints(
        float threshold, int iniX, int maxX, int iniY, int maxY,
        std::vector<cv::KeyPoint> &keypoints, bool nms);
    void computeDescriptors(
        const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,
        bool cuda);

private:
    torch::jit::script::Module m_traced_module_480_640;
    torch::jit::script::Module m_traced_module_400_533;
    torch::jit::script::Module m_traced_module_333_444;
    torch::jit::script::Module m_traced_module_384_512;
    torch::jit::script::Module m_traced_module_320_427;
    torch::jit::script::Module m_traced_module_267_356;
    torch::Tensor m_desc;
    torch::Tensor m_prob_cpu;
};

} // namespace ORB_SLAM3
#endif // ORB_SLAM3_SUPERPOINT_H
