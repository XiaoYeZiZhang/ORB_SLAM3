#ifndef _SensetimeSLAM_ORBEXTRACTORHPC_H_
#define _SensetimeSLAM_ORBEXTRACTORHPC_H_

#include "ORBExtractor.h"

namespace SLAMCommon {

class ORBExtractorHPC : public ORBExtractor {
public:
    ORBExtractorHPC(
        int nfeatures, float scaleFactor, int nlevels, int iniThFAST,
        int minThFAST);

    int ComputeDescriptorsWithoutScale(
        const cv::Mat &image, std::vector<cv::KeyPoint> &keyPoints,
        cv::Mat &outDescriptors);

protected:
    int DetectRawFastKeyPoints(
        const std::vector<cv::Mat> &imagePyramid,
        std::vector<std::vector<cv::KeyPoint>> &outAllKeyPoints);

private:
    float pattern_x[kPatternSize];
    float pattern_y[kPatternSize];
};
} // namespace SLAMCommon
#endif