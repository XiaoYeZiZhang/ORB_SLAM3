#include <opencv2/opencv.hpp>
#include "ORBExtractorHPC.h"

using namespace cv;
using namespace std;

namespace SLAMCommon {

static void ComputeTwoOrbDescriptor_HPC(
    const KeyPoint &kpt0, const KeyPoint &kpt1, const Mat &img,
    const float *pattern_x, const float *pattern_y, uchar *desc0,
    uchar *desc1) {
#if DESKTOP_PLATFORM
    // PRINT_W("[computeTwoOrbDescriptor_HPC] only used in MOBILE_PLATFORM");
#endif
    float angle0 = (float)kpt0.angle * (ORBExtractor::factorPI);
    float angle1 = (float)kpt1.angle * (ORBExtractor::factorPI);
    float a0 = (float)cos(angle0), b0 = (float)sin(angle0);
    float a1 = (float)cos(angle1), b1 = (float)sin(angle1);

    const uchar *center0 =
        &img.at<uchar>(cvRound(kpt0.pt.y), cvRound(kpt0.pt.x));
    const uchar *center1 =
        &img.at<uchar>(cvRound(kpt1.pt.y), cvRound(kpt1.pt.x));
    const int step = (int)img.step;

    int index0[512];
    int index1[512];

#if defined(ANDROID) || defined(__ANDOIRD__)

    float32x4_t v_a0 = vdupq_n_f32(a0);
    float32x4_t v_b0 = vdupq_n_f32(b0);
    float32x4_t v_a1 = vdupq_n_f32(a1);
    float32x4_t v_b1 = vdupq_n_f32(b1);

    int32x4_t v_step = vdupq_n_s32(step);

    // calculate ( cvRound(pattern[idx].x * b + pattern[idx].y * a) * step +
    //             cvRound(pattern[idx].x * a + pattern[idx].y * b)            )
    // pattern_x and pattern_y are been multiply 2.0f for round.

    for (int i = 0; i < 512; i += 4) {
        float32x4_t v_x = vld1q_f32(pattern_x + i);
        float32x4_t v_y = vld1q_f32(pattern_y + i);

        float32x4_t v_ftmp0 = vmlaq_f32(vmulq_f32(v_x, v_b0), v_y, v_a0);
        float32x4_t v_ftmp1 = vmlsq_f32(vmulq_f32(v_x, v_a0), v_y, v_b0);
        float32x4_t v_ftmp2 = vmlaq_f32(vmulq_f32(v_x, v_b1), v_y, v_a1);
        float32x4_t v_ftmp3 = vmlsq_f32(vmulq_f32(v_x, v_a1), v_y, v_b1);

        int32x4_t v_stmp0 = vcvtq_s32_f32(v_ftmp0);
        int32x4_t v_stmp1 = vcvtq_s32_f32(v_ftmp1);
        int32x4_t v_stmp2 = vcvtq_s32_f32(v_ftmp2);
        int32x4_t v_stmp3 = vcvtq_s32_f32(v_ftmp3);

        uint32x4_t v_utmp0 = vshrq_n_u32((uint32x4_t)v_stmp0, 31);
        uint32x4_t v_utmp1 = vshrq_n_u32((uint32x4_t)v_stmp1, 31);
        uint32x4_t v_utmp2 = vshrq_n_u32((uint32x4_t)v_stmp2, 31);
        uint32x4_t v_utmp3 = vshrq_n_u32((uint32x4_t)v_stmp3, 31);

        v_stmp0 = vsubq_s32(v_stmp0, (int32x4_t)v_utmp0);
        v_stmp1 = vsubq_s32(v_stmp1, (int32x4_t)v_utmp1);
        v_stmp2 = vsubq_s32(v_stmp2, (int32x4_t)v_utmp2);
        v_stmp3 = vsubq_s32(v_stmp3, (int32x4_t)v_utmp3);

        v_stmp0 = vrshrq_n_s32(v_stmp0, 1);
        v_stmp1 = vrshrq_n_s32(v_stmp1, 1);
        v_stmp2 = vrshrq_n_s32(v_stmp2, 1);
        v_stmp3 = vrshrq_n_s32(v_stmp3, 1);

        vst1q_s32(index0 + i, vmlaq_s32(v_stmp1, v_stmp0, v_step));
        vst1q_s32(index1 + i, vmlaq_s32(v_stmp3, v_stmp2, v_step));
    }
#else
    // PRINT_W("[computeTwoOrbDescriptor_HPC] only for android platform
    // !!!!!!");
#endif

#define GET_VALUE0(idx) center0[p0[idx]]
#define GET_VALUE1(idx) center1[p1[idx]]

    int *p0 = index0;
    int *p1 = index1;

    for (int i = 0; i < 32; ++i, p0 += 16, p1 += 16) {
        int t0, t1, val0;
        int t2, t3, val1;
        t0 = GET_VALUE0(0);
        t1 = GET_VALUE0(1);
        t2 = GET_VALUE1(0);
        t3 = GET_VALUE1(1);
        val0 = t0 < t1;
        val1 = t2 < t3;
        t0 = GET_VALUE0(2);
        t1 = GET_VALUE0(3);
        t2 = GET_VALUE1(2);
        t3 = GET_VALUE1(3);
        val0 |= (t0 < t1) << 1;
        val1 |= (t2 < t3) << 1;
        t0 = GET_VALUE0(4);
        t1 = GET_VALUE0(5);
        t2 = GET_VALUE1(4);
        t3 = GET_VALUE1(5);
        val0 |= (t0 < t1) << 2;
        val1 |= (t2 < t3) << 2;
        t0 = GET_VALUE0(6);
        t1 = GET_VALUE0(7);
        t2 = GET_VALUE1(6);
        t3 = GET_VALUE1(7);
        val0 |= (t0 < t1) << 3;
        val1 |= (t2 < t3) << 3;
        t0 = GET_VALUE0(8);
        t1 = GET_VALUE0(9);
        t2 = GET_VALUE1(8);
        t3 = GET_VALUE1(9);
        val0 |= (t0 < t1) << 4;
        val1 |= (t2 < t3) << 4;
        t0 = GET_VALUE0(10);
        t1 = GET_VALUE0(11);
        t2 = GET_VALUE1(10);
        t3 = GET_VALUE1(11);
        val0 |= (t0 < t1) << 5;
        val1 |= (t2 < t3) << 5;
        t0 = GET_VALUE0(12);
        t1 = GET_VALUE0(13);
        t2 = GET_VALUE1(12);
        t3 = GET_VALUE1(13);
        val0 |= (t0 < t1) << 6;
        val1 |= (t2 < t3) << 6;
        t0 = GET_VALUE0(14);
        t1 = GET_VALUE0(15);
        t2 = GET_VALUE1(14);
        t3 = GET_VALUE1(15);
        val0 |= (t0 < t1) << 7;
        val1 |= (t2 < t3) << 7;

        desc0[i] = (uchar)val0;
        desc1[i] = (uchar)val1;
    }
#undef GET_VALUE0
#undef GET_VALUE1
}

ORBExtractorHPC::ORBExtractorHPC(
    int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST)
    : ORBExtractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST) {
    // for HPC
    for (int i = 0; i < kPatternSize; ++i) {
        pattern_x[i] = pattern.at(i).x * 2.0f;
        pattern_y[i] = pattern.at(i).y * 2.0f;
    }
}

int ORBExtractorHPC::ComputeDescriptorsWithoutScale(
    const cv::Mat &image, std::vector<cv::KeyPoint> &keyPoints,
    cv::Mat &outDescriptors) {
#if DESKTOP_PLATFORM
    // PRINT_W("[computeDescriptors_HPC] only for mobile platform !!!!!!");
#endif

    if ((image.empty()) || (CV_8UC1 != image.type()) || (keyPoints.empty())) {
        return -1;
    }

    const int keyPointsCount = keyPoints.size();
    outDescriptors = Mat::zeros(keyPointsCount, 32, CV_8UC1);

    int i = 0;
    for (; i + 2 <= keyPointsCount; i += 2)
        ComputeTwoOrbDescriptor_HPC(
            keyPoints[i], keyPoints[i + 1], image, pattern_x, pattern_y,
            outDescriptors.ptr(i), outDescriptors.ptr((i + 1)));
    for (; i < keyPointsCount; ++i) {
        ComputeOrbDescriptor(
            keyPoints[i], image, &(pattern[0]), outDescriptors.ptr(i));
    }

    return 0;
}

int ORBExtractorHPC::DetectRawFastKeyPoints(
    const std::vector<cv::Mat> &imagePyramid,
    std::vector<std::vector<cv::KeyPoint>> &outAllKeyPoints) {
    if (nlevels != imagePyramid.size()) {
        return -1;
    }

    outAllKeyPoints.clear();

    const float W = 30;

    for (int level = 0; level < nlevels; ++level) {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = imagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = imagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        if (nRows >= 1 && nCols >= 1) {
            FAST(
                imagePyramid[level]
                    .rowRange(minBorderY, maxBorderY)
                    .colRange(minBorderX, maxBorderX),
                vToDistributeKeys, iniThFAST, true);
            vector<vector<bool>> cellFlag;
            cellFlag.resize(nRows);
            for (int i = 0; i < nRows; ++i) {
                cellFlag[i].resize(nCols);
                for (int j = 0; j < nCols; ++j) {
                    cellFlag[i][j] = false;
                }
            }
            float r_hCell = 1.0f / hCell;
            float r_wCell = 1.0f / wCell;

            // calculate which cells have point.
            for (auto vit = vToDistributeKeys.begin();
                 vit != vToDistributeKeys.end(); ++vit) {
                int i = floor((*vit).pt.y * r_hCell);
                int j = floor((*vit).pt.x * r_wCell);
                cellFlag[i][j] = true;
            }

            // calculate for rest cells with minThFAST
            for (int i = 0; i < nRows; i++) {
                const float iniY = minBorderY + i * hCell;
                float maxY = iniY + hCell + 6;

                if (iniY >= maxBorderY - 3)
                    continue;
                if (maxY > maxBorderY)
                    maxY = maxBorderY;

                for (int j = 0; j < nCols; j++) {
                    const float iniX = minBorderX + j * wCell;
                    float maxX = iniX + wCell + 6;
                    if (iniX >= maxBorderX - 6)
                        continue;
                    if (maxX > maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;

                    if (!cellFlag[i][j]) {
                        FAST(
                            imagePyramid[level]
                                .rowRange(iniY, maxY)
                                .colRange(iniX, maxX),
                            vKeysCell, minThFAST, true);
                    }

                    if (!vKeysCell.empty()) {
                        for (vector<cv::KeyPoint>::iterator vit =
                                 vKeysCell.begin();
                             vit != vKeysCell.end(); vit++) {
                            (*vit).pt.x += j * wCell;
                            (*vit).pt.y += i * hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }
                }
            }
        }

        outAllKeyPoints.emplace_back(std::move(vToDistributeKeys));
    }

    return 0;
}
} // namespace SLAMCommon