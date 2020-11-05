//
// Created by root on 2020/10/26.
//

#include "ORBSLAM3/SuperPointMatcher.h"
namespace ORB_SLAM3 {

// TODO(zhangye) find the meaning
const float SuperPointMatcher::TH_HIGH = 0.70;
const float SuperPointMatcher::TH_LOW = 0.30;
const int SuperPointMatcher::HISTO_LENGTH = 30;

SuperPointMatcher::SuperPointMatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri) {
}

int SuperPointMatcher::SearchForTriangulation(
    KeyFrame *pKF1, KeyFrame *pKF2, const cv::Mat &F12,
    std::vector<pair<size_t, size_t>> &vMatchedPairs) {
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec_superpoint;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec_superpoint;

    // Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w * Cw + t2w;
    const float invz = 1.0f / C2.at<float>(2);
    const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
    const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by SP Vocabulary
    // Compare only SP that share the same node

    int nmatches = 0;
    vector<bool> vbMatched2(pKF2->N_superpoint, false);
    vector<int> vMatches12(pKF1->N_superpoint, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                const size_t idx1 = f1it->second[i1];
                MapPoint *pMP1 = pKF1->GetSuperpointMapPoint(idx1);
                // If there is already a MapPoint skip
                if (pMP1)
                    continue;
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn_superpoint[idx1];
                const cv::Mat &d1 = pKF1->mDescriptors_superpoint.row(idx1);

                float bestDist = TH_LOW;
                int bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2;
                     i2++) {
                    size_t idx2 = f2it->second[i2];
                    MapPoint *pMP2 = pKF2->GetSuperpointMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if (vbMatched2[idx2] || pMP2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors_superpoint.row(idx2);
                    const float dist = DescriptorDistance(d1, d2);

                    if (dist > TH_LOW || dist > bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn_superpoint[idx2];

                    const float distex = ex - kp2.pt.x;
                    const float distey = ey - kp2.pt.y;
                    if (distex * distex + distey * distey <
                        100 * pKF2->mvScaleFactors_suerpoint[kp2.octave])
                        continue;

                    if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2)) {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestIdx2 >= 0) {
                    const cv::KeyPoint &kp2 =
                        pKF2->mvKeysUn_superpoint[bestIdx2];
                    vMatches12[idx1] = bestIdx2;
                    nmatches++;

                    if (false) {
                        float rot = kp1.angle - kp2.angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVec1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (false) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                vMatches12[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
        if (vMatches12[i] < 0)
            continue;
        vMatchedPairs.emplace_back(make_pair(i, vMatches12[i]));
    }
    return nmatches;
}

// project other keyframe mappoint to current keyframe
int SuperPointMatcher::SearchByProjection(
    KeyFrame &CurrentKeyFrame, KeyFrame &LastKeyFrame, const float th,
    const bool bMono) {
    int nmatches = 0;
    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    cv::Mat Rcw_cur = CurrentKeyFrame.GetRotation();
    cv::Mat tcw_cur = CurrentKeyFrame.GetTranslation();
    const cv::Mat twc = -Rcw_cur.t() * tcw_cur;

    cv::Mat Rcw_other = LastKeyFrame.GetRotation();
    cv::Mat tcw_other = LastKeyFrame.GetTranslation();
    const cv::Mat twc_other = Rcw_other * twc + tcw_other;

    for (int i = 0; i < LastKeyFrame.N_superpoint; i++) {
        MapPoint *pMP_superpoint = LastKeyFrame.GetSuperpointMapPoint(i);
        if (pMP_superpoint) {
            if (!pMP_superpoint->isBad()) {
                // Project
                cv::Mat x3Dw = pMP_superpoint->GetWorldPos();
                cv::Mat x3Dc = Rcw_cur * x3Dw + tcw_cur;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0 / x3Dc.at<float>(2);
                if (invzc < 0)
                    continue;

                float u = CurrentKeyFrame.fx * xc * invzc + CurrentKeyFrame.cx;
                float v = CurrentKeyFrame.fy * yc * invzc + CurrentKeyFrame.cy;

                if (u < CurrentKeyFrame.mnMinX || u > CurrentKeyFrame.mnMaxX)
                    continue;
                if (v < CurrentKeyFrame.mnMinY || v > CurrentKeyFrame.mnMaxY)
                    continue;

                int nLastOctave = LastKeyFrame.mvKeys_superpoint[i].octave;
                // Search in a window. Size depends on scale
                float radius =
                    th * CurrentKeyFrame.mvScaleFactors_suerpoint[nLastOctave];
                vector<size_t> vIndices2;

                vIndices2 =
                    CurrentKeyFrame.GetFeaturesInArea_Superpoint(u, v, radius);
                if (vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP_superpoint->GetDescriptor();

                float bestDist = 256;
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                                    vend = vIndices2.end();
                     vit != vend; vit++) {
                    const size_t i2 = *vit;
                    if (CurrentKeyFrame.GetSuperpointMapPoint(i2))
                        if (CurrentKeyFrame.GetSuperpointMapPoint(i2)
                                ->Observations() > 0)
                            continue;
                    const cv::Mat &d =
                        CurrentKeyFrame.mDescriptors_superpoint.row(i2);
                    const float dist = DescriptorDistance(dMP, d);

                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx2 = i2;
                    }
                }

                if (bestDist <= TH_LOW) {
                    CurrentKeyFrame.AddSuperpointMapPoint(
                        pMP_superpoint, bestIdx2);
                    // TODO(zhangye): check this condition
                    pMP_superpoint->AddObservation(
                        &CurrentKeyFrame, bestIdx2, true);
                    pMP_superpoint->ComputeDistinctiveDescriptors(true);
                    pMP_superpoint->UpdateNormalAndDepth(true);

                    nmatches++;

                    if (false) {
                        float rot =
                            LastKeyFrame.mvKeysUn_superpoint[i].angle -
                            CurrentKeyFrame.mvKeysUn_superpoint[bestIdx2].angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    if (false) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        // ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i != ind1 && i != ind2 && i != ind3) {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    CurrentKeyFrame.AddSuperpointMapPoint(
                        static_cast<MapPoint *>(NULL), rotHist[i][j]);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

bool SuperPointMatcher::CheckDistEpipolarLine(
    const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
    const KeyFrame *pKF2) {
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x * F12.at<float>(0, 0) +
                    kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
    const float b = kp1.pt.x * F12.at<float>(0, 1) +
                    kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
    const float c = kp1.pt.x * F12.at<float>(0, 2) +
                    kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

    const float num = a * kp2.pt.x + b * kp2.pt.y + c;
    const float den = a * a + b * b;

    if (den == 0)
        return false;

    const float dsqr = num * num / den;

    return dsqr < 3.84 * pKF2->mvLevelSigma2_superpoint[kp2.octave];
}

float SuperPointMatcher::DescriptorDistance(
    const cv::Mat &a, const cv::Mat &b) {
    float dist = (float)cv::norm(a, b, cv::NORM_L2);
    return dist;
}

} // namespace ORB_SLAM3