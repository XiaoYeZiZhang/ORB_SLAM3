//
// Created by root on 2020/10/26.
//

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
    SuperPointMatcher(float nnratio = 0.6, bool checkOri = true);

    // Computes the Hamming distance between two ORB descriptors
    static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Project MapPoints tracked in last frame into the current frame and search
    // matches. Used to track from previous frame (Tracking)
    static int SearchByProjection(
        KeyFrame &CurrentKeyFrame, KeyFrame &LastKeyFrame, const float th,
        const bool bMono);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node
    // (at a certain level) Used in Relocalisation and Loop Detection
    int SearchByBoW(
        KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
    int SearchByBoW(
        KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(
        Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched,
        std::vector<int> &vnMatches12, int windowSize = 10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(
        KeyFrame *pKF1, KeyFrame *pKF2, const cv::Mat &F12,
        std::vector<pair<size_t, size_t>> &vMatchedPairs);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a
    // Sim3 [s12*R12|t12] In the stereo and RGB-D case, s12=1
    int SearchBySim3(
        KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12,
        const float &s12, const cv::Mat &R12, const cv::Mat &t12,
        const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(
        KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints,
        const float th = 3.0, const bool bRight = false);

    // Project MapPoints into KeyFrame using a given Sim3 and search for
    // duplicated MapPoints.
    int Fuse(
        KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints,
        float th, vector<MapPoint *> &vpReplacePoint);

public:
    static const float TH_LOW;
    static const float TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    bool CheckDistEpipolarLine(
        const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
        const KeyFrame *pKF);
    //    bool CheckDistEpipolarLine(
    //        const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat
    //        &F12, const KeyFrame *pKF, const bool b1 = false);
    //    bool CheckDistEpipolarLine2(
    //        const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat
    //        &F12, const KeyFrame *pKF, const float unc);

    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(
        std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};
} // namespace ORB_SLAM3
#endif // ORB_SLAM3_SUPERPOINTMATCHER_H
