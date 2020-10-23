//
// Created by root on 2020/10/21.
//

#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "ORBSLAM3/SPextractor.h"
#include <glog/logging.h>
#include <chrono>
using namespace std::chrono;
void FindMatchByKNN(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches) {
    // STSLAMCommon::Timer detectionFindMatch("detection find match by KNN");
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    // use L2 norm instead of Hamming distance
    cv::BFMatcher matcher(cv::NormTypes::NORM_L2);
    matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);

    VLOG(5) << "KNN Matches size: " << knnMatches.size();

    for (size_t i = 0; i < knnMatches.size(); i++) {
        cv::DMatch &bestMatch = knnMatches[i][0];
        cv::DMatch &betterMatch = knnMatches[i][1];
        const float distanceRatio = bestMatch.distance / betterMatch.distance;
        VLOG(50) << "distanceRatio = " << distanceRatio;
        // the farest distance, the better result
        const float kMinDistanceRatioThreshld = 0.85;
        if (distanceRatio < kMinDistanceRatioThreshld) {
            matches.push_back(bestMatch);
        }
    }

    VLOG(15) << "after distance Ratio matches size: " << matches.size();

    double minDisKnn = 9999.0;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance < minDisKnn) {
            minDisKnn = matches[i].distance;
        }
    }
    VLOG(15) << "minDisKnn = " << minDisKnn;

    // set good_matches_threshold
    const int kgoodMatchesThreshold = 200;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= kgoodMatchesThreshold) {
            goodMatches.push_back(matches[i]);
        }
    }
    // VLOG(10) << "detection find match by KNN time: "
    //       << detectionFindMatch.Stop();
}

int main(int argc, char *argv[]) {
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    std::string path1 =
        "/home/zhangye/data1/superpoint/datasets/Box/1_box/1.png";
    std::string path2;
    for (size_t i = 2; i < 101; i++) {
        path2 = "/home/zhangye/data1/superpoint/datasets/Box/1_box/" +
                std::to_string(i) + ".png";
        cv::Mat img1 = cv::imread(path1, CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat img2 = cv::imread(path2, CV_LOAD_IMAGE_UNCHANGED);
        if (img1.channels() == 3) {
            cvtColor(img1, img1, CV_RGB2GRAY);
        }
        if (img2.channels() == 3) {
            cvtColor(img2, img2, CV_RGB2GRAY);
        }

        ORB_SLAM3::SPextractor *SPextractor =
            new ORB_SLAM3::SPextractor(1000, 1.2, 3, 0.015, 0.007, true);

        std::vector<cv::KeyPoint> keypoints1;
        cv::Mat descriptor1;
        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptor2;
        auto start = high_resolution_clock::now();
        (*SPextractor)(img1, cv::Mat(), keypoints1, descriptor1);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        VLOG(0) << "for single image: " << duration.count() / 1000.0 << " ms"
                << std::endl;
        (*SPextractor)(img2, cv::Mat(), keypoints2, descriptor2);
        VLOG(0) << "keypoints1 size: " << keypoints1.size();
        VLOG(0) << "keypoints2 size: " << keypoints2.size();
        if (!keypoints1.empty() && !keypoints2.empty()) {
            std::vector<cv::DMatch> dmatches;
            FindMatchByKNN(descriptor1, descriptor2, dmatches);
            cv::Mat imshow;
            cv::drawMatches(
                img1, keypoints1, img2, keypoints2, dmatches, imshow);
            cv::imshow("match", imshow);
            cv::imwrite(
                "/home/zhangye/data1/test" + std::to_string(i) + ".png",
                imshow);
        } else {
            VLOG(0) << "no keypoints in image";
        }
    }
    cv::waitKey(10);
    return 0;
}