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
    const std::vector<cv::KeyPoint> keypoints1,
    const std::vector<cv::KeyPoint> keypoints2, const cv::Mat &frmDesp,
    const cv::Mat &pcDesp, std::vector<cv::DMatch> &goodMatches) {
    // STSLAMCommon::Timer detectionFindMatch("detection find match by KNN");
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    // use L2 norm instead of Hamming distance
    cv::BFMatcher matcher(cv::NormTypes::NORM_L2, true);
    // matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);
    matcher.match(frmDesp, pcDesp, matches);
    //    VLOG(5) << "KNN Matches size: " << knnMatches.size();
    //
    //    for (size_t i = 0; i < knnMatches.size(); i++) {
    //        cv::DMatch &bestMatch = knnMatches[i][0];
    //        cv::DMatch &betterMatch = knnMatches[i][1];
    //        const float distanceRatio = bestMatch.distance /
    //        betterMatch.distance; VLOG(50) << "distanceRatio = " <<
    //        distanceRatio;
    //        // the farest distance, the better result
    //        const float kMinDistanceRatioThreshld = 0.95;
    //        if (distanceRatio < kMinDistanceRatioThreshld) {
    //            matches.push_back(bestMatch);
    //        }
    //    }

    VLOG(15) << "after distance Ratio matches size: " << matches.size();
    //
    //    double minDisKnn = 9999.0;
    //    for (size_t i = 0; i < matches.size(); i++) {
    //        if (matches[i].distance < minDisKnn) {
    //            minDisKnn = matches[i].distance;
    //        }
    //    }
    //    VLOG(15) << "minDisKnn = " << minDisKnn;

    const int kgoodMatchesThreshold = 200;
    for (size_t i = 0; i < matches.size(); i++) {
        //        if (matches[i].distance <= (2*minDisKnn > 30 ? 2*minDisKnn :
        //        30)) {
        goodMatches.push_back(matches[i]);
        //        }
    }

    // Prepare data for findHomography
    std::vector<cv::Point2f> srcPoints(goodMatches.size());
    std::vector<cv::Point2f> dstPoints(goodMatches.size());

    for (size_t i = 0; i < goodMatches.size(); i++) {
        srcPoints[i] = keypoints2[goodMatches[i].trainIdx].pt;
        dstPoints[i] = keypoints1[goodMatches[i].queryIdx].pt;
    }

    std::vector<uchar> inliersMask(srcPoints.size());
    auto homography =
        findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 4.5, inliersMask);

    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    goodMatches.swap(inliers);
}

void FindMatchByKNN_opencv(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches) {
    // STSLAMCommon::Timer detectionFindMatch("detection find match by KNN");
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    // use L2 norm instead of Hamming distance
    cv::BFMatcher matcher(cv::NormTypes::NORM_HAMMING);
    matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);

    VLOG(5) << "KNN Matches size: " << knnMatches.size();

    for (size_t i = 0; i < knnMatches.size(); i++) {
        cv::DMatch &bestMatch = knnMatches[i][0];
        cv::DMatch &betterMatch = knnMatches[i][1];
        const float distanceRatio = bestMatch.distance / betterMatch.distance;
        VLOG(50) << "distanceRatio = " << distanceRatio;
        // the farest distance, the better result
        const float kMinDistanceRatioThreshld = 0.80;
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
    ORB_SLAM3::SPextractor *SPextractor =
        new ORB_SLAM3::SPextractor(2000, 1.2, 1, 0.015, 0.007, true);
    cv::Ptr<cv::ORB> m_orb_detector = cv::ORB::create(1000, 1.2);
    m_orb_detector->setScoreType(cv::ORB::FAST_SCORE);
    m_orb_detector->setFastThreshold(7);
    std::string path1 =
        "/home/zhangye/data1/superpoint/datasets/Box/1_box/1.png";
    std::string path2;
    cv::Mat img1 = cv::imread(path1, CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptor1;
    std::vector<cv::KeyPoint> keypoints1_opencv;
    cv::Mat descriptor1_opencv;

    if (img1.channels() == 3) {
        cvtColor(img1, img1, CV_RGB2GRAY);
    }

    cv::Mat mask = cv::Mat();
    // cv::Mat mask = img1.clone();
    // mask = cv::Scalar::all(255);
    // for (size_t i = 0; i < 50; i++) {
    // for (size_t j = 0; j < img1.cols; j++) {
    // mask.at<float>(i, j) = 0;
    // }
    // }

    (*SPextractor)(img1, mask, keypoints1, descriptor1);
    m_orb_detector->detectAndCompute(
        img1, cv::Mat(), keypoints1_opencv, descriptor1_opencv);

    for (size_t i = 2; i < 55; i++) {
        VLOG(0) << "image id: " << i << std::endl;
        path2 = "/home/zhangye/data1/superpoint/datasets/Box/1_box/" +
                std::to_string(i) + ".png";
        cv::Mat img2 = cv::imread(path2, CV_LOAD_IMAGE_UNCHANGED);
        if (img2.channels() == 3) {
            cvtColor(img2, img2, CV_RGB2GRAY);
        }

        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptor2;
        std::vector<cv::KeyPoint> keypoints2_opencv;
        cv::Mat descriptor2_opencv;

        auto start = high_resolution_clock::now();
        (*SPextractor)(img2, mask, keypoints2, descriptor2);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        VLOG(0) << "for single image: " << duration.count() / 1000.0 << " ms"
                << std::endl;

        VLOG(0) << "keypoints1 size: " << keypoints1.size();
        VLOG(0) << "keypoints2 size: " << keypoints2.size();
        m_orb_detector->detectAndCompute(
            img2, cv::Mat(), keypoints2_opencv, descriptor2_opencv);

        if (!keypoints1.empty() && !keypoints2.empty()) {
            std::vector<cv::DMatch> dmatches;
            FindMatchByKNN(
                keypoints1, keypoints2, descriptor1, descriptor2, dmatches);
            if (dmatches.size() == 0) {
                VLOG(0) << "no keypoint match!";
            }
            cv::Mat imshow;
            cv::drawMatches(
                img1, keypoints1, img2, keypoints2, dmatches, imshow);
            cv::imshow("match", imshow);
            cv::imwrite(
                "/home/zhangye/data1/test_superpoint/test" + std::to_string(i) +
                    ".png",
                imshow);

            std::vector<cv::DMatch> dmatches_opencv;
            FindMatchByKNN_opencv(
                descriptor1_opencv, descriptor2_opencv, dmatches_opencv);
            cv::drawMatches(
                img1, keypoints1_opencv, img2, keypoints2_opencv,
                dmatches_opencv, imshow);
            cv::imshow("match_opencv", imshow);
            cv::imwrite(
                "/home/zhangye/data1/test_superpoint/test" + std::to_string(i) +
                    "_cv.png",
                imshow);

        } else {
            VLOG(0) << "no keypoints in image";
        }
    }
    cv::waitKey(10);
    return 0;
}