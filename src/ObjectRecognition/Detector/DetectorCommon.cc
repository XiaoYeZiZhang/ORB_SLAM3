#include <glog/logging.h>
#include <Eigen/Dense>
#include <cxcore.hpp>
#include <cv.hpp>
#include "StatisticsResult/Statistics.h"
#include "StatisticsResult/Timer.h"
#include "Detector/DetectorCommon.h"

namespace ObjRecognition {
namespace ObjDetectionCommon {
cv::Mat GetPointCloudDesp(const std::shared_ptr<Object> &pc) {
    cv::Mat result;
    if (!pc) {
        LOG(ERROR) << "GetPointCloudDesp: object is null";
        return result;
    }

    PointModel pm = pc->GetPointClouds();
    if (pm.empty()) {
        LOG(ERROR) << "GetPointCloudDesp: pointCloud is empty";
        return result;
    }

    cv::Mat ptDesp = pm[0]->GetDescriptor();
    for (size_t i = 1; i < pm.size(); i++) {
        // combine the desp of mapPoint
        cv::hconcat(ptDesp, pm[i]->GetDescriptor(), ptDesp);
    }
    return ptDesp.t();
}

cv::Mat GetPointCloudDespByConnection(
    const std::vector<MapPoint::Ptr> &associated_mappoints) {
    cv::Mat result;
    if (associated_mappoints.empty()) {
        LOG(ERROR) << "GetPointCloudDesp: object is null";
        return result;
    }

    cv::Mat ptDesp = associated_mappoints[0]->GetDescriptor();
    for (size_t i = 1; i < associated_mappoints.size(); i++) {
        cv::hconcat(ptDesp, associated_mappoints[i]->GetDescriptor(), ptDesp);
    }
    return ptDesp.t();
}

void GetBoxPoint(
    const std::shared_ptr<Object> &mObj,
    std::vector<Eigen::Vector3d> &pointBoxs) {
    std::vector<Eigen::Vector3d> pointsCloud;
    double xMin = INT_MAX, xMax = INT_MIN;
    double yMin = INT_MAX, yMax = INT_MIN;
    double zMin = INT_MAX, zMax = INT_MIN;
    Eigen::Vector3d temp;
    std::vector<MapPoint::Ptr> allMPs = mObj->GetPointClouds();
    for (int i = 0; i < allMPs.size(); i++) {
        temp = allMPs[i]->GetPose();
        if (temp.x() < xMin) {
            xMin = temp.x();
        }
        if (temp.x() > xMax) {
            xMax = temp.x();
        }
        if (temp.y() < yMin) {
            yMin = temp.y();
        }
        if (temp.y() > yMax) {
            yMax = temp.y();
        }
        if (temp.z() < zMin) {
            zMin = temp.z();
        }
        if (temp.z() > zMax) {
            zMax = temp.z();
        }
    }

    pointBoxs[0] = Eigen::Vector3d(xMin, yMin, zMin);
    pointBoxs[1] = Eigen::Vector3d(xMax, yMin, zMin);
    pointBoxs[2] = Eigen::Vector3d(xMin, yMin, zMax);
    pointBoxs[3] = Eigen::Vector3d(xMax, yMin, zMax);

    pointBoxs[4] = Eigen::Vector3d(xMin, yMax, zMin);
    pointBoxs[5] = Eigen::Vector3d(xMax, yMax, zMin);
    pointBoxs[6] = Eigen::Vector3d(xMin, yMax, zMax);
    pointBoxs[7] = Eigen::Vector3d(xMax, yMax, zMax);
}

void FindMatchByKNN_Homography(
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &frmDesp,
    const cv::Mat &pcDesp, std::vector<cv::DMatch> &goodMatches,
    const float ratio_threshold) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    cv::BFMatcher matcher(cv::NormTypes::NORM_HAMMING);
    matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);

    VLOG(5) << "KNN Matches size: " << knnMatches.size();
    for (size_t i = 0; i < knnMatches.size(); i++) {
        cv::DMatch &bestMatch = knnMatches[i][0];
        cv::DMatch &betterMatch = knnMatches[i][1];
        const float distanceRatio = bestMatch.distance / betterMatch.distance;
        VLOG(50) << "distanceRatio = " << distanceRatio;
        if (distanceRatio < ratio_threshold) {
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
        } else {
            VLOG(0) << "detector best dist bigger than 200";
        }
    }
    std::vector<cv::Point2f> srcPoints(goodMatches.size());
    std::vector<cv::Point2f> dstPoints(goodMatches.size());

    for (size_t i = 0; i < goodMatches.size(); i++) {
        srcPoints[i] = keypoints2[goodMatches[i].trainIdx].pt;
        dstPoints[i] = keypoints1[goodMatches[i].queryIdx].pt;
    }

    if (!srcPoints.empty()) {
        std::vector<uchar> inliersMask(srcPoints.size());
        auto homography = findHomography(
            srcPoints, dstPoints, CV_FM_RANSAC, 6.0, inliersMask);

        std::vector<cv::DMatch> inliers;
        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i])
                inliers.push_back(matches[i]);
        }
        goodMatches.swap(inliers);
    }
}

void FindMatchByKNN(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches, const float ratio_threshold) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    cv::BFMatcher matcher(cv::NormTypes::NORM_HAMMING);
    matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);

    VLOG(5) << "KNN Matches size: " << knnMatches.size();

    for (size_t i = 0; i < knnMatches.size(); i++) {
        cv::DMatch &bestMatch = knnMatches[i][0];
        cv::DMatch &betterMatch = knnMatches[i][1];
        const float distanceRatio = bestMatch.distance / betterMatch.distance;
        VLOG(50) << "distanceRatio = " << distanceRatio;
        // the farest distance, the better result
        if (distanceRatio < ratio_threshold) {
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
        } else {
            VLOG(0) << "detector 2d-3d best match bigger than 200";
        }
    }
}

void FindMatchByKNN_SuperPoint(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knnMatches;
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

    for (size_t i = 0; i < matches.size(); i++) {
        goodMatches.push_back(matches[i]);
    }
}

void FindMatchByKNN_SuperPoint_Homography(
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2, const cv::Mat &frmDesp,
    const cv::Mat &pcDesp, std::vector<cv::DMatch> &goodMatches) {
    TIMER_UTILITY::Timer timer;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    cv::BFMatcher matcher(cv::NormTypes::NORM_L2, true);
    // matcher.knnMatch(frmDesp, pcDesp, knnMatches, 2);
    matcher.match(frmDesp, pcDesp, goodMatches);

    // Prepare data for findHomography
    std::vector<cv::Point2f> srcPoints(goodMatches.size());
    std::vector<cv::Point2f> dstPoints(goodMatches.size());

    for (size_t i = 0; i < goodMatches.size(); i++) {
        srcPoints[i] = keypoints2[goodMatches[i].trainIdx].pt;
        dstPoints[i] = keypoints1[goodMatches[i].queryIdx].pt;
    }
    TIMER_UTILITY::Timer timer1;
    std::vector<uchar> inliersMask(srcPoints.size());
    TIMER_UTILITY::Timer timer_getmatch;
    auto homography =
        findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 6.0, inliersMask);
    STATISTICS_UTILITY::StatsCollector detector_2d_match(
        "Time: detector find 2d match by homography");
    detector_2d_match.AddSample(timer_getmatch.Stop());
    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i])
            inliers.push_back(goodMatches[i]);
    }
    goodMatches.swap(inliers);
}

void GetPointCloudBoundingBox(
    const std::shared_ptr<Object> &obj,
    std::vector<Eigen::Vector3d> &mapPointBoundingBox) {
    const std::vector<MapPoint::Ptr> pointClouds = obj->GetPointClouds();
    double xmin = INT_MAX;
    double ymin = INT_MAX;
    double zmin = INT_MAX;
    double xmax = INT_MIN;
    double ymax = INT_MIN;
    double zmax = INT_MIN;
    for (int i = 0; i < pointClouds.size(); i++) {
        Eigen::Vector3d mapPointPose = pointClouds[i]->GetPose();
        if (xmin > mapPointPose(0))
            xmin = mapPointPose(0);
        if (ymin > mapPointPose(1))
            ymin = mapPointPose(1);
        if (zmin > mapPointPose(2))
            zmin = mapPointPose(2);
        if (xmax < mapPointPose(0))
            xmax = mapPointPose(0);
        if (ymax < mapPointPose(1))
            ymax = mapPointPose(1);
        if (zmax < mapPointPose(2))
            zmax = mapPointPose(2);
    }
    Eigen::Vector3d corner0 = Eigen::Vector3d(xmin, ymin, zmin);
    Eigen::Vector3d corner1 = Eigen::Vector3d(xmax, ymin, zmin);
    Eigen::Vector3d corner2 = Eigen::Vector3d(xmax, ymax, zmin);
    Eigen::Vector3d corner3 = Eigen::Vector3d(xmin, ymax, zmin);
    Eigen::Vector3d corner4 = Eigen::Vector3d(xmin, ymin, zmax);
    Eigen::Vector3d corner5 = Eigen::Vector3d(xmax, ymin, zmax);
    Eigen::Vector3d corner6 = Eigen::Vector3d(xmax, ymax, zmax);
    Eigen::Vector3d corner7 = Eigen::Vector3d(xmin, ymax, zmax);
    mapPointBoundingBox.emplace_back(corner0);
    mapPointBoundingBox.emplace_back(corner1);
    mapPointBoundingBox.emplace_back(corner2);
    mapPointBoundingBox.emplace_back(corner3);
    mapPointBoundingBox.emplace_back(corner4);
    mapPointBoundingBox.emplace_back(corner5);
    mapPointBoundingBox.emplace_back(corner6);
    mapPointBoundingBox.emplace_back(corner7);
}

void DrawBoundingBox(
    const cv::Mat &showResult, std::vector<cv::Point2d> &boxProjResult,
    cv::Scalar &color) {

    cv::line(showResult, boxProjResult[0], boxProjResult[1], color);
    cv::line(showResult, boxProjResult[1], boxProjResult[2], color);
    cv::line(showResult, boxProjResult[2], boxProjResult[3], color);
    cv::line(showResult, boxProjResult[3], boxProjResult[0], color);
    cv::line(showResult, boxProjResult[4], boxProjResult[5], color);
    cv::line(showResult, boxProjResult[5], boxProjResult[6], color);
    cv::line(showResult, boxProjResult[6], boxProjResult[7], color);
    cv::line(showResult, boxProjResult[7], boxProjResult[4], color);
    cv::line(showResult, boxProjResult[0], boxProjResult[4], color);
    cv::line(showResult, boxProjResult[1], boxProjResult[5], color);
    cv::line(showResult, boxProjResult[2], boxProjResult[6], color);
    cv::line(showResult, boxProjResult[3], boxProjResult[7], color);
}
} // namespace ObjDetectionCommon
} // namespace ObjRecognition