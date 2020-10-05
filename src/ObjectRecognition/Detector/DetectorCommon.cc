//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include <Eigen/Dense>
#include <cxcore.hpp>
#include <cv.hpp>
#include "Utility/Camera.h"
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

void GetBoxPoint(
    const std::shared_ptr<Object> &mObj,
    std::vector<Eigen::Vector3d> &pointBoxs) {
    std::vector<Eigen::Vector3d> pointsCloud;
    double xMin = 99999, xMax = -99999;
    double yMin = 99999, yMax = -99999;
    double zMin = 99999, zMax = -99999;
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
    pointBoxs[1] = Eigen::Vector3d(xMin, yMax, zMin);
    pointBoxs[2] = Eigen::Vector3d(xMin, yMin, zMax);
    pointBoxs[3] = Eigen::Vector3d(xMin, yMax, zMax);

    pointBoxs[4] = Eigen::Vector3d(xMax, yMin, zMin);
    pointBoxs[5] = Eigen::Vector3d(xMax, yMax, zMin);
    pointBoxs[6] = Eigen::Vector3d(xMax, yMin, zMax);
    pointBoxs[7] = Eigen::Vector3d(xMax, yMax, zMax);
}

void FindMatchByKNN(
    const cv::Mat &frmDesp, const cv::Mat &pcDesp,
    std::vector<cv::DMatch> &goodMatches) {
    // STSLAMCommon::Timer detectionFindMatch("detection find match by KNN");
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
        const float kMinDistanceRatioThreshld = 0.75;
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

std::vector<cv::Mat> ToDescriptorVector(const cv::Mat &Descriptors) {
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j = 0; j < Descriptors.rows; j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

/*void FindMatchByBow(
    const cv::Mat &pcDesp, const cv::Mat &frmDesp, DBoW3::Vocabulary *&voc,
    std::map<int, MapPointIndex> &matches2dTo3d) {
    // 得到当前帧的词袋
    std::vector<cv::Mat> vCurrentDesc = ToDescriptorVector(frmDesp);
    DBoW3::BowVector frameBowVec;
    DBoW3::FeatureVector frameFeatVec;
    voc->transform(vCurrentDesc, frameBowVec, frameFeatVec, 5);

    // 得到地图点的词袋
    std::vector<cv::Mat> mapPointDesc = ToDescriptorVector(pcDesp);
    DBoW3::BowVector mapPointBowVec;
    DBoW3::FeatureVector mapPointFeatVec;
    voc->transform(mapPointDesc, mapPointBowVec, mapPointFeatVec, 5);

    auto f1it = frameFeatVec.begin();
    auto f2it = mapPointFeatVec.begin();
    auto f1end = frameFeatVec.end();
    auto f2end = mapPointFeatVec.end();

    const double kDesDistanceThreshold = 100;
    const double kRatioThreshold = 0.65;
    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                size_t idx1 = f1it->second[i1];
                cv::Mat d1;
                if (matches2dTo3d.find(idx1) != matches2dTo3d.end()) {
                    continue;
                }
                d1 = frmDesp.row(idx1);

                int bestDist1 = INT_MAX;
                int bestIdx2 = -1;
                int bestDist2 = INT_MAX;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2;
                     i2++) {
                    size_t idx2 = f2it->second[i2];
                    cv::Mat d2 = pcDesp.row(idx2);

                    int dist = STSLAMCommon::DescriptorDistance(d1, d2);

                    if (dist < bestDist1) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    } else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }

                VLOG(5) << "bowdist: " << bestDist1 << " " << bestDist2;
                if (bestDist1 < kDesDistanceThreshold) {
                    if (static_cast<float>(bestDist1) <
                        kRatioThreshold * static_cast<float>(bestDist2)) {
                        matches2dTo3d.insert(
                            std::pair<int, MapPointIndex>(idx1, bestIdx2));
                    }
                }
            }
            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = frameFeatVec.lower_bound(f2it->first);
        } else {
            f2it = mapPointFeatVec.lower_bound(f1it->first);
        }
    }
}
*/

Eigen::Isometry3f
GetTMatrix(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    Eigen::Isometry3f T;
    T.rotate(R.cast<float>());
    T.pretranslate(t.cast<float>());
    return T;
}

void DrawBox(
    cv::Mat &imgRGB, const Eigen::Isometry3f &T,
    const std::vector<Eigen::Vector3d> &pointBoxs) {

    const cv::Mat &cameraMatrix = CameraIntrinsic::GetInstance().GetCVK();
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    Eigen::Vector3f tempCameraBox;
    tempCameraBox = (T * pointBoxs[0].cast<float>());
    cv::Point2f point2fXiyizi(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);
    tempCameraBox = (T * pointBoxs[1].cast<float>());
    cv::Point2f point2fXiyxzi(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);
    tempCameraBox = (T * pointBoxs[2].cast<float>());
    cv::Point2f point2fXiyizx(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);
    tempCameraBox = (T * pointBoxs[3].cast<float>());
    cv::Point2f point2fXiyxzx(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);

    tempCameraBox = (T * pointBoxs[4].cast<float>());
    cv::Point2f point2fXxyizi(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);
    tempCameraBox = (T * pointBoxs[5].cast<float>());
    cv::Point2f point2fXxyxzi(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);
    tempCameraBox = (T * pointBoxs[6].cast<float>());
    cv::Point2f point2fXxyizx(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);
    tempCameraBox = (T * pointBoxs[7].cast<float>());
    cv::Point2f point2fXxyxzx(
        fx * tempCameraBox.x() / tempCameraBox.z() + cx,
        fy * tempCameraBox.y() / tempCameraBox.z() + cy);

    CvScalar color = CV_RGB(255, 255, 0);
    cv::line(imgRGB, point2fXiyizi, point2fXiyxzi, color);
    cv::line(imgRGB, point2fXiyizi, point2fXiyizx, color);
    cv::line(imgRGB, point2fXiyizi, point2fXxyizi, color);
    cv::line(imgRGB, point2fXiyxzi, point2fXiyxzx, color);

    cv::line(imgRGB, point2fXiyxzi, point2fXxyxzi, color);
    cv::line(imgRGB, point2fXiyizx, point2fXiyxzx, color);
    cv::line(imgRGB, point2fXiyizx, point2fXxyizx, color);
    cv::line(imgRGB, point2fXiyxzx, point2fXxyxzx, color);

    cv::line(imgRGB, point2fXxyizi, point2fXxyxzi, color);
    cv::line(imgRGB, point2fXxyizi, point2fXxyizx, color);
    cv::line(imgRGB, point2fXxyxzi, point2fXxyxzx, color);
    cv::line(imgRGB, point2fXxyizx, point2fXxyxzx, color);

    drawMarker(imgRGB, point2fXiyizi, cvScalar(255, 255, 0));
    drawMarker(imgRGB, point2fXiyxzi, cvScalar(255, 255, 0));
    drawMarker(imgRGB, point2fXiyizx, cvScalar(255, 255, 0));
    drawMarker(imgRGB, point2fXiyxzx, cvScalar(255, 255, 0));

    drawMarker(imgRGB, point2fXxyizi, cvScalar(255, 255, 0));
    drawMarker(imgRGB, point2fXxyxzi, cvScalar(255, 255, 0));
    drawMarker(imgRGB, point2fXxyizx, cvScalar(255, 255, 0));
    drawMarker(imgRGB, point2fXxyxzx, cvScalar(255, 255, 0));
}

void ShowDetectResult(
    const std::shared_ptr<ObjRecognition::FrameData> &frm,
    const std::shared_ptr<Object> &mObj, const Eigen::Isometry3f &T,
    const ObjRecogState &detectState,
    const std::map<int, MapPointIndex> &matches2dTo3d) {
#ifdef MOBILE_PLATFORM
    return;
#endif
    cv::Mat imgRGB = frm->img.clone();
    const std::vector<MapPoint::Ptr> pointClouds = mObj->GetPointClouds();
    const std::vector<cv::KeyPoint> keyPoints = frm->mKpts;
    std::vector<Eigen::Vector3d> pointBoxs; // the 8 point of box
    pointBoxs.resize(8);

    const cv::Mat &cameraMatrix = CameraIntrinsic::GetInstance().GetCVK();
    std::vector<cv::Point3f> point3f;
    std::vector<cv::Point2f> point2f;
    std::vector<cv::Point2f> imagePoint2f;
    cv::Point2f tempPoint2f;
    Eigen::Vector3f tempPoint3f;
    Eigen::Vector3f tempCameraPoint3f;

    if (detectState == DetectionGood) {
        ObjDetectionCommon::GetBoxPoint(mObj, pointBoxs);
        DrawBox(imgRGB, T, pointBoxs);
    }
    std::vector<cv::KeyPoint> keyPointsShow;
    std::vector<MapPointIndex> mapPointId;

    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        keyPointsShow.emplace_back(keyPoints[iter->first]);
        mapPointId.emplace_back(iter->second);
    }
    drawKeypoints(imgRGB, keyPointsShow, imgRGB, cv::Scalar(0, 0, 255));
    // GlobalPointCloudMatchViewer::SetMatchedMapPoint(mapPointId);
    // GlobalOcvViewer::UpdateView("ObjDetectorResult", imgRGB);
}

void GetMaskKeypointAndDesp(
    const cv::Mat &image,
    const std::shared_ptr<ObjRecognition::FrameData> &frm) {
    // select frame 52 to test
    if (frm->mFrmIndex == 52) {
        cv::Rect rect(192, 29, 123, 247);
        cv::Mat imageShow;
        cv::Mat imageSrc = image.clone();
        cv::Mat imageRoi = imageSrc(rect);
        cv::Mat mask = cv::Mat::zeros(imageSrc.size(), CV_8UC1);
        mask(rect).setTo(255);
        image.copyTo(imageShow, mask);
        // cv::imshow("imageShow", imageShow);
        // cv::waitKey(1);

        // extract keypoints
        cv::Ptr<cv::ORB> orb = cv::ORB::create(
            2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        orb->detect(imageShow, frm->mKpts);
        // compute descriptors
        orb->compute(imageShow, frm->mKpts, frm->mDesp);
    } else {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(
            2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        orb->detect(image, frm->mKpts);
        // compute descriptors
        orb->compute(image, frm->mKpts, frm->mDesp);
    }
}
} // namespace ObjDetectionCommon
} // namespace ObjRecognition