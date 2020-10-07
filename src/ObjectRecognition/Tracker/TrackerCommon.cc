//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include "Tracker/TrackerCommon.h"
#include "Utility/Camera.h"
#include "Utility/Utility.h"

namespace ObjRecognition {
namespace ObjTrackerCommon {
void KeyPointsToPoints(
    const std::vector<cv::KeyPoint> &kPts, std::vector<cv::Point2d> &pts) {
    for (int i = 0; i < kPts.size(); i++) {
        pts.emplace_back(kPts[i].pt);
    }
}

void GetCoordsInBorder(
    Eigen::Vector2d &pt, const int &xMin, const int &yMin, const int &xMax,
    const int &yMax) {
    if (pt(0) < xMin)
        pt(0) = xMin;
    if (pt(0) > xMax)
        pt(0) = xMax;
    if (pt(1) < yMin)
        pt(1) = yMin;
    if (pt(1) > yMax)
        pt(1) = yMax;
}
bool InBorder(
    const Eigen::Vector2d &pt, const int &xMin, const int &yMin,
    const int &xMax, const int &yMax) {
    int imgX = static_cast<int>(pt(0));
    int imgY = static_cast<int>(pt(1));
    return xMin <= imgX && imgX < xMax && yMin <= imgY && imgY < yMax;
}
void GetMapPointPositions(
    const std::vector<MapPoint::Ptr> &pointClouds, const Eigen::Matrix3d &Rwo,
    const Eigen::Vector3d &Two, std::vector<Eigen::Vector3d> &mapPointsObj,
    std::vector<Eigen::Vector3d> &mapPointsWorld) {
    mapPointsObj.reserve(pointClouds.size());
    mapPointsWorld.reserve(pointClouds.size());
    int pointCloudNum = pointClouds.size();
    for (int i = 0; i < pointCloudNum; i++) {
        Eigen::Vector3d mapPointPose = pointClouds[i]->GetPose();
        mapPointsObj.emplace_back(mapPointPose);
        Eigen::Vector3d mapPointPoseWorld = Rwo * mapPointPose + Two;
        mapPointsWorld.emplace_back(mapPointPoseWorld);
    }
}
void Project(
    const std::vector<Eigen::Vector3d> &pointCloudsWorld,
    const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &Tcw,
    std::vector<bool> &projectFailState,
    std::vector<Eigen::Vector2d> &projectPoints, bool isBox) {

    const int width = CameraIntrinsic::GetInstance().Width();
    const int height = CameraIntrinsic::GetInstance().Height();
    const Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();

    const int kBorderSize = 1;
    for (int i = 0; i < pointCloudsWorld.size(); i++) {
        projectFailState[i] = false;
        // world coords
        Eigen::Vector3d point3d = pointCloudsWorld[i];
        // world coordinates -> camera coordinates
        point3d = Rcw * point3d + Tcw;
        Eigen::Vector2d point2d;
        point3d = K * point3d;
        point2d(0) = point3d(0) / point3d(2);
        point2d(1) = point3d(1) / point3d(2);
        if (!InBorder(
                point2d, kBorderSize, kBorderSize, width - kBorderSize,
                height - kBorderSize)) {
            projectFailState[i] = true;
            if (!isBox) {
                continue;
            } else {
                GetCoordsInBorder(
                    point2d, kBorderSize, kBorderSize, width - kBorderSize,
                    height - kBorderSize);
                projectPoints.emplace_back(point2d);
            }
        } else {
            projectPoints.emplace_back(point2d);
        }
    }
    VLOG(10) << "PointCloudObjTracker::Project done";
}
void GetFeaturesInArea(
    const Eigen::Vector2d &point, const int &width, const int &height,
    const std::vector<cv::KeyPoint> &keyPoints, std::vector<int> &vIndices) {
    const int kWindowSizeThreshold = 30;
    int nMinX = static_cast<int>(point(0)) - kWindowSizeThreshold;
    nMinX = 0 < nMinX ? nMinX : 0;
    int nMaxX = static_cast<int>(point(0)) + kWindowSizeThreshold;
    nMaxX = width < nMaxX ? width : nMaxX;

    int nMinY = static_cast<int>(point(1)) - kWindowSizeThreshold;
    nMinY = 0 < nMinY ? nMinY : 0;
    int nMaxY = static_cast<int>(point(1)) + kWindowSizeThreshold;
    nMaxY = height < nMaxY ? height : nMaxY;
    for (int k = 0; k < keyPoints.size(); k++) {

        if (InBorder(
                Eigen::Vector2d(keyPoints[k].pt.x, keyPoints[k].pt.y), nMinX,
                nMinY, nMaxX, nMaxY)) {
            vIndices.emplace_back(k);
        }
    }
}

int TrackerHammingDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

int SearchByProjection(
    const std::vector<Eigen::Vector2d> &projectPoints,
    const std::vector<MapPoint::Ptr> &pointClouds,
    const std::vector<bool> &projectFailState,
    const std::vector<cv::KeyPoint> &keyPoints, const cv::Mat &descriptors,
    std::vector<bool> &matchKeyPointsState,
    std::map<int, MapPointIndex> &matches2dTo3d) {
    // STSLAMCommon::Timer timer_projection("Search by projection");

    const int width = CameraIntrinsic::GetInstance().Width();
    const int height = CameraIntrinsic::GetInstance().Height();

    int matches = 0;
    const int kDistThreshold = 80;
    const float kRatioThreshold = 0.95;
    matchKeyPointsState.resize(projectPoints.size());
    for (int i = 0, j = 0; i < pointClouds.size() && j < projectPoints.size();
         i++) {
        if (projectFailState[i]) {
            continue;
        }

        Eigen::Vector2d pointProject = projectPoints[j];
        matchKeyPointsState[j] = false;
        std::vector<int> vIndices;
        vIndices.reserve(keyPoints.size());
        // is this method need??  if the object is moved by a long distance???
        GetFeaturesInArea(pointProject, width, height, keyPoints, vIndices);
        if (vIndices.empty()) {
            j++;
            continue;
        }
        MapPoint::Ptr point = pointClouds[i];
        cv::Mat pointDescriptor = point->GetDescriptor();

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIndex = -1;
        for (int k = 0; k < vIndices.size(); k++) {
            // keypoint id
            int kpi = vIndices[k];
            // if already match to a 3D point, continue
            if (matches2dTo3d.find(kpi) != matches2dTo3d.end()) {
                continue;
            }
            cv::Mat keyPointDescriptor = descriptors.row(kpi);
            int dist =
                TrackerHammingDistance(pointDescriptor, keyPointDescriptor);

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIndex = kpi;
            } else if (dist < bestDist2) {
                bestDist2 = dist;
            }
        }
        MapPointIndex mpIndex = i;
        // VLOG(5) << "Trackerbest: " << bestDist;
        if (bestDist < kDistThreshold) {
            //            if (bestDist < bestDist2 * kRatioThreshold) {
            matches++;
            matchKeyPointsState[j] = true;
            matches2dTo3d.insert(
                std::pair<int, MapPointIndex>(bestIndex, mpIndex));
            //            }
        }
        j++;
    }
    // TODO(Zhangye) :check the orientation????
    // timer_projection.Stop();
    VLOG(20) << ("PointCloudObjTracker::SearchByProjection done");
    return matches;
}

bool SolvePnP(
    const std::map<int, MapPointIndex> &matches2dTo3d,
    const std::vector<cv::Point2d> &keyPoints,
    const std::vector<Eigen::Vector3d> &pointClouds3dObj,
    const Eigen::Matrix3d &initialRco, const Eigen::Vector3d &initialTco,
    const cv::Mat &Kcv, Eigen::Matrix3d &resultRco, Eigen::Vector3d &resultTco,
    std::map<int, MapPointIndex> &matches2dTo3dNew, int &inlierNum) {
    std::vector<cv::Point2f> points2d;
    std::vector<cv::Point3f> points3dObj;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        cv::Point2f point2d = static_cast<cv::Point2f>(keyPoints[iter->first]);
        points2d.emplace_back(point2d);
        cv::Point3f point3fObj = static_cast<cv::Point3f>(cv::Point3d(
            pointClouds3dObj[iter->second](0),
            pointClouds3dObj[iter->second](1),
            pointClouds3dObj[iter->second](2)));
        points3dObj.emplace_back(point3fObj);
    }

    cv::Mat r, rco, tco, D, tmpRco;
    TypeConverter::Eigen2CV(initialRco, tmpRco);
    cv::Rodrigues(tmpRco, rco);
    TypeConverter::Eigen2CV(initialTco, tco);
    bool pnpSuccess = false;
    cv::Mat inliers;
    const int pnpIterationCount = 100;
    const double pnpConfidence = 0.99;
    const float pnpReprojectionError = 5.9;
    pnpSuccess = cv::solvePnPRansac(
        points3dObj, points2d, Kcv, D, rco, tco, true, pnpIterationCount,
        pnpReprojectionError, pnpConfidence, inliers);
    inlierNum = inliers.rows;
    if (pnpSuccess) {

        cv::Rodrigues(rco, r);
        TypeConverter::CV2Eigen(r, resultRco);
        TypeConverter::CV2Eigen(tco, resultTco);

        std::vector<int> inlierIndex;
        for (int i = 0; i < inliers.rows; i++) {
            int index = inliers.at<int>(i, 0);
            inlierIndex.emplace_back(index);
        }

        int index_ = 0;
        int inliersNum = 0;
        for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
             iter++) {
            if (index_ == inlierIndex[inliersNum]) {
                matches2dTo3dNew.insert(
                    std::pair<int, MapPointIndex>(iter->first, iter->second));
                inliersNum++;
            }
            if (inliersNum == inliers.rows)
                break;
            index_++;
        }

        // compute project error
        double projectError = 0.0;
        Eigen::Matrix3d K = CameraIntrinsic::GetInstance().GetEigenK();
        for (int i = 0; i < inliers.rows; i++) {
            int index = inliers.at<int>(i, 0);
            cv::Point3f inlier3d = points3dObj[index];
            cv::Point2f inlier2d = points2d[index];
            Eigen::Vector3d inlierProjectedMat =
                Eigen::Vector3d(inlier3d.x, inlier3d.y, inlier3d.z);
            Eigen::Vector3d inlierProjectedMat3d =
                K * (resultRco * inlierProjectedMat + resultTco);
            Eigen::Vector3d inlierProjectedMat2d;
            inlierProjectedMat2d(0) =
                inlierProjectedMat3d(0) / inlierProjectedMat3d(2);
            inlierProjectedMat2d(1) =
                inlierProjectedMat3d(1) / inlierProjectedMat3d(2);
            cv::Point2f inlierProjected =
                cv::Point2f(inlierProjectedMat2d(0), inlierProjectedMat2d(1));
            projectError += sqrt(
                (inlierProjected.x - inlier2d.x) *
                    (inlierProjected.x - inlier2d.x) +
                (inlierProjected.y - inlier2d.y) *
                    (inlierProjected.y - inlier2d.y));
        }
        projectError /= inliers.rows;
        VLOG(5) << "pnp projectedError:::" << projectError;
        if (projectError > pnpReprojectionError)
            pnpSuccess = false;
        //        if(projectError > 5.0)
        //            LOG(FATAL) << "something go wrong";
        VLOG(5) << "pnp inliers: " << inliers.rows;
    }
    VLOG(5) << "pnp result: R" << resultRco;
    VLOG(5) << "pnp result: t" << resultTco;
    return pnpSuccess;
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

void ExtractKeyPointsAndDes(
    const std::shared_ptr<ObjRecognition::FrameData> &frm,
    std::vector<cv::KeyPoint> &imgKeyPoints, cv::Mat &imgDescriptor) {
    cv::Mat img = frm->img.clone();
    imgKeyPoints = frm->mKpts;
    imgDescriptor = frm->mDesp;
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
} // namespace ObjTrackerCommon
} // namespace ObjRecognition