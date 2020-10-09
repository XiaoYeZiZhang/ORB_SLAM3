/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef VIEWER_H
#define VIEWER_H

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"
#include <mutex>
#include <utility>
#include "ScannerStruct/Struct.h"
#include "Struct/PointCloudObject.h"
namespace ORB_SLAM3 {

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class Viewer {
public:
    Viewer(
        System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,
        Tracking *pTracking, const string &strSettingPath);

    // Main thread function. Draw points, keyframes, the current camera pose and
    // the last processed frame. Drawing is refreshed according to the camera
    // fps. We use Pangolin.
    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    bool isStepByStep();

    void Release();

    void SetTrackingPause();
    void SetPointCloudModel(
        std::shared_ptr<ObjRecognition::Object> &pointCloud_model) {
        m_pointCloud_model = pointCloud_model;
    }
    bool both;
    void SetObjectRecognitionPose(Eigen::Matrix3d Row, Eigen::Vector3d tow);

    // draw another window for objRecognition
    void SwitchWindow();
    void Draw();
    void SetFrameAndState(const cv::Mat &img, const int &state);
    void GetFrameAndState(cv::Mat &img, int &state);

    void DrawSLAMInit();
    void DrawObjRecognitionInit();
    void DrawBoundingboxInImage(const vector<Eigen::Vector3d> &boundingbox);
    void
    DrawPointCloudInImage(const std::vector<Eigen::Vector3d> &pointcloud_pos);

private:
    bool ParseViewerParamFile(cv::FileStorage &fSettings);

    bool Stop();

    System *mpSystem;
    FrameDrawer *mpFrameDrawer;
    MapDrawer *mpMapDrawer;
    Tracking *mpTracker;

    // 1/fps in ms
    double mT;
    float mImageWidth, mImageHeight;

    float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

    bool mbStopTrack;

    // objectRecognition
    std::shared_ptr<ObjRecognition::Object> m_pointCloud_model;
    Eigen::Matrix<double, 3, 3> m_Row = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 1> m_tow = Eigen::Matrix<double, 3, 1>::Zero();
    std::vector<cv::Mat> m_trajectory;

    // draw another window for objRecognition
    bool switch_window_flag;
    pangolin::OpenGlRenderState s_cam_slam;
    pangolin::View d_cam_slam;
    pangolin::OpenGlRenderState s_cam_objRecognition;
    pangolin::View d_cam_objRecognition;

    Camera m_camera;
    cv::Mat img_from_objRecognition;
    int state_from_objRecognition;
    std::mutex mMutexPoseImage;
    std::unique_ptr<pangolin::Var<bool>> menuFollowCamera;
    std::unique_ptr<pangolin::Var<bool>> menuCamView;
    std::unique_ptr<pangolin::Var<bool>> menuTopView;
    std::unique_ptr<pangolin::Var<bool>> menuShowPoints;
    std::unique_ptr<pangolin::Var<bool>> menuShowKeyFrames;
    std::unique_ptr<pangolin::Var<bool>> menuShowGraph;
    std::unique_ptr<pangolin::Var<bool>> menuShowCameraTrajectory;
    std::unique_ptr<pangolin::Var<bool>> menuShow3DObject;
    std::unique_ptr<pangolin::Var<bool>> menuShowInertialGraph;
    std::unique_ptr<pangolin::Var<bool>> menuLocalizationMode;
    std::unique_ptr<pangolin::Var<bool>> menuReset;
    std::unique_ptr<pangolin::Var<bool>> menuStepByStep; // false, true
    std::unique_ptr<pangolin::Var<bool>> menuStep;
    std::unique_ptr<pangolin::Var<bool>> menuStop;
};

} // namespace ORB_SLAM3

#endif // VIEWER_H
