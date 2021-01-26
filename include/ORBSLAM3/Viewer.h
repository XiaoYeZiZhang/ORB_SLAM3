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
#include "ObjectRecognitionThread.h"
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

    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

    void SetPointCloudModel(
        std::shared_ptr<ObjRecognition::Object> &pointCloud_model) {
        m_pointCloud_model = pointCloud_model;
    }
    void SetThreadHandler(
        std::shared_ptr<ObjRecognition::ObjRecogThread> &thread_handler) {
        m_thread_handler = thread_handler;
    }

    bool both;
    // draw another window for objRecognition
    void SwitchWindow();
    void Draw();
    void SetSLAMInfo(
        const cv::Mat &img, const int &slam_state, const int &image_num,
        const cv::Mat &camPos);
    void GetSLAMInfo(cv::Mat &img, int &state, int &image_num);

    void DrawSLAMInit();
    void DrawObjRecognitionInit();
    void DrawDetectorInit();
    static void
    DrawBoundingboxInImage(const vector<Eigen::Vector3d> &boundingbox);
    void Draw3dText();
    void
    DrawPointCloudInImage(const std::vector<Eigen::Vector3d> &pointcloud_pos);
    void ShowConnectedKeyframes();
    void ShowConnectedMapPoints();

    bool GetIsStopFlag() {
        return m_is_stop;
    }

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
    std::shared_ptr<ObjRecognition::ObjRecogThread> m_thread_handler;
    Eigen::Matrix<double, 3, 3> m_Row = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 1> m_tow = Eigen::Matrix<double, 3, 1>::Zero();
    std::vector<cv::Mat> m_trajectory;

    // draw another window for objRecognition
    int m_switch_window_flag;
    bool m_is_stop;
    pangolin::OpenGlRenderState m_s_cam_slam;
    pangolin::View m_d_cam_slam;
    pangolin::OpenGlRenderState m_s_cam_objRecognition;
    pangolin::View m_d_cam_objRecognition;
    pangolin::OpenGlRenderState m_s_cam_detector;
    pangolin::View m_d_cam_detector;

    Camera m_camera;
    cv::Mat m_img_from_objRecognition;
    int m_slam_state_from_objRecognition;
    int m_img_num;
    cv::Mat m_cam_pos;
    std::mutex m_pose_image_mutex;
    std::unique_ptr<pangolin::Var<bool>> m_menu_follow_camera;
    std::unique_ptr<pangolin::Var<bool>> m_menu_cam_view;
    std::unique_ptr<pangolin::Var<bool>> m_menu_top_view;
    std::unique_ptr<pangolin::Var<bool>> m_menu_show_points;
    std::unique_ptr<pangolin::Var<bool>> m_menu_show_keyframes;
    std::unique_ptr<pangolin::Var<bool>> m_menu_show_graph;
    std::unique_ptr<pangolin::Var<bool>> m_menu_show_camera_trajectory;
    std::unique_ptr<pangolin::Var<bool>> m_menu_show_inertial_graph;
    std::unique_ptr<pangolin::Var<bool>> m_menu_reset;
    std::unique_ptr<pangolin::Var<bool>> m_menu_stepbystep;
    std::unique_ptr<pangolin::Var<bool>> m_menu_step;
    std::unique_ptr<pangolin::Var<bool>> m_menu_stop;
    pangolin::GlTexture m_image_texture;
    int m_image_width;
    int m_image_height;
    std::vector<Eigen::Vector3d> m_boundingbox;
    ObjRecognition::ObjRecogResult GetObjRecognitionResult();
};

} // namespace ORB_SLAM3

#endif // VIEWER_H
