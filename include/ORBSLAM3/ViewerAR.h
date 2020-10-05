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

#ifndef VIEWERAR_H
#define VIEWERAR_H

#include <mutex>
#include <opencv2/core/core.hpp>
#include <pangolin/pangolin.h>
#include <string>
#include "ORBSLAM3/ScannerStruct/Struct.h"
#include "ORBSLAM3/System.h"
#include "System.h"

namespace ORB_SLAM3 {
class System;
struct MapHandler3D : public pangolin::Handler3D {

    MapHandler3D(
        pangolin::OpenGlRenderState &cam_state,
        pangolin::AxisDirection enforce_up = pangolin::AxisNone,
        float trans_scale = 0.01f,
        float zoom_fraction = PANGO_DFLT_HANDLER3D_ZF)
        : Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction) {
        m_view_state = &cam_state;
    };

    void Mouse(
        pangolin::View &display, pangolin::MouseButton button, int x, int y,
        bool pressed, int button_state) {

        pangolin::Handler3D::Mouse(
            display, button, x, y, pressed, button_state);

        if (pangolin::MouseButtonLeft == button) {
            if (button_state == 0) {
                m_is_left_button_down = false;
            }
            if (button_state == 1) {
                m_left_button_pos.x() = x;
                m_left_button_pos.y() = y;
                m_is_left_button_down = true;
            }
        }

        if (pangolin::MouseButtonRight == button) {
            if (button_state == 0) {
                m_right_button_pos.x() = x;
                m_right_button_pos.y() = y;
                m_is_right_button_down = true;
            } else if (button_state == 1) {
                m_is_right_button_down = false;
            }
        }
    }

    void MouseMotion(pangolin::View &display, int x, int y, int button_state) {
        pangolin::Handler3D::MouseMotion(display, x, y, button_state);
        if (button_state ==
            pangolin::MouseButtonLeft + pangolin::KeyModifierCtrl) {
            m_left_button_pos.x() = x;
            m_left_button_pos.y() = y;
        }
    }

    int GetRightButtonPos(Eigen::Vector2d &right_button_pos) {
        right_button_pos = m_right_button_pos;
        return m_is_right_button_down;
    }

    int GetLeftButtonPos(Eigen::Vector2d &left_button_pos) {
        left_button_pos = m_left_button_pos;
        return m_is_left_button_down;
    }

    bool m_is_left_button_down = false;
    bool m_is_right_button_down = false;
    pangolin::OpenGlRenderState *m_view_state;
    Eigen::Vector2d m_left_button_pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_right_button_pos = Eigen::Vector2d::Zero();
};

class Plane {
public:
    Plane(const std::vector<MapPoint *> &vMPs, const cv::Mat &Tcw);
    Plane(
        const float &nx, const float &ny, const float &nz, const float &ox,
        const float &oy, const float &oz);

    void Recompute();

    // normal
    cv::Mat n;
    // origin
    cv::Mat o;
    // arbitrary orientation along normal
    float rang;
    // transformation from world to the plane
    cv::Mat Tpw;
    pangolin::OpenGlMatrix glTpw;
    // MapPoints that define the plane
    std::vector<MapPoint *> mvMPs;
    // camera pose when the plane was first observed (to compute normal
    // direction)
    cv::Mat mTcw, XC;
};

class ViewerAR {
public:
    ViewerAR();
    std::unique_ptr<MapHandler3D> mp_handler3d;
    void SetFPS(const float fps) {
        mFPS = fps;
        mT = 1e3 / fps;
    }

    bool GetDebugFlag() {
        return m_is_debug_mode;
    }
    bool GetStopFlag() {
        return m_is_stop;
    }
    bool GetFixFlag() {
        return m_is_fix;
    }
    void SetFixFlag(bool state) {
        m_is_fix = state;
    }

    void ComputeAndSetBoundingbox(const pangolin::OpenGlMatrix &Twp_opengl);

    std::vector<Eigen::Vector3d> GetBoundingbox() {
        return m_boundingbox_vertices;
    }

    void SetSLAM(ORB_SLAM3::System *pSystem) {
        mpSystem = pSystem;
    }

    // Main thread function.
    void Run();

    void ChangeShape(pangolin::OpenGlMatrix);
    void SetCameraCalibration(
        const float &fx_, const float &fy_, const float &cx_, const float &cy_);

    void SetImagePose(
        const cv::Mat &im, const cv::Mat &Tcw, const int &status,
        const std::vector<cv::KeyPoint> &vKeys,
        const std::vector<MapPoint *> &vMPs);

    void GetImagePose(
        cv::Mat &im, cv::Mat &Tcw, int &status,
        std::vector<cv::KeyPoint> &vKeys, std::vector<MapPoint *> &vMPs);

private:
    // SLAM
    ORB_SLAM3::System *mpSystem;

    void PrintStatus(const int &status, const bool &bLocMode, cv::Mat &im);
    void AddTextToImage(
        const std::string &s, cv::Mat &im, const int r = 0, const int g = 0,
        const int b = 0);
    void LoadCameraPose(const cv::Mat &Tcw);
    void DrawImageTexture(pangolin::GlTexture &imageTexture, cv::Mat &im);
    void DrawCube(const float x = 0, const float y = 0, const float z = 0);
    void DrawPlane(int ndivs, float ndivsize);
    void DrawPlane(Plane *pPlane, int ndivs, float ndivsize);
    void DrawTrackedPoints(
        const std::vector<cv::KeyPoint> &vKeys,
        const std::vector<MapPoint *> &vMPs, cv::Mat &im);

    Plane *DetectPlane(
        const cv::Mat Tcw, const std::vector<MapPoint *> &vMPs,
        const int iterations = 50);

    // frame rate
    float mFPS, mT;
    float fx, fy, cx, cy;

    Eigen::Vector3d GetRay(
        const Eigen::Matrix4d &transformationMatrix,
        const Eigen::Matrix4d &projectionMatrix);
    Eigen::Vector3d Change2PlaneCoords(
        pangolin::OpenGlMatrix Plane_wp, Eigen::Vector3d word_coords,
        bool is_point = true);

    Eigen::Matrix4d Change2EigenMatrix(pangolin::OpenGlMatrix opengl_matrix);

    virtual void RegistEvents() {
    }

    // user interface
    void decrease_shape();
    void increase_shape();
    void up_move();
    void down_move();
    void left_move();
    void right_move();
    void front_move();
    void back_move();

    // Last processed image and computed pose by the SLAM
    std::mutex mMutexPoseImage;
    cv::Mat mTcw;
    cv::Mat mImage;
    int mStatus;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<MapPoint *> mvMPs;

    // scann
    Object m_boundingbox;
    MouseState m_mouseState;
    Scene m_scene;
    Camera m_camera;
    bool m_is_debug_mode;
    bool m_is_stop;
    bool m_is_fix;
    pangolin::OpenGlRenderState s_cam;
    std::vector<Eigen::Vector3d> m_boundingbox_vertices;
};
} // namespace ORB_SLAM3

#endif // VIEWERAR_H
