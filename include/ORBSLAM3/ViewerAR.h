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

typedef double M3DVector2d[2];
typedef double M3DVector3d[3];
typedef double M3DVector4d[4];
typedef double M3DMatrix44d[16];
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
            } else if (button_state == 1) {
                m_left_button_pos.x() = x;
                m_left_button_pos.y() = y;
                m_is_left_button_down = true;
            } else if (
                button_state ==
                pangolin::MouseButtonLeft + pangolin::KeyModifierCtrl) {
                m_left_button_pos.x() = x;
                m_left_button_pos.y() = y;
                m_select_area_flag = true;
            } else if (button_state == pangolin::KeyModifierCtrl) {
                m_select_area_flag = false;
            }

            m_region_right_top.x() = x;
            m_region_right_top.y() = y;
            m_region_left_bottom.x() = x;
            m_region_left_bottom.y() = y;
            VLOG(5) << "mousex: " << m_region_left_bottom.x() << " "
                    << m_region_left_bottom.y() << m_region_right_top.x() << " "
                    << m_region_right_top.y();
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
            m_region_right_top.x() = x;
            m_region_right_top.y() = y;
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

    int GetSelected2DRegion(
        Eigen::Vector2d &region_left_bottom,
        Eigen::Vector2d &region_right_top) {
        region_left_bottom = m_region_left_bottom;
        region_right_top = m_region_right_top;
        return m_select_area_flag;
    }

    bool m_select_area_flag = false;
    bool m_is_left_button_down = false;
    bool m_is_right_button_down = false;
    pangolin::OpenGlRenderState *m_view_state;
    Eigen::Vector2d m_left_button_pos = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_right_button_pos = Eigen::Vector2d::Zero();

    Eigen::Vector2d m_region_left_bottom = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_region_right_top = Eigen::Vector2d::Zero();
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
    std::unique_ptr<MapHandler3D> mp_scan_handler3d;
    std::unique_ptr<MapHandler3D> mp_SfM_handler3d;
    void SetFPS(const float fps) {
        mFPS = fps;
        mT = 1e3 / fps;
    }

    bool GetScanDebugFlag() {
        return m_is_scan_debug_mode;
    }
    void SetSfMDebugReverse() {
        *menu_SfM_debug = !(*menu_SfM_debug);
    }
    bool GetSfMDebugFlag() {
        return m_is_SfM_debug_mode;
    }
    bool GetSfMContinueFlag() {
        return m_is_SfM_continue_mode;
    }
    bool GetSfMContinueLBAFlag() {
        return m_is_SfM_continue_LBA_mode;
    }
    bool GetSaveMapPointAfterLBAFlag() {
        return m_is_SfM_save_mpp_after_LBA_mode;
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

    void SetSfMFinishFlag() {
        m_is_SfMFinish = true;
    }

    std::vector<Eigen::Vector3d>
    ComputeBoundingbox_W(const pangolin::OpenGlMatrix &Twp_opengl);

    std::vector<Eigen::Vector3d> GetScanBoundingbox_W() {
        return m_boundingbox_w;
    }

    void SetSLAM(ORB_SLAM3::System *pSystem) {
        mpSystem = pSystem;
        mpMapDrawer = pSystem->mpMapDrawer;
    }

    // Main thread function.
    void Run();
    std::vector<Map *> GetAllMapPoints();
    void ProjectMapPointInImage(
        const cv::Mat &Tcw, const std::vector<double> &bbx,
        std::vector<cv::KeyPoint> &keypoints_outbbx,
        std::vector<cv::KeyPoint> &keypoints_inbbx);
    void GetCurrentMapPointInBBX(
        const vector<Plane *> &vpPlane, const bool &is_insert_cube,
        int &mappoint_num_inbbx);

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
    MapDrawer *mpMapDrawer;

    void
    DrawBoundingbox(const float x = 0, const float y = 0, const float z = 0);
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

    static Eigen::Matrix4d
    ChangeOpenglMatrix2EigenMatrix(pangolin::OpenGlMatrix opengl_matrix);

    virtual void RegistEvents() {
    }

    void DrawSelected2DRegion();
    void Select2DRegion();
    bool DropInArea(
        M3DVector3d x, const M3DMatrix44d model_view, const M3DMatrix44d proj,
        const int viewport[4], const Eigen::Vector2d &left_bottom,
        const Eigen::Vector2d &right_top);
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

    // show another pangolin window for sfm
    bool switch_window_flag;
    void SwitchWindow();
    void Draw(int w, int h);
    void DrawScanInit(int w, int h);
    void DrawSfMInit(int w, int h);
    std::unique_ptr<pangolin::Var<bool>> menu_setboundingbox;
    std::unique_ptr<pangolin::Var<bool>> menu_fixBBX;
    std::unique_ptr<pangolin::Var<bool>> menu_stop;
    std::unique_ptr<pangolin::Var<bool>> menu_clear;
    std::unique_ptr<pangolin::Var<bool>> menu_scan_debug;
    std::unique_ptr<pangolin::Var<float>> menu_cubesize;
    std::unique_ptr<pangolin::Var<bool>> menu_drawgrid;
    std::unique_ptr<pangolin::Var<int>> menu_ngrid;
    std::unique_ptr<pangolin::Var<float>> menu_sizegrid;
    std::unique_ptr<pangolin::Var<bool>> menu_drawTrackedpoints;
    std::unique_ptr<pangolin::Var<bool>> menu_drawMappoints;
    std::unique_ptr<pangolin::Var<bool>> menu_SfM_debug;
    std::unique_ptr<pangolin::Var<bool>> menu_SfM_continue;
    std::unique_ptr<pangolin::Var<bool>> menu_SfM_continue_LBA;
    std::unique_ptr<pangolin::Var<bool>> menu_SfM_savemmp_after_LBA;

    cv::Mat im_scan, Tcw_scan;
    int status_scan;
    vector<cv::KeyPoint> vKeys_scan;
    vector<MapPoint *> vMPs_scan;

    void DrawMapPoints_SuperPoint(
        const std::vector<double> &boundingbox_w_corner,
        const std::set<MapPoint *> mappoint_picked);
    void Pick3DPointCloud();
    // scan
    Object m_boundingbox_p;
    MouseState m_mouseState;
    Scene m_scene;
    Camera m_camera;
    bool m_is_scan_debug_mode;
    bool m_is_SfM_debug_mode;
    bool m_is_SfM_continue_mode;
    bool m_is_SfM_continue_LBA_mode;
    bool m_is_SfM_save_mpp_after_LBA_mode;
    bool m_is_stop;
    bool m_is_fix;
    bool m_is_SfMFinish;
    pangolin::OpenGlRenderState s_cam_scan;
    pangolin::View d_cam_scan;

    pangolin::OpenGlRenderState s_cam_SfM;
    pangolin::View d_cam_SfM;
    std::vector<Eigen::Vector3d> m_boundingbox_w;
    std::vector<double> m_boundingbox_corner;

    // sfm visualization
    bool m_select_area_flag = false;
    Eigen::Vector2d m_region_right_top = Eigen::Vector2d::Zero();
    Eigen::Vector2d m_region_left_down = Eigen::Vector2d::Zero();
    std::set<MapPoint *> mappoints_picked;

    float change_shape_unit;
};
} // namespace ORB_SLAM3

#endif // VIEWERAR_H
