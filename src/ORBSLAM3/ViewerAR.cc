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

#include "Visualizer/GlobalImageViewer.h"
#include "ORBSLAM3/ViewerAR.h"
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <cstdlib>
#include "ORBSLAM3/ScannerStruct/Struct.h"
#include "ORBSLAM3/ViewerCommon.h"
#include "include/Tools.h"
#include <glm/gtc/type_ptr.hpp>
#include "mode.h"
#include <opencv2/core/eigen.hpp>
using namespace std;

namespace ORB_SLAM3 {

const float eps = 1e-4;

cv::Mat ExpSO3(const float &x, const float &y, const float &z) {
    cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
    const float d2 = x * x + y * y + z * z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3, 3) << 0, -z, y, z, 0, -x, -y, x, 0);
    if (d < eps)
        return (I + W + 0.5f * W * W);
    else
        return (I + W * sin(d) / d + W * W * (1.0f - cos(d)) / d2);
}

cv::Mat ExpSO3(const cv::Mat &v) {
    return ExpSO3(v.at<float>(0), v.at<float>(1), v.at<float>(2));
}

ViewerAR::ViewerAR() {
    m_is_scan_debug_mode = false;
    m_is_SfM_debug_mode = false;
    m_is_SfM_continue_LBA_mode = false;
    m_is_SfM_save_mpp_after_LBA_mode = false;
    m_change_shape_unit = 0.02;
    m_is_stop = false;
    m_is_fix = false;
    m_switch_window_flag = false;
    RegistEvents();
}

void ViewerAR::decrease_shape() {
    m_boundingbox_p.SetChangeShapeOffset(-m_change_shape_unit);
}

void ViewerAR::increase_shape() {
    m_boundingbox_p.SetChangeShapeOffset(m_change_shape_unit);
}

void ViewerAR::up_move() {
    m_boundingbox_p.MoveObject(-m_change_shape_unit, 1);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::down_move() {
    m_boundingbox_p.MoveObject(+m_change_shape_unit, 1);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::left_move() {
    m_boundingbox_p.MoveObject(-m_change_shape_unit, 0);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::right_move() {
    m_boundingbox_p.MoveObject(m_change_shape_unit, 0);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::front_move() {
    m_boundingbox_p.MoveObject(m_change_shape_unit, 2);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::back_move() {
    m_boundingbox_p.MoveObject(-m_change_shape_unit, 2);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

Eigen::Vector3d ViewerAR::FromWorld2Plane(
    const Eigen::Vector3d &mappoint_w,
    const pangolin::OpenGlMatrix &Twp_opengl) {
    Eigen::Matrix4d Twp = ChangeOpenglMatrix2EigenMatrix(Twp_opengl);
    Eigen::Matrix4d Tpw = Twp.inverse();
    Eigen::Vector4d mappoint_p_4_4 =
        Tpw * Eigen::Vector4d(mappoint_w(0), mappoint_w(1), mappoint_w(2), 1.0);
    return Eigen::Vector3d(
        mappoint_p_4_4(0) / mappoint_p_4_4(3),
        mappoint_p_4_4(1) / mappoint_p_4_4(3),
        mappoint_p_4_4(2) / mappoint_p_4_4(3));
}

std::vector<Eigen::Vector3d>
ViewerAR::ComputeBoundingbox_W(const pangolin::OpenGlMatrix &Twp_opengl) {
    std::vector<Eigen::Vector3d> boundingbox_w;
    for (size_t i = 0; i < 8; i++) {
        Eigen::Vector3d point_p = Eigen::Vector3d(
            m_boundingbox_p.m_vertex_list_p[i][0],
            m_boundingbox_p.m_vertex_list_p[i][1],
            m_boundingbox_p.m_vertex_list_p[i][2]);
        Eigen::Vector4d p_4 =
            Eigen::Vector4d(point_p(0), point_p(1), point_p(2), 1.0f);
        Eigen::Matrix4d Twp = ChangeOpenglMatrix2EigenMatrix(Twp_opengl);
        Eigen::Vector4d bbx_4 = Twp * p_4;
        boundingbox_w.emplace_back(Eigen::Vector3d(
            bbx_4[0] / bbx_4[3], bbx_4[1] / bbx_4[3], bbx_4[2] / bbx_4[3]));
    }
    return boundingbox_w;
}

void ViewerAR::SetCameraCalibration(
    const float &fx_, const float &fy_, const float &cx_, const float &cy_) {
    fx = fx_;
    fy = fy_;
    cx = cx_;
    cy = cy_;
}

void ViewerAR::DrawMapPoints_SuperPoint(
    const std::vector<double> &boundingbox_p_corner,
    const std::set<MapPoint *> &mappoint_picked,
    const vector<Plane *> &vpPlane) {
    if (!vpPlane.empty()) {
        Plane *pPlane = vpPlane[0];
        Eigen::Matrix4d Twp = ChangeOpenglMatrix2EigenMatrix(pPlane->glTpw);
        mpMapDrawer->DrawMapPoints_SuperPoint(
            boundingbox_p_corner, mappoint_picked, Twp);
    }
}

std::vector<Map *> ViewerAR::GetAllMapPoints() {
    std::vector<Map *> saved_map;
    struct compFunctor {
        inline bool operator()(Map *elem1, Map *elem2) {
            return elem1->GetId() < elem2->GetId();
        }
    };
    std::copy(
        mpSystem->mpAtlas->mspMaps.begin(), mpSystem->mpAtlas->mspMaps.end(),
        std::back_inserter(saved_map));
    sort(saved_map.begin(), saved_map.end(), compFunctor());
    return saved_map;
}

void ViewerAR::GetCurrentMapPointInBBX(
    const vector<Plane *> &vpPlane, const bool &is_insert_cube,
    int &mappoint_num_inbbx) {
    mappoint_num_inbbx = -1;
    if (!is_insert_cube) {
        m_boundingbox_corner_p = std::vector<double>();
    }

    if (!vpPlane.empty()) {
        mappoint_num_inbbx = 0;
        Plane *pPlane = vpPlane[0];
        m_boundingbox_corner_p = {m_boundingbox_p.minCornerPoint(0),
                                  m_boundingbox_p.minCornerPoint(1),
                                  m_boundingbox_p.minCornerPoint(2),
                                  m_boundingbox_p.maxCornerPoint(0),
                                  m_boundingbox_p.maxCornerPoint(1),
                                  m_boundingbox_p.maxCornerPoint(2)};

        std::vector<Map *> saved_map = GetAllMapPoints();
        for (Map *pMi : saved_map) {
            for (MapPoint *pMPi : pMi->GetAllMapPoints()) {
                cv::Mat tmpPos = pMPi->GetWorldPos();

                Eigen::Vector3d mappoint_w = Eigen::Vector3d::Zero();
                cv::cv2eigen(tmpPos, mappoint_w);
                // change to plane coords

                Eigen::Vector3d mappoint_p =
                    FromWorld2Plane(mappoint_w, pPlane->glTpw);
                if (mappoint_p(0) >= m_boundingbox_corner_p[0] &&
                    mappoint_p(0) <= m_boundingbox_corner_p[3] &&
                    mappoint_p(1) >= m_boundingbox_corner_p[1] &&
                    mappoint_p(1) <= m_boundingbox_corner_p[4] &&
                    mappoint_p(2) >= m_boundingbox_corner_p[2] &&
                    mappoint_p(2) <= m_boundingbox_corner_p[5]) {
                    mappoint_num_inbbx++;
                }
            }
        }
    }
}

void ViewerAR::ProjectMapPointInImage(
    const vector<Plane *> &vpPlane, const cv::Mat &Tcw,
    const std::vector<double> &bbx_p,
    std::vector<cv::KeyPoint> &keypoints_outbbx,
    std::vector<cv::KeyPoint> &keypoints_inbbx) {

    keypoints_outbbx.clear();
    keypoints_inbbx.clear();
    if (Tcw.empty()) {
        VLOG(0) << "Tcw is null";
        return;
    }

    if (!vpPlane.empty()) {
        Plane *pPlane = vpPlane[0];
        std::vector<Map *> saved_map = GetAllMapPoints();
        for (Map *pMi : saved_map) {
            for (MapPoint *pMPi : pMi->GetAllMapPoints()) {
                cv::Mat tmpPos_w = pMPi->GetWorldPos();
                Eigen::Vector3d mappoint_w = Eigen::Vector3d::Zero();
                cv::cv2eigen(tmpPos_w, mappoint_w);

                Eigen::Vector3d mappoint_p =
                    FromWorld2Plane(mappoint_w, pPlane->glTpw);

                Eigen::Vector3d pos_w;
                Eigen::Matrix4d Tcw_eigen;
                cv2eigen(tmpPos_w, pos_w);
                cv2eigen(Tcw, Tcw_eigen);
                Eigen::Vector4d pos_w_4 =
                    Eigen::Vector4d(pos_w(0), pos_w(1), pos_w(2), 1.0);
                Eigen::Vector4d pos_c = Tcw_eigen * pos_w_4;
                Eigen::Vector3d pos_c_3 = Eigen::Vector3d(
                    pos_c(0) / pos_c(3), pos_c(1) / pos_c(3),
                    pos_c(2) / pos_c(3));
                Eigen::Vector3d pos_i =
                    ObjRecognition::CameraIntrinsic::GetInstance().GetEigenK() *
                    pos_c_3;
                Eigen::Vector2d pos_i_2 =
                    Eigen::Vector2d(pos_i(0) / pos_i(2), pos_i(1) / pos_i(2));
                // change cv::point2f to cv::keypoint
                cv::KeyPoint kp(cv::Point2f(pos_i_2(0), pos_i_2(1)), 8);

                if (!bbx_p.empty() && mappoint_p(0) > bbx_p[0] &&
                    mappoint_p(0) < bbx_p[3] && mappoint_p(1) > bbx_p[1] &&
                    mappoint_p(1) < bbx_p[4] && mappoint_p(2) > bbx_p[2] &&
                    mappoint_p(2) < bbx_p[5]) {
                    keypoints_inbbx.emplace_back(kp);
                } else {
                    keypoints_outbbx.emplace_back(kp);
                }
            }
        }
    }
}

void ViewerAR::SwitchWindow() {
    std::cout << "switch flag" << std::endl;
    if (m_switch_window_flag) {
        d_cam_scan.show = false;
        d_cam_SfM = pangolin::CreateDisplay()
                        .SetBounds(
                            0.0, 1.0f,
                            pangolin::Attach::Pix(m_scene.GetSceneBarWidth()),
                            1.0, (float)m_im_scan.cols / m_im_scan.rows)
                        .SetHandler(mp_SfM_handler3d.get());
        d_cam_SfM.show = true;
    } else {
        d_cam_SfM.show = false;
        d_cam_scan =
            pangolin::CreateDisplay()
                .SetBounds(
                    0, 1.0f, pangolin::Attach::Pix(m_scene.GetSceneBarWidth()),
                    1.0f, (float)m_im_scan.cols / m_im_scan.rows)
                .SetLock(pangolin::LockLeft, pangolin::LockTop)
                .SetHandler(mp_scan_handler3d.get());
        d_cam_scan.show = true;
    }
    m_switch_window_flag = !m_switch_window_flag;
}

void ViewerAR::DrawScanInit(int w, int h) {
    pangolin::CreatePanel("menu").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(m_scene.GetSceneBarWidth()));
    m_menu_clear = std::make_unique<pangolin::Var<bool>>(
        "menu.Choose Another Plane", false, false);
    m_menu_setboundingbox =
        std::make_unique<pangolin::Var<bool>>("menu.Set a BBX", false, false);
    m_menu_fixBBX =
        std::make_unique<pangolin::Var<bool>>("menu.Fix BBX", false, false);
    m_menu_stop =
        std::make_unique<pangolin::Var<bool>>("menu.Finish Scan", false, false);

    m_menu_scan_debug =
        std::make_unique<pangolin::Var<bool>>("menu.Scan Debug", false, false);
    m_boundingbox_p.SetSize(0.05);
    // plane grid number
    m_menu_ngrid =
        std::make_unique<pangolin::Var<int>>("menu. Grid Elements", 3, 1, 10);
    m_menu_drawTrackedpoints =
        std::make_unique<pangolin::Var<bool>>("menu.Draw Points", false, true);
    m_menu_drawMappoints = std::make_unique<pangolin::Var<bool>>(
        "menu.Draw Proj Mappoints", false, true);
    m_menu_SfM_debug =
        std::make_unique<pangolin::Var<bool>>("menu.SfM Debug", false, false);
    m_menu_SfM_continue = std::make_unique<pangolin::Var<bool>>(
        "menu.SfM Continue", false, false);
    m_menu_SfM_continue_LBA = std::make_unique<pangolin::Var<bool>>(
        "menu.SfM Continue LBA", false, false);
    m_menu_SfM_savemmp_after_LBA = std::make_unique<pangolin::Var<bool>>(
        "menu.SfM SaveMappoints", false, false);
    // handle keyboard event
    std::function<void(void)> decrease_shape_key_callback =
        std::bind(&ViewerAR::decrease_shape, this);
    std::function<void(void)> increase_shape_key_callback =
        std::bind(&ViewerAR::increase_shape, this);

    std::function<void(void)> up_key_callback =
        std::bind(&ViewerAR::up_move, this);
    std::function<void(void)> down_key_callback =
        std::bind(&ViewerAR::down_move, this);
    std::function<void(void)> left_key_callback =
        std::bind(&ViewerAR::left_move, this);
    std::function<void(void)> right_key_callback =
        std::bind(&ViewerAR::right_move, this);
    std::function<void(void)> front_key_callback =
        std::bind(&ViewerAR::front_move, this);
    std::function<void(void)> back_key_callback =
        std::bind(&ViewerAR::back_move, this);
    pangolin::RegisterKeyPressCallback('-', decrease_shape_key_callback);
    pangolin::RegisterKeyPressCallback('=', increase_shape_key_callback);
    pangolin::RegisterKeyPressCallback('w', up_key_callback);
    pangolin::RegisterKeyPressCallback('s', down_key_callback);
    pangolin::RegisterKeyPressCallback('a', left_key_callback);
    pangolin::RegisterKeyPressCallback('d', right_key_callback);
    pangolin::RegisterKeyPressCallback('f', front_key_callback);
    pangolin::RegisterKeyPressCallback('b', back_key_callback);
    // define projection and initial movelview matrix: default
    s_cam_scan = pangolin::OpenGlRenderState();
    mp_scan_handler3d.reset(new MapHandler3D(s_cam_scan));

    d_cam_scan =
        pangolin::Display("image")
            .SetBounds(
                0, 1.0f, pangolin::Attach::Pix(m_scene.GetSceneBarWidth()),
                1.0f, (float)w / h)
            .SetLock(pangolin::LockLeft, pangolin::LockTop)
            .SetHandler(mp_scan_handler3d.get());
}

void ViewerAR::DrawSfMInit(int w, int h) {
    s_cam_SfM = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.7, -3.5, 0, 0, 0, 0.0, -1.0, 0.0));
    mp_SfM_handler3d.reset(new MapHandler3D(s_cam_SfM));

    std::function<void(void)> stop_selected_2d_region_callback =
        std::bind(&ViewerAR::Select2DRegion, this);
    pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_CTRL + 's', stop_selected_2d_region_callback);
}

void ViewerAR::Select2DRegion() {
    m_select_area_flag = mp_SfM_handler3d->GetSelected2DRegion(
        m_region_left_down, m_region_right_top);
}

void Project3DXYZToUV(
    M3DVector2d point_out, const M3DMatrix44d model_view,
    const M3DMatrix44d proj, const int view_port[4],
    const M3DVector3d point_in) {
    M3DVector3d point_result;
    gluProject(
        point_in[0], point_in[1], point_in[2], model_view, proj, view_port,
        &point_result[0], &point_result[1], &point_result[2]);
    point_out[0] = point_result[0];
    point_out[1] = point_result[1];
}

bool ViewerAR::DropInArea(
    M3DVector3d x, const M3DMatrix44d model_view, const M3DMatrix44d proj,
    const int viewport[4], const Eigen::Vector2d &left_bottom,
    const Eigen::Vector2d &right_top) {
    M3DVector2d win_coords;
    Project3DXYZToUV(win_coords, model_view, proj, viewport, x);
    if ((win_coords[0] < left_bottom[0] && win_coords[0] < right_top[0]) ||
        (win_coords[0] > left_bottom[0] && win_coords[0] > right_top[0]))
        return false;
    if ((win_coords[1] < left_bottom[1] && win_coords[1] < right_top[1]) ||
        (win_coords[1] > left_bottom[1] && win_coords[1] > right_top[1]))
        return false;
    return true;
}

void ViewerAR::Pick3DPointCloud() {
    if (!m_select_area_flag)
        return;
    m_select_area_flag = false;
    GLint viewport[4];
    glPushMatrix();
    glGetIntegerv(GL_VIEWPORT, viewport);

    pangolin::OpenGlMatrix modelview_matrix = s_cam_SfM.GetModelViewMatrix();
    pangolin::OpenGlMatrix projection_matrix = s_cam_SfM.GetProjectionMatrix();

    int covisualize_keyframe_num = 0;
#ifdef SUPERPOINT
    covisualize_keyframe_num = 4;
#endif

#ifdef MONO
    covisualize_keyframe_num = 4;
#endif
    auto all_mappoints = mpMapDrawer->mpAtlas_superpoint->GetAllMapPoints(
        covisualize_keyframe_num);
    for (auto mappoint : all_mappoints) {
        cv::Mat pose_cv = mappoint->GetWorldPos();
        Eigen::Vector3d pose;
        cv::cv2eigen(pose_cv, pose);
        GLdouble pose_array[3] = {pose[0], pose[1], pose[2]};
        if (DropInArea(
                pose_array, modelview_matrix.m, projection_matrix.m, viewport,
                m_region_left_down, m_region_right_top)) {
            if (!m_mappoints_picked.count(mappoint)) {
                auto observations = mappoint->GetObservations();
                for (auto observation : observations) {
                    auto keyframe = observation.first;
                    auto idx = std::get<0>(observation.second);
                    auto keyframe_id = keyframe->mnId;
                    cv::Mat img = keyframe->imgLeft.clone();
                    std::vector<cv::KeyPoint> keypoints;
                    cv::KeyPoint keypoint = keyframe->mvKeysUn_superpoint[idx];
                    keypoints.emplace_back(keypoint);
                    cv::drawKeypoints(img, keypoints, img);
                    cv::imwrite(
                        "/home/zhangye/data1/sfm/mappoints/" +
                            std::to_string(mappoint->mnId) + "--" +
                            std::to_string(keyframe_id) + ".png",
                        img);
                }
                m_mappoints_picked.insert(mappoint);
            }
        }
    }
}

void ViewerAR::DrawSelected2DRegion() {
    if (!m_select_area_flag)
        return;
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(
        m_scene.GetSceneBarWidth(),
        m_scene.GetSceneWidth() + m_scene.GetSceneBarWidth(), 0,
        m_scene.GetSceneHeight());

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glEnable(GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1.0, 1.0, 0, 0.2);
    glRectf(
        m_region_left_down.x(), m_region_left_down.y(), m_region_right_top.x(),
        m_region_right_top.y());
    glEnable(GL_LINE_STIPPLE);

    glColor4f(1, 0, 0, 0.5);
    glLineStipple(3, 0xAAAA);
    glBegin(GL_LINE_STIPPLE);
    glVertex2f(m_region_left_down.x(), m_region_left_down.y());
    glVertex2f(m_region_right_top.x(), m_region_left_down.y());
    glVertex2f(m_region_right_top.x(), m_region_right_top.y());
    glVertex2f(m_region_left_down.x(), m_region_right_top.y());
    glEnd();
    glDisable(GL_LINE_STIPPLE);
    glDisable(GL_BLEND);
}

void ViewerAR::Draw(int w, int h) {
    pangolin::GlTexture imageTexture(
        w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::OpenGlMatrixSpec P = pangolin::ProjectionMatrixRDF_TopLeft(
        w, h, fx, fy, cx, cy, 0.001, 1000);

    vector<Plane *> vpPlane;

    while (true) {
        m_is_scan_debug_mode = *m_menu_scan_debug;
        m_is_SfM_debug_mode = *m_menu_SfM_debug;
        m_is_SfM_continue_mode = *m_menu_SfM_continue;
        m_is_SfM_continue_LBA_mode = *m_menu_SfM_continue_LBA;
        m_is_SfM_save_mpp_after_LBA_mode = *m_menu_SfM_savemmp_after_LBA;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!m_switch_window_flag) {
            d_cam_scan.show = true;
            // Activate m_camera view
            d_cam_scan.Activate(s_cam_scan);
            glColor3f(1.0, 1.0, 1.0);
            // Get last image and its computed pose from SLAM
            GetImagePose(
                m_im_scan, m_Tcw_scan, m_status_scan, m_vKeys_scan,
                m_vMPs_scan);

            if (!m_Tcw_scan.empty()) {
                cv::Mat Rwc = m_Tcw_scan.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * m_Tcw_scan.rowRange(0, 3).col(3);
                // set m_camera position
                m_camera.SetCamPos(
                    twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));
            }

            // get mappoint num in boundingbox
            int current_mappoint_num_inbbx = 0;
            GetCurrentMapPointInBBX(
                vpPlane, *m_menu_setboundingbox, current_mappoint_num_inbbx);

            // Add text to image
            PrintStatus(m_status_scan, current_mappoint_num_inbbx, m_im_scan);
            cv::cvtColor(m_im_scan, m_im_scan, CV_GRAY2RGB);
            if (*m_menu_drawTrackedpoints)
                DrawTrackedPoints(m_vKeys_scan, m_vMPs_scan, m_im_scan);

            // Draw image
            if (*m_menu_drawMappoints) {
                // draw mappoints in and out the boundingbox
                std::vector<cv::KeyPoint> keypoints_outbbx;
                std::vector<cv::KeyPoint> keypoints_inbbx;
                ProjectMapPointInImage(
                    vpPlane, m_Tcw_scan, m_boundingbox_corner_p,
                    keypoints_outbbx, keypoints_inbbx);
                cv::drawKeypoints(
                    m_im_scan, keypoints_outbbx, m_im_scan,
                    cv::Scalar(255, 0, 255));
                cv::drawKeypoints(
                    m_im_scan, keypoints_inbbx, m_im_scan,
                    cv::Scalar(0, 255, 255));
            }
            DrawImageTexture(imageTexture, m_im_scan);

            ObjRecognition::GlobalOcvViewer::Draw();
            glClear(GL_DEPTH_BUFFER_BIT);

            // Load m_camera projection
            glMatrixMode(GL_PROJECTION);
            P.Load();

            // load model view matrix
            glMatrixMode(GL_MODELVIEW);

            // Load m_camera pose  set opengl coords, same as slam coords
            // view matrix Tcw
            LoadCameraPose(m_Tcw_scan);

            if (*m_menu_clear) {
                if (!vpPlane.empty()) {
                    for (size_t i = 0; i < vpPlane.size(); i++) {
                        delete vpPlane[i];
                    }
                    vpPlane.clear();
                    m_is_fix = false;
                    m_is_stop = false;
#ifdef SUPERPOINT
                    m_is_SfMFinish = false;
#endif
                    m_boundingbox_p.Reset();
                    m_boundingbox_p.SetSize(0.05);
                    VLOG(0) << "ORBSLAM3: All cubes erased!";
                }
                *m_menu_clear = false;
            }
            // Draw virtual things
            // can only insert cube when slam state is fine
            if (m_status_scan == 2) {
                Plane *pPlane = DetectPlane(m_Tcw_scan, m_vMPs_scan, 50);
                if (pPlane && vpPlane.empty()) {
                    VLOG(0) << "ORBSLAM3: New virtual plane is detected!";
                    vpPlane.push_back(pPlane);
                }
            }

            // draw cube no mater what slam state is
            if (!vpPlane.empty()) {
                m_boundingbox_p.SetExist(true);
                bool bRecompute = false;
                if (mpSystem->MapChanged()) {
                    VLOG(0) << "ORBSLAM3: Map changed. All virtual "
                               "elements are "
                               "recomputed!";
                    bRecompute = true;
                }

                if (vpPlane.size() > 1) {
                    LOG(FATAL) << "plane error";
                }

                Plane *pPlane = vpPlane[0];
                if (pPlane) {
                    if (*m_menu_fixBBX) {
                        // get m_boundingbox_w
                        m_boundingbox_w = ComputeBoundingbox_W(pPlane->glTpw);
                        Eigen::Matrix4d Twp =
                            ChangeOpenglMatrix2EigenMatrix(pPlane->glTpw);
                        mpSystem->mpAtlas->SetTwp(Twp);
#ifdef SUPERPOINT
                        mpSystem->mpAtlas_superpoint->SetTwp(Twp);
#endif

                        m_is_fix = true;
                        *m_menu_fixBBX = false;
                    }
                    if (*m_menu_stop) {
                        m_is_stop = true;
                        *m_menu_stop = false;

#ifdef SUPERPOINT
                        // change to sfm panglin dispaly
                        m_switch_window_flag = !m_switch_window_flag;
                        d_cam_scan.show = false;
                        d_cam_SfM =
                            pangolin::CreateDisplay()
                                .SetBounds(
                                    0.0, 1.0,
                                    pangolin::Attach::Pix(
                                        m_scene.GetSceneBarWidth()),
                                    1.0, (float)m_im_scan.cols / m_im_scan.rows)
                                .SetHandler(mp_SfM_handler3d.get());
                        d_cam_SfM.show = true;
#else
                        break;
#endif
                    }
                    if (bRecompute) {
                        pPlane->Recompute();
                    }

                    ChangeShape(pPlane->glTpw);

                    // plane coords, model matrix:
                    glPushMatrix();
                    // Twp
                    pPlane->glTpw.Multiply();

                    // Draw cube
                    if (*m_menu_setboundingbox) {
                        DrawBoundingbox();
                    }

                    // Draw grid plane
                    DrawPlane(*m_menu_ngrid, 0.05);
                    glPopMatrix();
                }
            }

        } else {
#ifdef SUPERPOINT
            // mpSystem->Shutdown();
            d_cam_SfM.show = true;
            d_cam_SfM.Activate(s_cam_SfM);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            pangolin::glDrawAxis(0.6f);
            DrawSelected2DRegion();
            Pick3DPointCloud();
            DrawMapPoints_SuperPoint(
                m_boundingbox_corner_p, m_mappoints_picked, vpPlane);
            DrawBoundingboxForSfM(m_boundingbox_w);

            // Pick3DPointCloud();
            if (m_is_SfMFinish) {
                break;
            }
#endif
        }

        pangolin::FinishFrame();
        usleep(mT * 1000);
    }
}

void ViewerAR::Run() {
    m_scene.SetSceneSize(
        ObjRecognition::CameraIntrinsic::GetInstance().Width(),
        ObjRecognition::CameraIntrinsic::GetInstance().Height(), 200);

    int w, h;

    while (true) {
        GetImagePose(
            m_im_scan, m_Tcw_scan, m_status_scan, m_vKeys_scan, m_vMPs_scan);
        if (m_im_scan.empty())
            cv::waitKey(mT);
        else {
            w = m_im_scan.cols;
            h = m_im_scan.rows;
            break;
        }
    }

    pangolin::CreateWindowAndBind("Viewer", w + m_scene.GetSceneBarWidth(), h);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // std::function<void(void)> switch_win_callback =
    // std::bind(&ViewerAR::SwitchWindow, this);
    // pangolin::RegisterKeyPressCallback('h', switch_win_callback);

    DrawScanInit(w, h);
#ifdef SUPERPOINT
    DrawSfMInit(w, h);
#endif
    Draw(w, h);
}

void ViewerAR::SetImagePose(
    const cv::Mat &im, const cv::Mat &Tcw, const int &status,
    const vector<cv::KeyPoint> &vKeys,
    const vector<ORB_SLAM3::MapPoint *> &vMPs) {
    unique_lock<mutex> lock(mMutexPoseImage);
    mImage = im.clone();
    mTcw = Tcw.clone();
    mStatus = status;
    mvKeys = vKeys;
    mvMPs = vMPs;
}

void ViewerAR::GetImagePose(
    cv::Mat &im, cv::Mat &Tcw, int &status, std::vector<cv::KeyPoint> &vKeys,
    std::vector<MapPoint *> &vMPs) {
    unique_lock<mutex> lock(mMutexPoseImage);
    im = mImage.clone();
    Tcw = mTcw.clone();
    status = mStatus;
    vKeys = mvKeys;
    vMPs = mvMPs;
}

Eigen::Vector3d ViewerAR::GetRay(
    const float mouse_x, const float mouse_y,
    const Eigen::Matrix4d &transformationMatrix,
    const Eigen::Matrix4d &projectionMatrix) {
    float x = (2.0f * mouse_x) / m_scene.GetSceneWidth() - 1.0f;
    float y = 1.0f - (2.0f * mouse_y) / m_scene.GetSceneHeight();
    float z = 1.0f;
    Eigen::Vector3d ray_nds = Eigen::Vector3d(x, y, z);
    Eigen::Vector4d ray_clip =
        Eigen::Vector4d(ray_nds(0), ray_nds(1), ray_nds(2), 1.0);
    Eigen::Vector4d ray_eye = projectionMatrix.inverse() * ray_clip;
    Eigen::Vector4d ray_world = transformationMatrix.inverse() * ray_eye;

    if (ray_world(3) != 0.0) {
        ray_world(0) /= ray_world(3);
        ray_world(1) /= ray_world(3);
        ray_world(2) /= ray_world(3);
    }

    return Eigen::Vector3d(ray_world(0), ray_world(1), ray_world(2));
}

bool IsIntersectWithTriangle(
    const Eigen::Vector3d &orig, const Eigen::Vector3d &dir,
    Eigen::Vector3d &v0, Eigen::Vector3d &v1, Eigen::Vector3d &v2, float &t,
    float &u, float &v) {
    Eigen::Vector3d E1 = v1 - v0;
    Eigen::Vector3d E2 = v2 - v0;
    Eigen::Vector3d P = dir.cross(E2);
    float det = E1.dot(P);
    Eigen::Vector3d T;
    if (det > 0) {
        T = orig - v0;
    } else {
        T = v0 - orig;
        det = -det;
    }
    if (det < 0.0001f)
        return false;
    u = T.dot(P);
    if (u < 0.0f || u > det)
        return false;
    Eigen::Vector3d Q = T.cross(E1);
    v = dir.dot(Q);
    if (v < 0.0f || u + v > det)
        return false;
    t = E2.dot(Q);
    float fInvDet = 1.0f / det;
    t *= fInvDet;
    u *= fInvDet;
    v *= fInvDet;
    return true;
}

void ViewerAR::ChangeShape(pangolin::OpenGlMatrix Twp) {
#if 0
    Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d projectionMatrix = Eigen::Matrix4d::Identity();
    GLfloat view_model[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, view_model);
    GLdouble projection[16];
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    transformationMatrix << view_model[0], view_model[1], view_model[2],
        view_model[3], view_model[4], view_model[5], view_model[6],
        view_model[7], view_model[8], view_model[9], view_model[10],
        view_model[11], view_model[12], view_model[13], view_model[14],
        view_model[15];

    projectionMatrix << projection[0], projection[1], projection[2],
        projection[3], projection[4], projection[5], projection[6],
        projection[7], projection[8], projection[9], projection[10],
        projection[11], projection[12], projection[13], projection[14],
        projection[15];
        if (m_is_fix) {
            m_boundingbox_p.minTriangleIndex = -1;
            // interaction scan
            Eigen::Vector3d ray = GetRay(
                m_scene.GetSceneWidth() / 2, m_scene.GetSceneHeight() / 2,
                transformationMatrix.transpose(),
                projectionMatrix.transpose());
            // from world point to camera
            Eigen::Vector3d ray_dir_w = (m_camera.GetCamPos() -
            ray).normalized();
            //            glBegin(GL_LINES);
            //            glColor3f(1.0, 1.0, 0.0);
            //            glVertex3f(
            //                m_camera.GetCamPos()[0],
            m_camera.GetCamPos()[1],
            //                m_camera.GetCamPos()[2]);
            //            glVertex3f(ray(0), ray(1), ray(2));
            //            glEnd();
            Eigen::Vector3d CameraPosition_w = m_camera.GetCamPos();
            // change to plane coords
            Eigen::Vector3d CameraPosition_p =
                Change2PlaneCoords(Twp, CameraPosition_w, true);
            Eigen::Vector3d ray_dir_p = Change2PlaneCoords(Twp, ray_dir_w,
            false); float minDistance = INT_MAX; Eigen::Vector3d
            intersection_point = Eigen::Vector3d::Zero();

            for (size_t i = 0; i < m_boundingbox_p.GetAllTriangles().size();
            i++) {
                Triangle thisTriangle = m_boundingbox_p.GetAllTriangles()[i];
                if (i == 0) {
                    auto pts = thisTriangle.GetVertex();
                }
                float t;
                float u;
                float v;

                // use plane coords to compute intersection
                if (IsIntersectWithTriangle(
                        CameraPosition_p, ray_dir_p,
                        thisTriangle.GetVertex()[0],
                        thisTriangle.GetVertex()[1],
                        thisTriangle.GetVertex()[2], t, u, v)) {

                    Eigen::Vector3d intersectionPoint =
                        (1 - u - v) * thisTriangle.GetVertex()[0] +
                        thisTriangle.GetVertex()[1] * u +
                        thisTriangle.GetVertex()[2] * v;

                    float distance =
                        pow((intersectionPoint(0) - CameraPosition_p[0]), 2) +
                        pow((intersectionPoint(1) - CameraPosition_p[1]), 2) +
                        pow((intersectionPoint(2) - CameraPosition_p[2]), 2);
                    if (distance < minDistance) {
                        minDistance = distance;
                        m_boundingbox_p.minTriangleIndex = i;
                        intersection_point = intersectionPoint;
                    }
                }
            }

            // draw this intersection plane
            if (m_boundingbox_p.minTriangleIndex != -1) {
                // draw: under plane coords:
                glPushMatrix();
                Twp.Multiply();
                glBegin(GL_QUADS);
                glColor3f(0.0f, 0.0f, 1.0f);
                Eigen::Vector3d plane_norm_dir =
                    m_boundingbox_p
                        .triangle_plane[m_boundingbox_p.minTriangleIndex + 1]
                        .second -
                    Eigen::Vector3d(0, 0, 0);

                double radian_angle = atan2(
                    plane_norm_dir.cross(ray_dir_p).norm(),
                    plane_norm_dir.transpose() * ray_dir_p);
                // draw small cube
                if (radian_angle < 1 / 6 * 3.14 &&
                    intersection_point != Eigen::Vector3d::Zero()) {

                    vector<int> point_id = {
                        m_boundingbox_p
                            .triangle_plane[m_boundingbox_p.minTriangleIndex +
                            1] .first[0],
                        m_boundingbox_p
                            .triangle_plane[m_boundingbox_p.minTriangleIndex +
                            1] .first[1],
                        m_boundingbox_p
                            .triangle_plane[m_boundingbox_p.minTriangleIndex +
                            1] .first[2],
                        m_boundingbox_p
                            .triangle_plane[m_boundingbox_p.minTriangleIndex +
                            1] .first[3]};

                    auto point0 =
                        m_boundingbox_p.m_vertex_list_p
                            [m_boundingbox_p
                                 .triangle_plane
                                     [m_boundingbox_p.minTriangleIndex + 1]
                                 .first[0]];
                    auto point1 =
                        m_boundingbox_p.m_vertex_list_p
                            [m_boundingbox_p
                                 .triangle_plane
                                     [m_boundingbox_p.minTriangleIndex + 1]
                                 .first[1]];
                    auto point2 =
                        m_boundingbox_p.m_vertex_list_p
                            [m_boundingbox_p
                                 .triangle_plane
                                     [m_boundingbox_p.minTriangleIndex + 1]
                                 .first[2]];
                    auto point3 =
                        m_boundingbox_p.m_vertex_list_p
                            [m_boundingbox_p
                                 .triangle_plane
                                     [m_boundingbox_p.minTriangleIndex + 1]
                                 .first[3]];

                    if (point_id[0] == 6) {
                        if(intersection_point.x() < (point1[0] - point0[0])
                        / 3.0) { }else if(intersection_point.x() < (point1[0]
                        - point0[0]) * (2.0 / 3.0)) {

                        }else {

                        }
                    }else if(point_id[0] == 7) {
                        if(intersection_point.z() < (point0[2] - point1[2])
                        / 3.0) {

                        }else if(intersection_point.z() < (point0[2] -
                        point1[2]) * (2.0 / 3.0)) {

                        }else {

                        }
                    }else if(point_id[0] == 3) {
                        if(intersection_point.x() < (point0[0] - point1[0])
                        / 3.0) {

                        }else if(intersection_point.x() < (point0[0] -
                        point1[0]) * (2.0 / 3.0)) {

                        }else {

                        }
                    }else if(point_id[0] == 2 && point_id[1] == 6) {
                        if(intersection_point.z() < (point1[2] - point0[2])
                        / 3.0) {

                        }else if(intersection_point.z() < (point1[2] -
                        point0[2]) * (2.0 / 3.0)) {

                        }else {

                        }
                    }else if(point_id[0] == 2 && point_id[1] == 3) {
                        if(intersection_point.x() < (point1[0] - point0[0])
                        / 3.0) {

                        }else if(intersection_point.x() < (point1[0] -
                        point0[0]) * (2.0 / 3.0)) {

                        }else {

                        }
                    }
                }
                glEnd();
                glPopMatrix();
            } else {
            }
        }
#endif

    if (m_boundingbox_p.IsExist()) {
        Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d projectionMatrix = Eigen::Matrix4d::Identity();
        GLfloat view_model[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, view_model);
        GLdouble projection[16];
        glGetDoublev(GL_PROJECTION_MATRIX, projection);
        transformationMatrix << view_model[0], view_model[1], view_model[2],
            view_model[3], view_model[4], view_model[5], view_model[6],
            view_model[7], view_model[8], view_model[9], view_model[10],
            view_model[11], view_model[12], view_model[13], view_model[14],
            view_model[15];

        projectionMatrix << projection[0], projection[1], projection[2],
            projection[3], projection[4], projection[5], projection[6],
            projection[7], projection[8], projection[9], projection[10],
            projection[11], projection[12], projection[13], projection[14],
            projection[15];

        Eigen::Vector2d left_button;
        if (mp_scan_handler3d->GetLeftButtonPos(left_button)) {
            m_boundingbox_p.minTriangleIndex = -1;
            m_mouseState.SetMousePoseX(
                left_button.x() - m_scene.GetSceneBarWidth());
            m_mouseState.SetMousePoseY(
                ObjRecognition::CameraIntrinsic::GetInstance().Height() -
                left_button.y());
            m_scene.SetIsChangingPlane(false);
            m_boundingbox_p.minTriangleIndex = -1;
            m_boundingbox_p.SetChangeShapeOffset(0.0);

            Eigen::Vector3d ray = GetRay(
                m_mouseState.GetMousePoseX(), m_mouseState.GetMousePoseY(),
                transformationMatrix.transpose(), projectionMatrix.transpose());

            Eigen::Vector3d ray_dir = (ray - m_camera.GetCamPos()).normalized();
            // ray and campos: under world coords
            Eigen::Vector3d CameraPosition_w = m_camera.GetCamPos();
            // change to plane coords
            Eigen::Vector3d CameraPosition_p =
                Change2PlaneCoords(Twp, CameraPosition_w, true);
            Eigen::Vector3d ray_dir_p = Change2PlaneCoords(Twp, ray_dir, false);
            // m_boundingbox_p: under plane coords
            float minDistance = INT_MAX;
            for (size_t i = 0; i < m_boundingbox_p.GetAllTriangles().size();
                 i++) {
                Triangle thisTriangle = m_boundingbox_p.GetAllTriangles()[i];
                if (i == 0) {
                    auto pts = thisTriangle.GetVertex();
                }
                float t;
                float u;
                float v;

                // use plane coords to compute intersection
                if (IsIntersectWithTriangle(
                        CameraPosition_p, ray_dir_p,
                        thisTriangle.GetVertex()[0],
                        thisTriangle.GetVertex()[1],
                        thisTriangle.GetVertex()[2], t, u, v)) {

                    Eigen::Vector3d intersectionPoint =
                        (1 - u - v) * thisTriangle.GetVertex()[0] +
                        thisTriangle.GetVertex()[1] * u +
                        thisTriangle.GetVertex()[2] * v;

                    float distance =
                        pow((intersectionPoint(0) - CameraPosition_p[0]), 2) +
                        pow((intersectionPoint(1) - CameraPosition_p[1]), 2) +
                        pow((intersectionPoint(2) - CameraPosition_p[2]), 2);
                    if (distance < minDistance) {
                        minDistance = distance;
                        m_boundingbox_p.minTriangleIndex = i;
                    }
                }
            }

            // draw this intersection plane
            if (m_boundingbox_p.minTriangleIndex != -1) {
                m_scene.SetIsChangingPlane(true);

                // draw: under plane coords:
                glPushMatrix();
                Twp.Multiply();
                glBegin(GL_QUADS);
                glColor3f(0.0f, 0.0f, 1.0f);
                glVertex3fv(m_boundingbox_p.m_vertex_list_p
                                [m_boundingbox_p
                                     .triangle_plane
                                         [m_boundingbox_p.minTriangleIndex + 1]
                                     .first[0]]);
                glVertex3fv(m_boundingbox_p.m_vertex_list_p
                                [m_boundingbox_p
                                     .triangle_plane
                                         [m_boundingbox_p.minTriangleIndex + 1]
                                     .first[1]]);
                glVertex3fv(m_boundingbox_p.m_vertex_list_p
                                [m_boundingbox_p
                                     .triangle_plane
                                         [m_boundingbox_p.minTriangleIndex + 1]
                                     .first[2]]);
                glVertex3fv(m_boundingbox_p.m_vertex_list_p
                                [m_boundingbox_p
                                     .triangle_plane
                                         [m_boundingbox_p.minTriangleIndex + 1]
                                     .first[3]]);
                glEnd();
                glPopMatrix();
            } else {
                m_scene.SetIsChangingPlane(false);
            }
        }

        if (m_scene.GetIsChangingPlane()) {
            float offset = m_boundingbox_p.GetChangeShapeOffset();
            m_boundingbox_p.ChangePlane(
                m_boundingbox_p.minTriangleIndex / 2, offset);
            m_boundingbox_p.SetChangeShapeOffset(0.0);
        }
    }
}

Eigen::Matrix4d
ViewerAR::ChangeOpenglMatrix2EigenMatrix(pangolin::OpenGlMatrix opengl_matrix) {
    Eigen::Matrix4d eigen_matrix = Eigen::Matrix4d::Identity();
    eigen_matrix << opengl_matrix.m[0], opengl_matrix.m[1], opengl_matrix.m[2],
        opengl_matrix.m[3], opengl_matrix.m[4], opengl_matrix.m[5],
        opengl_matrix.m[6], opengl_matrix.m[7], opengl_matrix.m[8],
        opengl_matrix.m[9], opengl_matrix.m[10], opengl_matrix.m[11],
        opengl_matrix.m[12], opengl_matrix.m[13], opengl_matrix.m[14],
        opengl_matrix.m[15];
    eigen_matrix.transposeInPlace();
    return eigen_matrix;
}

Eigen::Vector3d ViewerAR::Change2PlaneCoords(
    pangolin::OpenGlMatrix Plane_wp, Eigen::Vector3d world_coords,
    bool is_point) {

    Eigen::Vector4d Pp_4, Pw_4;
    Eigen::Vector3d Pp;
    Eigen::Matrix4d Twp = ChangeOpenglMatrix2EigenMatrix(Plane_wp);
    // WARNING: maybe some numerical problem
    Eigen::Matrix4d Tpw = Twp.inverse();
    if (is_point) {
        Pw_4 << world_coords[0], world_coords[1], world_coords[2], 1.0f;
        Pp_4 = Tpw * Pw_4;
        Pp_4[0] = Pp_4[0] / Pp_4[3];
        Pp_4[1] = Pp_4[1] / Pp_4[3];
        Pp_4[2] = Pp_4[2] / Pp_4[3];
        Pp << Pp_4[0], Pp_4[1], Pp_4[2];
    } else {
        Pw_4 << world_coords[0], world_coords[1], world_coords[2], 0.0f;
        Pp_4 = Tpw * Pw_4;
        Pp << Pp_4[0], Pp_4[1], Pp_4[2];
    }
    return Pp;
}

void ViewerAR::DrawBoundingboxForSfM(
    const std::vector<Eigen::Vector3d> &boundingbox_w) {
    glColor3f(0.0f, 1.0f, 0.0f);
    glPointSize(4.0);
    glBegin(GL_LINES);

    Eigen::Vector3d point0 = boundingbox_w[0];
    Eigen::Vector3d point1 = boundingbox_w[1];
    Eigen::Vector3d point2 = boundingbox_w[2];
    Eigen::Vector3d point3 = boundingbox_w[3];
    Eigen::Vector3d point4 = boundingbox_w[4];
    Eigen::Vector3d point5 = boundingbox_w[5];
    Eigen::Vector3d point6 = boundingbox_w[6];
    Eigen::Vector3d point7 = boundingbox_w[7];

    glVertex3d(point0.x(), point0.y(), point0.z());
    glVertex3d(point1.x(), point1.y(), point1.z());

    glVertex3d(point5.x(), point5.y(), point5.z());
    glVertex3d(point1.x(), point1.y(), point1.z());

    glVertex3d(point5.x(), point5.y(), point5.z());
    glVertex3d(point4.x(), point4.y(), point4.z());

    glVertex3d(point0.x(), point0.y(), point0.z());
    glVertex3d(point4.x(), point4.y(), point4.z());

    glVertex3d(point2.x(), point2.y(), point2.z());
    glVertex3d(point3.x(), point3.y(), point3.z());

    glVertex3d(point3.x(), point3.y(), point3.z());
    glVertex3d(point7.x(), point7.y(), point7.z());

    glVertex3d(point7.x(), point7.y(), point7.z());
    glVertex3d(point6.x(), point6.y(), point6.z());

    glVertex3d(point6.x(), point6.y(), point6.z());
    glVertex3d(point2.x(), point2.y(), point2.z());

    glVertex3d(point6.x(), point6.y(), point6.z());
    glVertex3d(point4.x(), point4.y(), point4.z());

    glVertex3d(point7.x(), point7.y(), point7.z());
    glVertex3d(point5.x(), point5.y(), point5.z());

    glVertex3d(point3.x(), point3.y(), point3.z());
    glVertex3d(point1.x(), point1.y(), point1.z());

    glVertex3d(point0.x(), point0.y(), point0.z());
    glVertex3d(point2.x(), point2.y(), point2.z());
    glEnd();
}

void ViewerAR::DrawBoundingbox(const float x, const float y, const float z) {
    glColor3f(0.0, 1.0, 0);
    int i, j;
    glBegin(GL_LINES);
    for (i = 0; i < 12; ++i) {
        for (j = 0; j < 2; ++j) {
            glVertex3fv(
                m_boundingbox_p
                    .m_vertex_list_p[m_boundingbox_p.m_index_list[i][j]]);
        }
    }
    glEnd();

    // draw plane color
    glBegin(GL_QUADS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[0]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[1]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[5]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[4]);
    glEnd();

    glBegin(GL_QUADS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[2]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[3]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[7]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[6]);
    glEnd();

    glBegin(GL_QUADS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[2]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[0]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[4]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[6]);
    glEnd();

    glBegin(GL_QUADS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[3]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[7]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[5]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[1]);
    glEnd();

    glBegin(GL_QUADS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[6]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[7]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[5]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[4]);
    glEnd();

    glBegin(GL_QUADS);
    glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[2]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[3]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[1]);
    glVertex3fv(m_boundingbox_p.m_vertex_list_p[0]);
    glEnd();
}

void ViewerAR::DrawPlane(int ndivs, float ndivsize) {
    // Plane parallel to x-z at origin with normal -y
    const float minx = -ndivs * ndivsize;
    const float minz = -ndivs * ndivsize;
    const float maxx = ndivs * ndivsize;
    const float maxz = ndivs * ndivsize;

    glLineWidth(2);
    glColor3f(0.7f, 0.7f, 1.0f);
    glBegin(GL_LINES);

    for (int n = 0; n <= 2 * ndivs; n++) {
        glVertex3f(minx + ndivsize * n, 0, minz);
        glVertex3f(minx + ndivsize * n, 0, maxz);
        glVertex3f(minx, 0, minz + ndivsize * n);
        glVertex3f(maxx, 0, minz + ndivsize * n);
    }

    glEnd();
}

void ViewerAR::DrawTrackedPoints(
    const std::vector<cv::KeyPoint> &vKeys, const std::vector<MapPoint *> &vMPs,
    cv::Mat &im) {
    const int N = vKeys.size();

    for (int i = 0; i < N; i++) {
        if (vMPs[i]) {
            cv::circle(im, vKeys[i].pt, 1, cv::Scalar(0, 255, 0), -1);
        }
    }
}

Plane *ViewerAR::DetectPlane(
    const cv::Mat Tcw, const std::vector<MapPoint *> &vMPs,
    const int iterations) {
    // Retrieve 3D points
    vector<cv::Mat> vPoints;
    vPoints.reserve(vMPs.size());
    vector<MapPoint *> vPointMP;
    vPointMP.reserve(vMPs.size());

    for (size_t i = 0; i < vMPs.size(); i++) {
        MapPoint *pMP = vMPs[i];
        if (pMP) {
            if (pMP->Observations() > 5) {
                vPoints.push_back(pMP->GetWorldPos());
                vPointMP.push_back(pMP);
            }
        }
    }

    const int N = vPoints.size();

    if (N < 50)
        return NULL;

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for (int i = 0; i < N; i++) {
        vAllIndices.push_back(i);
    }

    float bestDist = 1e10;
    vector<float> bestvDist;

    // RANSAC
    for (int n = 0; n < iterations; n++) {
        vAvailableIndices = vAllIndices;

        cv::Mat A(3, 4, CV_32F);
        A.col(3) = cv::Mat::ones(3, 1, CV_32F);

        // Get min set of points
        for (short i = 0; i < 3; ++i) {
            int randi =
                DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            A.row(i).colRange(0, 3) = vPoints[idx].t();

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        cv::Mat u, w, vt;
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        const float a = vt.at<float>(3, 0);
        const float b = vt.at<float>(3, 1);
        const float c = vt.at<float>(3, 2);
        const float d = vt.at<float>(3, 3);

        vector<float> vDistances(N, 0);

        const float f = 1.0f / sqrt(a * a + b * b + c * c + d * d);

        for (int i = 0; i < N; i++) {
            vDistances[i] =
                fabs(
                    vPoints[i].at<float>(0) * a + vPoints[i].at<float>(1) * b +
                    vPoints[i].at<float>(2) * c + d) *
                f;
        }

        vector<float> vSorted = vDistances;
        sort(vSorted.begin(), vSorted.end());

        int nth = max((int)(0.2 * N), 20);
        const float medianDist = vSorted[nth];

        if (medianDist < bestDist) {
            bestDist = medianDist;
            bestvDist = vDistances;
        }
    }

    // Compute threshold inlier/outlier
    const float th = 1.4 * bestDist;
    vector<bool> vbInliers(N, false);
    int nInliers = 0;
    for (int i = 0; i < N; i++) {
        if (bestvDist[i] < th) {
            nInliers++;
            vbInliers[i] = true;
        }
    }

    vector<MapPoint *> vInlierMPs(nInliers, NULL);
    int nin = 0;
    for (int i = 0; i < N; i++) {
        if (vbInliers[i]) {
            vInlierMPs[nin] = vPointMP[i];
            nin++;
        }
    }

    return new Plane(vInlierMPs, Tcw);
}

Plane::Plane(const std::vector<MapPoint *> &vMPs, const cv::Mat &Tcw)
    : mvMPs(vMPs), mTcw(Tcw.clone()) {
    rang = -3.14f / 2 + ((float)rand() / RAND_MAX) * 3.14f;
    Recompute();
}

void Plane::Recompute() {
    const int N = mvMPs.size();

    // Recompute plane with all points
    cv::Mat A = cv::Mat(N, 4, CV_32F);
    A.col(3) = cv::Mat::ones(N, 1, CV_32F);

    o = cv::Mat::zeros(3, 1, CV_32F);

    int nPoints = 0;
    for (int i = 0; i < N; i++) {
        MapPoint *pMP = mvMPs[i];
        if (!pMP->isBad()) {
            cv::Mat Xw = pMP->GetWorldPos();
            o += Xw;
            A.row(nPoints).colRange(0, 3) = Xw.t();
            nPoints++;
        }
    }
    A.resize(nPoints);

    cv::Mat u, w, vt;
    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    float a = vt.at<float>(3, 0);
    float b = vt.at<float>(3, 1);
    float c = vt.at<float>(3, 2);

    o = o * (1.0f / nPoints);
    const float f = 1.0f / sqrt(a * a + b * b + c * c);

    // Compute XC just the first time
    if (XC.empty()) {
        cv::Mat Oc = -mTcw.colRange(0, 3).rowRange(0, 3).t() *
                     mTcw.rowRange(0, 3).col(3);
        XC = Oc - o;
    }

    if ((XC.at<float>(0) * a + XC.at<float>(1) * b + XC.at<float>(2) * c) > 0) {
        a = -a;
        b = -b;
        c = -c;
    }

    const float nx = a * f;
    const float ny = b * f;
    const float nz = c * f;

    n = (cv::Mat_<float>(3, 1) << nx, ny, nz);

    cv::Mat up = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);

    cv::Mat v = up.cross(n);
    const float sa = cv::norm(v);
    const float ca = up.dot(n);
    const float ang = atan2(sa, ca);
    Tpw = cv::Mat::eye(4, 4, CV_32F);

    Tpw.rowRange(0, 3).colRange(0, 3) =
        ExpSO3(v * ang / sa) * ExpSO3(up * rang);
    o.copyTo(Tpw.col(3).rowRange(0, 3));
    Tools::ChangeCV44ToGLMatrixFloat(Tpw, glTpw);
}

Plane::Plane(
    const float &nx, const float &ny, const float &nz, const float &ox,
    const float &oy, const float &oz) {
    n = (cv::Mat_<float>(3, 1) << nx, ny, nz);
    o = (cv::Mat_<float>(3, 1) << ox, oy, oz);

    cv::Mat up = (cv::Mat_<float>(3, 1) << 0.0f, 1.0f, 0.0f);

    cv::Mat v = up.cross(n);
    const float s = cv::norm(v);
    const float c = up.dot(n);
    const float a = atan2(s, c);
    Tpw = cv::Mat::eye(4, 4, CV_32F);
    const float rang = -3.14f / 2 + ((float)rand() / RAND_MAX) * 3.14f;
    Tpw.rowRange(0, 3).colRange(0, 3) = ExpSO3(v * a / s) * ExpSO3(up * rang);
    o.copyTo(Tpw.col(3).rowRange(0, 3));
    Tools::ChangeCV44ToGLMatrixFloat(Tpw, glTpw);
}

} // namespace ORB_SLAM3
