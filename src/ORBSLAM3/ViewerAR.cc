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
#include "include/ORBSLAM3/ViewerAR.h"
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <cstdlib>
#include "ORBSLAM3/ScannerStruct/Struct.h"
#include "ORBSLAM3/ViewerCommon.h"
#include "include/Tools.h"
#include <glm/gtc/type_ptr.hpp>
#include "mode.h"
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
    m_is_debug_mode = false;
    m_is_stop = false;
    switch_window_flag = false;
    RegistEvents();
}

void ViewerAR::decrease_shape() {
    m_boundingbox_p.SetChangeShapeOffset(-0.05);
}

void ViewerAR::increase_shape() {
    m_boundingbox_p.SetChangeShapeOffset(0.05);
}

void ViewerAR::up_move() {
    m_boundingbox_p.MoveObject(-0.03, 1);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::down_move() {
    m_boundingbox_p.MoveObject(+0.03, 1);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::left_move() {
    m_boundingbox_p.MoveObject(-0.03, 0);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::right_move() {
    m_boundingbox_p.MoveObject(0.03, 0);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::front_move() {
    m_boundingbox_p.MoveObject(0.03, 2);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

void ViewerAR::back_move() {
    m_boundingbox_p.MoveObject(-0.03, 2);
    m_boundingbox_p.SetChangeShapeOffset(0.0);
}

std::vector<Eigen::Vector3d>
ViewerAR::ComputeBoundingbox_W(const pangolin::OpenGlMatrix &Twp_opengl) {
    std::vector<Eigen::Vector3d> boundingbox_w;
    for (size_t i = 0; i < 8; i++) {
        Eigen::Vector3d point_p = Eigen::Vector3d(
            m_boundingbox_p.m_vertex_list_p[i][0],
            m_boundingbox_p.m_vertex_list_p[i][1],
            m_boundingbox_p.m_vertex_list_p[i][2]);
        // TODO(zhangye): transpose???
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
    const std::vector<double> &boundingbox_w_corner) {
    mpMapDrawer->DrawMapPoints_SuperPoint(boundingbox_w_corner);
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
        m_boundingbox_corner = std::vector<double>();
    }

    if (!vpPlane.empty()) {
        mappoint_num_inbbx = 0;
        Plane *pPlane = vpPlane[0];

        vector<Eigen::Vector3d> boundingbox_w =
            ComputeBoundingbox_W(pPlane->glTpw);
        double bbx_xmin = 10000;
        double bbx_ymin = 10000;
        double bbx_zmin = 10000;
        double bbx_xmax = -10000;
        double bbx_ymax = -10000;
        double bbx_zmax = -10000;

        for (int i = 0; i < boundingbox_w.size(); i++) {
            if (boundingbox_w[i](0) < bbx_xmin) {
                bbx_xmin = boundingbox_w[i](0);
            }
            if (boundingbox_w[i](0) > bbx_xmax) {
                bbx_xmax = boundingbox_w[i](0);
            }

            if (boundingbox_w[i](1) < bbx_ymin) {
                bbx_ymin = boundingbox_w[i](1);
            }
            if (boundingbox_w[i](1) > bbx_ymax) {
                bbx_ymax = boundingbox_w[i](1);
            }

            if (boundingbox_w[i](2) < bbx_zmin) {
                bbx_zmin = boundingbox_w[i](2);
            }
            if (boundingbox_w[i](2) > bbx_zmax) {
                bbx_zmax = boundingbox_w[i](2);
            }
        }

        m_boundingbox_corner = {bbx_xmin, bbx_ymin, bbx_zmin,
                                bbx_xmax, bbx_ymax, bbx_zmax};
        std::vector<Map *> saved_map = GetAllMapPoints();
        for (Map *pMi : saved_map) {
            for (MapPoint *pMPi : pMi->GetAllMapPoints()) {
                cv::Mat tmpPos = pMPi->GetWorldPos();
                // condition???
                if (tmpPos.at<float>(0) >= bbx_xmin &&
                    tmpPos.at<float>(0) <= bbx_xmax &&
                    tmpPos.at<float>(1) >= bbx_ymin &&
                    tmpPos.at<float>(1) <= bbx_ymax &&
                    tmpPos.at<float>(2) >= bbx_zmin &&
                    tmpPos.at<float>(2) <= bbx_zmax) {
                    mappoint_num_inbbx++;
                }
            }
        }
    }
}

void ViewerAR::ProjectMapPointInImage(
    const cv::Mat &Tcw, const std::vector<double> &bbx,
    std::vector<cv::KeyPoint> &keypoints_outbbx,
    std::vector<cv::KeyPoint> &keypoints_inbbx) {

    keypoints_outbbx.clear();
    keypoints_inbbx.clear();
    if (Tcw.empty()) {
        VLOG(0) << "Tcw is null";
        return;
    }

    std::vector<Map *> saved_map = GetAllMapPoints();
    for (Map *pMi : saved_map) {
        for (MapPoint *pMPi : pMi->GetAllMapPoints()) {
            cv::Mat tmpPos = pMPi->GetWorldPos();
            Eigen::Vector3d pos_w;
            Eigen::Matrix4d Tcw_eigen;
            cv2eigen(tmpPos, pos_w);
            cv2eigen(Tcw, Tcw_eigen);
            Eigen::Vector4d pos_w_4 =
                Eigen::Vector4d(pos_w(0), pos_w(1), pos_w(2), 1.0);
            Eigen::Vector4d pos_c = Tcw_eigen * pos_w_4;
            Eigen::Vector3d pos_c_3 = Eigen::Vector3d(
                pos_c(0) / pos_c(3), pos_c(1) / pos_c(3), pos_c(2) / pos_c(3));
            // TODO(zhangye): distort???
            Eigen::Vector3d pos_i =
                ObjRecognition::CameraIntrinsic::GetInstance().GetEigenK() *
                pos_c_3;
            Eigen::Vector2d pos_i_2 =
                Eigen::Vector2d(pos_i(0) / pos_i(2), pos_i(1) / pos_i(2));
            // change cv::point2f to cv::keypoint
            cv::KeyPoint kp(cv::Point2f(pos_i_2(0), pos_i_2(1)), 8);

            if (!bbx.empty() && tmpPos.at<float>(0) >= bbx[0] &&
                tmpPos.at<float>(0) <= bbx[3] &&
                tmpPos.at<float>(1) >= bbx[1] &&
                tmpPos.at<float>(1) <= bbx[4] &&
                tmpPos.at<float>(2) >= bbx[2] &&
                tmpPos.at<float>(2) <= bbx[5]) {
                keypoints_inbbx.emplace_back(kp);
            } else {
                keypoints_outbbx.emplace_back(kp);
            }
        }
    }
}

void ViewerAR::SwitchWindow() {
    std::cout << "switch flag" << std::endl;
    if (switch_window_flag) {
        d_cam_scan.show = false;
        d_cam_SfM = pangolin::CreateDisplay()
                        .SetBounds(
                            0.0, 1.0f,
                            pangolin::Attach::Pix(m_scene.GetSceneBarWidth()),
                            1.0, (float)im_scan.cols / im_scan.rows)
                        .SetHandler(new pangolin::Handler3D(s_cam_SfM));
        d_cam_SfM.show = true;
    } else {
        d_cam_SfM.show = false;
        d_cam_scan =
            pangolin::CreateDisplay()
                .SetBounds(
                    0, 1.0f, pangolin::Attach::Pix(m_scene.GetSceneBarWidth()),
                    1.0f, (float)im_scan.cols / im_scan.rows)
                .SetLock(pangolin::LockLeft, pangolin::LockTop)
                .SetHandler(mp_handler3d.get());
        d_cam_scan.show = true;
    }
    switch_window_flag = !switch_window_flag;
}

void ViewerAR::DrawScanInit(int w, int h) {
    pangolin::CreatePanel("menu").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(m_scene.GetSceneBarWidth()));
    menu_clear = std::make_unique<pangolin::Var<bool>>(
        "menu.Choose Another Plane", false, false);
    menu_setboundingbox =
        std::make_unique<pangolin::Var<bool>>("menu.Set a BBX", false, false);
    menu_fixBBX =
        std::make_unique<pangolin::Var<bool>>("menu.Fix BBX", false, false);
    menu_stop =
        std::make_unique<pangolin::Var<bool>>("menu.Finish Scan", false, false);

    menu_debug =
        std::make_unique<pangolin::Var<bool>>("menu.Debug", false, false);
    m_boundingbox_p.SetSize(0.05);
    // plane grid number
    menu_ngrid =
        std::make_unique<pangolin::Var<int>>("menu. Grid Elements", 3, 1, 10);
    menu_drawTrackedpoints =
        std::make_unique<pangolin::Var<bool>>("menu.Draw Points", false, true);
    menu_drawMappoints = std::make_unique<pangolin::Var<bool>>(
        "menu.Draw Proj Mappoints", false, true);

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
    mp_handler3d.reset(new MapHandler3D(s_cam_scan));

    d_cam_scan =
        pangolin::Display("image")
            .SetBounds(
                0, 1.0f, pangolin::Attach::Pix(m_scene.GetSceneBarWidth()),
                1.0f, (float)w / h)
            .SetLock(pangolin::LockLeft, pangolin::LockTop)
            .SetHandler(mp_handler3d.get());
}

void ViewerAR::DrawSfMInit(int w, int h) {
    s_cam_SfM = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.7, -3.5, 0, 0, 0, 0.0, -1.0, 0.0));
}

void ViewerAR::Draw(int w, int h) {
    pangolin::GlTexture imageTexture(
        w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    pangolin::OpenGlMatrixSpec P = pangolin::ProjectionMatrixRDF_TopLeft(
        w, h, fx, fy, cx, cy, 0.001, 1000);

    vector<Plane *> vpPlane;

    while (true) {
        m_is_debug_mode = *menu_debug;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!switch_window_flag) {
            d_cam_scan.show = true;
            // Activate m_camera view
            d_cam_scan.Activate(s_cam_scan);
            glColor3f(1.0, 1.0, 1.0);
            // Get last image and its computed pose from SLAM
            GetImagePose(im_scan, Tcw_scan, status_scan, vKeys_scan, vMPs_scan);

            if (!Tcw_scan.empty()) {
                cv::Mat Rwc = Tcw_scan.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * Tcw_scan.rowRange(0, 3).col(3);
                // set m_camera position
                m_camera.SetCamPos(
                    twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));
            }

            // get mappoint num in boundingbox
            int current_mappoint_num_inbbx = 0;
            GetCurrentMapPointInBBX(
                vpPlane, *menu_setboundingbox, current_mappoint_num_inbbx);

            // Add text to image
            PrintStatus(status_scan, current_mappoint_num_inbbx, im_scan);
            cv::cvtColor(im_scan, im_scan, CV_GRAY2RGB);
            if (*menu_drawTrackedpoints)
                DrawTrackedPoints(vKeys_scan, vMPs_scan, im_scan);

            // Draw image
            if (*menu_drawMappoints) {
                // draw mappoints in and out the boundingbox
                std::vector<cv::KeyPoint> keypoints_outbbx;
                std::vector<cv::KeyPoint> keypoints_inbbx;
                ProjectMapPointInImage(
                    Tcw_scan, m_boundingbox_corner, keypoints_outbbx,
                    keypoints_inbbx);
                cv::drawKeypoints(
                    im_scan, keypoints_outbbx, im_scan,
                    cv::Scalar(255, 0, 255));
                cv::drawKeypoints(
                    im_scan, keypoints_inbbx, im_scan, cv::Scalar(0, 255, 255));
            }
            DrawImageTexture(imageTexture, im_scan);

            ObjRecognition::GlobalOcvViewer::DrawAllView();
            glClear(GL_DEPTH_BUFFER_BIT);

            // Load m_camera projection
            glMatrixMode(GL_PROJECTION);
            P.Load();

            // load model view matrix
            glMatrixMode(GL_MODELVIEW);

            // Load m_camera pose  set opengl coords, same as slam coords
            // view matrix Tcw
            LoadCameraPose(Tcw_scan);

            if (*menu_clear) {
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
                *menu_clear = false;
            }
            // Draw virtual things
            // can only insert cube when slam state is fine
            if (status_scan == 2) {
                Plane *pPlane = DetectPlane(Tcw_scan, vMPs_scan, 50);
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
                    if (*menu_fixBBX) {
                        // get m_boundingbox_w
                        m_boundingbox_w = ComputeBoundingbox_W(pPlane->glTpw);
                        m_is_fix = true;
                        *menu_fixBBX = false;
                    }
                    if (*menu_stop) {
                        m_is_stop = true;
                        *menu_stop = false;

#ifdef SUPERPOINT
                        // change to sfm panglin dispaly
                        switch_window_flag = !switch_window_flag;
                        d_cam_scan.show = false;
                        d_cam_SfM =
                            pangolin::CreateDisplay()
                                .SetBounds(
                                    0.0, 1.0,
                                    pangolin::Attach::Pix(
                                        m_scene.GetSceneBarWidth()),
                                    1.0, (float)im_scan.cols / im_scan.rows)
                                .SetHandler(new pangolin::Handler3D(s_cam_SfM));
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
                    if (*menu_setboundingbox) {
                        DrawBoundingbox();
                    }

                    // Draw grid plane
                    DrawPlane(*menu_ngrid, 0.05);
                    glPopMatrix();
                }
            }

        } else {
#ifdef SUPERPOINT
            d_cam_SfM.show = true;
            d_cam_SfM.Activate(s_cam_SfM);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            pangolin::glDrawAxis(0.6f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
            pangolin::glDraw_z0(0.5f, 100);

            DrawMapPoints_SuperPoint(m_boundingbox_corner);
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
        GetImagePose(im_scan, Tcw_scan, status_scan, vKeys_scan, vMPs_scan);
        if (im_scan.empty())
            cv::waitKey(mT);
        else {
            w = im_scan.cols;
            h = im_scan.rows;
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
    const Eigen::Matrix4d &transformationMatrix,
    const Eigen::Matrix4d &projectionMatrix) {
    float x =
        (2.0f * m_mouseState.GetMousePoseX()) / m_scene.GetSceneWidth() - 1.0f;
    float y =
        1.0f - (2.0f * m_mouseState.GetMousePoseY()) / m_scene.GetSceneHeight();
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
    if (m_boundingbox_p.IsExist()) {
        Eigen::Vector2d left_button;
        if (mp_handler3d->GetLeftButtonPos(left_button)) {
            m_boundingbox_p.minTriangleIndex = -1;
            m_mouseState.SetMousePoseX(
                left_button.x() - m_scene.GetSceneBarWidth());
            m_mouseState.SetMousePoseY(
                ObjRecognition::CameraIntrinsic::GetInstance().Height() -
                left_button.y());
            m_scene.SetIsChangingPlane(false);
            m_boundingbox_p.minTriangleIndex = -1;
            m_boundingbox_p.SetChangeShapeOffset(0.0);

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

            Eigen::Vector3d ray = GetRay(
                transformationMatrix.transpose(), projectionMatrix.transpose());

            Eigen::Vector3d ray_dir = (ray - m_camera.GetCamPos()).normalized();
            ray = m_camera.GetCamPos() + ray_dir * 5.0f;

            //            glBegin(GL_LINES);
            //            glColor3f(1.0, 1.0, 0.0);
            //            glVertex3f(
            //                m_camera.GetCamPos()[0], m_camera.GetCamPos()[1],
            //                m_camera.GetCamPos()[2]);
            //            glVertex3f(ray(0), ray(1), ray(2));
            //            glEnd();

            // ray and campos: under world coords
            Eigen::Vector3d CameraPosition_w = m_camera.GetCamPos();
            //            Eigen::Vector3d RayDir_w = ray;
            Eigen::Vector3d RayDir_w = ray_dir;
            // change to plane coords
            Eigen::Vector3d CameraPosition_p =
                Change2PlaneCoords(Twp, CameraPosition_w, true);
            Eigen::Vector3d RayDir_p = Change2PlaneCoords(Twp, RayDir_w, false);

            // m_boundingbox_p: under plane coords
            //      intersection
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
                        CameraPosition_p, RayDir_p, thisTriangle.GetVertex()[0],
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
                glVertex3fv(
                    m_boundingbox_p.m_vertex_list_p
                        [m_boundingbox_p.triangle_plane
                             [m_boundingbox_p.minTriangleIndex + 1][0]]);
                glVertex3fv(
                    m_boundingbox_p.m_vertex_list_p
                        [m_boundingbox_p.triangle_plane
                             [m_boundingbox_p.minTriangleIndex + 1][1]]);
                glVertex3fv(
                    m_boundingbox_p.m_vertex_list_p
                        [m_boundingbox_p.triangle_plane
                             [m_boundingbox_p.minTriangleIndex + 1][2]]);
                glVertex3fv(
                    m_boundingbox_p.m_vertex_list_p
                        [m_boundingbox_p.triangle_plane
                             [m_boundingbox_p.minTriangleIndex + 1][3]]);
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
