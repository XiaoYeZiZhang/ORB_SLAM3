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

#include <pangolin/pangolin.h>
#include <mutex>
#include <glog/logging.h>
#include "Visualizer/GlobalImageViewer.h"
#include "ORBSLAM3/Viewer.h"
#include "ORBSLAM3/ViewerCommon.h"
#include "Tools.h"
#include "mode.h"
#include "src/ORBSLAM3/GLModel/model.h"
#include <opencv2/core/eigen.hpp>

namespace ORB_SLAM3 {

std::pair<Eigen::Matrix3d, Eigen::Vector3d>
cal_trans(std::vector<Eigen::Vector3d> boundingbox);
std::pair<Eigen::Vector3d, Eigen::Vector3d>
get_bound(const vector<Eigen::Vector3d> &boundingbox);

Model *textModel;

Viewer::Viewer(
    System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,
    Tracking *pTracking, const string &strSettingPath)
    : both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer), mpTracker(pTracking), mbFinishRequested(false),
      mbFinished(true), mbStopped(true), mbStopRequested(false) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    bool is_correct = ParseViewerParamFile(fSettings);

    if (!is_correct) {
        std::cerr << "**ERROR in the config file, the format is not correct**"
                  << std::endl;
        try {
            throw - 1;
        } catch (exception &e) {
        }
    }

    mbStopTrack = false;
    m_switch_window_flag = 0;
    m_is_stop = false;
    m_image_width = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    m_image_height = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    m_image_texture = pangolin::GlTexture();
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d>
cal_trans(std::vector<Eigen::Vector3d> boundingbox) {
    Eigen::Matrix3d rot;
    Eigen::Vector3d x, y, z, offset;
#ifdef OBJECT_TOY
    z = boundingbox[1] - boundingbox[0];
    x = boundingbox[4] - boundingbox[0];
#endif
#ifdef OBJECT_BAG
    z = boundingbox[1] - boundingbox[0];
    x = -boundingbox[4] - boundingbox[0];
#endif
#ifdef OBJECT_BOX
    x = boundingbox[1] - boundingbox[0];
    z = boundingbox[4] - boundingbox[0];
#endif
    y = boundingbox[2] - boundingbox[0];
    rot.block<3, 1>(0, 0) = x;
    rot.block<3, 1>(0, 1) = y;
    rot.block<3, 1>(0, 2) = z;
    double ratio = 3.0 / 4;
    offset = (boundingbox[0] + boundingbox[7]) / 2;
    //    offset = offset - y/2;
    offset = offset - ratio * y;
    return std::make_pair(rot, offset);
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings) {
    bool b_miss_params = false;

    float fps = fSettings["Camera.fps"];
    if (fps < 1)
        fps = 30;
    mT = 1e3 / fps;

    cv::FileNode node = fSettings["Camera.width"];
    if (!node.empty()) {
        mImageWidth = node.real();
    } else {
        std::cerr
            << "*Camera.width parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if (!node.empty()) {
        mImageHeight = node.real();
    } else {
        std::cerr
            << "*Camera.height parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointX"];
    if (!node.empty()) {
        mViewpointX = node.real();
    } else {
        std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointY"];
    if (!node.empty()) {
        mViewpointY = node.real();
    } else {
        std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointZ"];
    if (!node.empty()) {
        mViewpointZ = node.real();
    } else {
        std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointF"];
    if (!node.empty()) {
        mViewpointF = node.real();
    } else {
        std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void Viewer::SetObjectRecognitionPose(
    const Eigen::Matrix3d &Row, const Eigen::Vector3d &tow) {
    m_Row = Row;
    m_tow = tow;
}

void Viewer::DrawObjRecognitionInit() {
    // define projection and initial movelview matrix: default
    m_s_cam_objRecognition = pangolin::OpenGlRenderState();
    m_image_texture = pangolin::GlTexture(
        m_image_width, m_image_height, GL_RGB, false, 0, GL_RGB,
        GL_UNSIGNED_BYTE);
}

void Viewer::DrawDetectorInit() {
    // define projection and initial movelview matrix: default
    // Define Camera Render Object (for view / scene browsing)
    m_s_cam_detector = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(
            1024, 768, mViewpointF, mViewpointF, 512, 389, 0.01, 10000),
        pangolin::ModelViewLookAt(
            mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
}

void Viewer::DrawSLAMInit() {
    pangolin::CreatePanel("menu").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(200));
    m_menu_follow_camera = std::make_unique<pangolin::Var<bool>>(
        "menu.Follow Camera", false, true);
    m_menu_cam_view =
        std::make_unique<pangolin::Var<bool>>("menu.Camera View", false, false);
    m_menu_top_view =
        std::make_unique<pangolin::Var<bool>>("menu.Top View", false, false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    m_menu_show_points =
        std::make_unique<pangolin::Var<bool>>("menu.Show Points", true, true);
    m_menu_show_keyframes = std::make_unique<pangolin::Var<bool>>(
        "menu.Show KeyFrames", true, true);
    m_menu_show_graph =
        std::make_unique<pangolin::Var<bool>>("menu.Show Graph", false, true);
    m_menu_show_camera_trajectory = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Camera trajectory", true, true);
    m_menu_show_3DObject =
        std::make_unique<pangolin::Var<bool>>("menu.Show 3DObject", true, true);
    m_menu_show_matched_3DObject = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Matched 3DObject", true, true);
    m_menu_show_inertial_graph = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Inertial Graph", true, true);
    m_menu_reset =
        std::make_unique<pangolin::Var<bool>>("menu.Reset", false, false);
    m_menu_stepbystep = std::make_unique<pangolin::Var<bool>>(
        "menu.Step By Step", false, true); // false, true
    m_menu_step =
        std::make_unique<pangolin::Var<bool>>("menu.Step", false, false);
    m_menu_stop =
        std::make_unique<pangolin::Var<bool>>("menu.Stop", false, false);

    // Define Camera Render Object (for view / scene browsing)
    m_s_cam_slam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(
            1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(
            mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    m_d_cam_slam =
        pangolin::CreateDisplay()
            .SetBounds(
                0.0, 1.0, pangolin::Attach::Pix(200), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(m_s_cam_slam));
}

void Viewer::Draw3dText() {
    textModel->Draw();
}

std::pair<Eigen::Vector3d, Eigen::Vector3d>
get_bound(const vector<Eigen::Vector3d> &boundingbox) {
    Eigen::Vector3d min_bound(1e9, 1e9, 1e9), max_bound(-1e9, -1e9, -1e9);
    for (auto bbox : boundingbox) {
        min_bound.x() = std::min(min_bound.x(), bbox.x());
        min_bound.y() = std::min(min_bound.y(), bbox.y());
        min_bound.z() = std::min(min_bound.z(), bbox.z());

        max_bound.x() = std::max(max_bound.x(), bbox.x());
        max_bound.y() = std::max(max_bound.y(), bbox.y());
        max_bound.z() = std::max(max_bound.z(), bbox.z());
    }
    return std::make_pair(min_bound, max_bound);
}

void Viewer::DrawBoundingboxInImage(
    const vector<Eigen::Vector3d> &boundingbox) {
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(4.0);
    glBegin(GL_LINES);

    Eigen::Vector3d point0 = boundingbox[0];
    Eigen::Vector3d point1 = boundingbox[1];
    Eigen::Vector3d point2 = boundingbox[2];
    Eigen::Vector3d point3 = boundingbox[3];
    Eigen::Vector3d point4 = boundingbox[4];
    Eigen::Vector3d point5 = boundingbox[5];
    Eigen::Vector3d point6 = boundingbox[6];
    Eigen::Vector3d point7 = boundingbox[7];

    //    auto bound = get_bound(boundingbox);
    //    std::cout << "min_bound" << bound.first << std::endl;
    //    std::cout << "max_bound" << bound.second << std::endl;

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

void Viewer::DrawPointCloudInImage(
    const std::vector<Eigen::Vector3d> &pointcloud_pos) {
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_POINTS);
    for (auto pos : pointcloud_pos) {
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();
}

void Viewer::ShowConnectedMapPoints() {

    if (m_pointCloud_model) {
        Eigen::Isometry3f T = Eigen::Isometry3f::Identity();

        for (const auto &mappoint : m_pointCloud_model->GetPointClouds()) {
            if (m_pointCloud_model->m_associated_mappoints_id.count(
                    mappoint->GetID())) {
                glPointSize(3.0);
                glColor3f(0.0f, 1.0f, 0.0f);
            } else {
                glPointSize(2.0);
                glColor3f(1.0f, 0.0f, 0.0f);
            }
            glBegin(GL_POINTS);
            Eigen::Vector3f p = mappoint->GetPose().cast<float>();
            p = T.inverse() * p;
            glVertex3f(p.x(), p.y(), p.z());
            glEnd();
        }
    }
}

void Viewer::ShowConnectedKeyframes() {
    if (m_pointCloud_model) {
        float cam_size = 0.1f;
        for (const auto &keyframe : m_pointCloud_model->GetKeyFrames()) {
            if (m_pointCloud_model->m_associated_keyframes_id.count(
                    keyframe->GetID())) {
                glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
                Eigen::Matrix3d Rcw;
                Eigen::Vector3d tcw;
                keyframe->GetPose(Rcw, tcw);

                Eigen::Matrix3d Rwc = Rcw.transpose();
                Eigen::Vector3d twc = -Rwc * tcw;

                Eigen::Vector3f p = twc.cast<float>();
                Eigen::Quaterniond Qwc(Rwc);

                const Eigen::Vector3f &center = p;
                Eigen::Quaternionf q = Qwc.cast<float>();
                const float length = cam_size;
                Eigen::Vector3f m_cam[5] = {
                    Eigen::Vector3f(0.0f, 0.0f, 0.0f),
                    Eigen::Vector3f(-length, -length, length),
                    Eigen::Vector3f(-length, length, length),
                    Eigen::Vector3f(length, length, length),
                    Eigen::Vector3f(length, -length, length)};

                for (int i = 0; i < 5; ++i)
                    m_cam[i] = q * m_cam[i] + center;
                glBegin(GL_LINE_LOOP);
                glVertex3fv(m_cam[0].data());
                glVertex3fv(m_cam[1].data());
                glVertex3fv(m_cam[4].data());
                glVertex3fv(m_cam[3].data());
                glVertex3fv(m_cam[2].data());
                glEnd();
                glBegin(GL_LINES);
                glVertex3fv(m_cam[0].data());
                glVertex3fv(m_cam[3].data());
                glEnd();
                glBegin(GL_LINES);
                glVertex3fv(m_cam[0].data());
                glVertex3fv(m_cam[4].data());
                glEnd();
                glBegin(GL_LINES);
                glVertex3fv(m_cam[1].data());
                glVertex3fv(m_cam[2].data());
                glEnd();
            } else {
                glColor4f(1.0f, 0.0f, 1.0f, 0.1f);
            }
        }
    }
}

void Viewer::Draw() {
    mbFinished = false;
    mbStopped = false;
    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    pangolin::OpenGlMatrix
        Twwp; // Oriented with g in the z axis, but y and x from camera
    Twwp.SetIdentity();
    cv::namedWindow("ORB-SLAM3: Current Frame");

    bool bFollow = true;
    bool bStepByStep = false;
    bool bCameraView = true;

    if (mpTracker->mSensor == mpSystem->MONOCULAR ||
        mpTracker->mSensor == mpSystem->STEREO ||
        mpTracker->mSensor == mpSystem->RGBD) {
        *m_menu_show_graph = true;
    }

    if (mpTracker->m_objRecognition_mode_) {
        *m_menu_show_3DObject = true;
        *m_menu_show_camera_trajectory = true;
        *m_menu_show_matched_3DObject = true;
    }

#ifdef OBJECTRECOGNITION
    // for objrecognition
    float fx = ObjRecognition::CameraIntrinsic::GetInstance().FX();
    float fy = ObjRecognition::CameraIntrinsic::GetInstance().FY();
    float cx = ObjRecognition::CameraIntrinsic::GetInstance().CX();
    float cy = ObjRecognition::CameraIntrinsic::GetInstance().CY();

    pangolin::OpenGlMatrixSpec P = pangolin::ProjectionMatrixRDF_TopLeft(
        m_image_width, m_image_height, fx, fy, cx, cy, 0.001, 1000);

#endif
    cv::Mat im;
    int slam_status;
    int image_num = 0;
    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (*m_menu_stepbystep && !bStepByStep) {
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        } else if (!(*m_menu_stepbystep) && bStepByStep) {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if (*m_menu_step) {
            mpTracker->mbStep = true;
            *m_menu_step = false;
        }

        GetSLAMInfo(im, slam_status, image_num);

        if (m_switch_window_flag == 0) {
            m_d_cam_slam.show = true;
            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc, Ow, Twwp);
            if (mbStopTrack) {
                *m_menu_stepbystep = true;
                mbStopTrack = false;
            }

            Tools::DrawTxt("IMAGE: " + std::to_string(image_num), 220, 10);
            if (!(*m_menu_follow_camera)) {
                cv::Mat cam_pos;
                mpMapDrawer->GetCurrentCameraPos(cam_pos);
                m_trajectory.push_back(cam_pos);
            }

            if (*m_menu_follow_camera && bFollow) {
                if (bCameraView)
                    m_s_cam_slam.Follow(Twc);
                else
                    m_s_cam_slam.Follow(Ow);
            } else if (*m_menu_follow_camera && !bFollow) {
                if (bCameraView) {
                    m_s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                        1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1,
                        1000));
                    m_s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                        mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0,
                        -1.0, 0.0));
                    m_s_cam_slam.Follow(Twc);
                } else {
                    m_s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                        1024, 768, 3000, 3000, 512, 389, 0.1, 1000));
                    m_s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                        0, 0.01, 10, 0, 0, 0, 0.0, 0.0, 1.0));
                    m_s_cam_slam.Follow(Ow);
                }
                bFollow = true;
            } else if (!(*m_menu_follow_camera) && bFollow) {
                bFollow = false;
            }

            if (*m_menu_cam_view) {
                *m_menu_cam_view = false;
                bCameraView = true;
                m_s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 10000));
                m_s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0,
                    0.0));
                m_s_cam_slam.Follow(Twc);
            }

            if (*m_menu_top_view && mpMapDrawer->mpAtlas->isImuInitialized()) {
                *m_menu_top_view = false;
                bCameraView = false;
                m_s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, 3000, 3000, 512, 389, 0.1, 10000));
                m_s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
                m_s_cam_slam.Follow(Ow);
            }

            m_d_cam_slam.Activate(m_s_cam_slam);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            // pangolin::glDrawAxis(0.6f);
            // glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
            // pangolin::glDraw_z0(0.5f, 100);

            // mpMapDrawer->DrawCurrentCamera(Twc);
            if (*m_menu_show_keyframes || *m_menu_show_graph ||
                *m_menu_show_inertial_graph)
                mpMapDrawer->DrawKeyFrames(
                    *m_menu_show_keyframes, *m_menu_show_graph,
                    *m_menu_show_inertial_graph);
            if (*m_menu_show_points)
                mpMapDrawer->DrawMapPoints();

            if (*m_menu_show_camera_trajectory) {
                mpMapDrawer->DrawCameraTrajectory(m_trajectory);
            }

            if (*m_menu_show_3DObject) {
                typedef std::shared_ptr<ObjRecognition::MapPoint> MPPtr;
                glColor3f(0.0f, 1.0f, 0.0f);
                glPointSize(4.0);

                Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
                T.rotate(m_Row.cast<float>());
                T.pretranslate(m_tow.cast<float>());

                glBegin(GL_POINTS);
                if (m_pointCloud_model) {
                    std::vector<MPPtr> &pointClouds =
                        m_pointCloud_model->GetPointClouds();
                    for (int i = 0; i < pointClouds.size(); i++) {
                        Eigen::Vector3f p =
                            pointClouds[i]->GetPose().cast<float>();
                        p = T.inverse() * p;
                        glVertex3f(p.x(), p.y(), p.z());
                    }
                }
                glEnd();
                if (*m_menu_show_matched_3DObject) {
                }
            }

            if (*m_menu_stop) {
                m_is_stop = true;
                break;
            }

            if (*m_menu_reset) {
                *m_menu_show_graph = true;
                *m_menu_show_inertial_graph = true;
                *m_menu_show_keyframes = true;
                *m_menu_show_points = true;
                bFollow = true;
                *m_menu_follow_camera = false;
                *m_menu_show_3DObject = true;
                *m_menu_show_matched_3DObject = true;
                *m_menu_show_camera_trajectory = true;
                // mpSystem->Reset();
                mpSystem->ResetActiveMap();
                *m_menu_reset = false;
                *m_menu_stop = false;
            }
        } else if (m_switch_window_flag == 1) {
#ifdef OBJECTRECOGNITION
            m_d_cam_objRecognition.show = true;
            glColor3f(1.0, 1.0, 1.0);
            cv::Mat Tcw;

            if (!im.empty()) {
                Tcw = m_cam_pos;
                cv::cvtColor(im, im, CV_GRAY2RGB);
                PrintSLAMStatusForViewer(slam_status, image_num, im);
                DrawImageTexture(m_image_texture, im);

                m_d_cam_objRecognition.Activate(m_s_cam_objRecognition);
                // draw boundingbox:
                ObjRecognition::ObjRecogResult result =
                    ObjRecognitionExd::ObjRecongManager::Instance()
                        .GetObjRecognitionResult();

                Eigen::Matrix<float, 3, 3> Rwo;
                Rwo.col(0) = Eigen::Vector3f::Map(&result.R_obj_buffer[0], 3);
                Rwo.col(1) = Eigen::Vector3f::Map(&result.R_obj_buffer[3], 3);
                Rwo.col(2) = Eigen::Vector3f::Map(&result.R_obj_buffer[6], 3);
                Eigen::Vector3f two =
                    Eigen::Vector3f::Map(&result.t_obj_buffer[0], 3);

                Eigen::Matrix4d Two = Eigen::Matrix4d::Identity();
                Two << Rwo(0, 0), Rwo(0, 1), Rwo(0, 2), two(0), Rwo(1, 0),
                    Rwo(1, 1), Rwo(1, 2), two(1), Rwo(2, 0), Rwo(2, 1),
                    Rwo(2, 2), two(2), 0, 0, 0, 1;

                cv::Mat Two_cv;
                cv::eigen2cv(Two, Two_cv);
                pangolin::OpenGlMatrix glTwo;
                Tools::ChangeCV44ToGLMatrixDouble(Two_cv, glTwo);
                std::vector<Eigen::Vector3d> boundingbox;
                for (size_t i = 0; i < 8; i++) {
                    boundingbox.emplace_back(Eigen::Vector3d(
                        result.bounding_box[i * 3],
                        result.bounding_box[i * 3 + 1],
                        result.bounding_box[i * 3 + 2]));
                }

                glClear(GL_DEPTH_BUFFER_BIT);
                // Load m_camera projection
                glMatrixMode(GL_PROJECTION);
                P.Load();
                glMatrixMode(GL_MODELVIEW);
                LoadCameraPose(Tcw);

                static bool flag = true;
                if (flag) {
                    flag = false;
                    auto trans = cal_trans(boundingbox);
                    textModel->set_trans(trans.first, trans.second);
                    //                    std::cout << "rot" << trans.first <<
                    //                    std::endl; std::cout << "offset" <<
                    //                    trans.second << std::endl; exit(0);
                }

                if (result.state_buffer[0] == 0) {
                    // tracking good
                    // draw under slam camera coords
                    glPushMatrix();
                    glTwo.Multiply();
                    DrawBoundingboxInImage(boundingbox);
                    //                    Draw3dText();
                    DrawPointCloudInImage(result.pointCloud_pos);
                    glPopMatrix();
                }
            }
#endif
        } else if (m_switch_window_flag == 2) {
            //            m_d_cam_detector.show = true;
            m_d_cam_detector.Activate(m_s_cam_detector);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            // pangolin::glDrawAxis(0.6f);
            // glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
            // pangolin::glDraw_z0(0.5f, 100);
            ShowConnectedKeyframes();
            ShowConnectedMapPoints();
        }

        mpFrameDrawer->DrawFrame(true);
        ObjRecognition::GlobalOcvViewer::Draw();
        pangolin::FinishFrame();

        if (Stop()) {
            while (isStopped()) {
                usleep(3000);
            }
        }

        if (CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::SetSLAMInfo(
    const cv::Mat &img, const int &slam_state, const int &image_num,
    const cv::Mat &camPos) {
    unique_lock<mutex> lock(m_pose_image_mutex);
    m_img_from_objRecognition = img.clone();
    m_slam_state_from_objRecognition = slam_state;
    m_img_num = image_num;
    m_cam_pos = camPos;
}

void Viewer::GetSLAMInfo(cv::Mat &img, int &state, int &image_num) {
    unique_lock<mutex> lock(m_pose_image_mutex);
    img = m_img_from_objRecognition.clone();
    state = m_slam_state_from_objRecognition;
    image_num = m_img_num;
}

void Viewer::SwitchWindow() {
    m_switch_window_flag = (m_switch_window_flag + 1) % 3;
    if (m_switch_window_flag == 0) {
        m_d_cam_objRecognition.show = false;
        m_d_cam_detector.show = false;
        m_d_cam_slam = pangolin::CreateDisplay()
                           .SetBounds(
                               0.0, 1.0, pangolin::Attach::Pix(200), 1.0,
                               -1024.0f / 768.0f)
                           .SetHandler(new pangolin::Handler3D(m_s_cam_slam));
        m_d_cam_slam.show = true;
    } else if (m_switch_window_flag == 1) {
        m_d_cam_slam.show = false;
        m_d_cam_detector.show = false;
        m_d_cam_objRecognition =
            pangolin::CreateDisplay()
                .SetBounds(
                    0, 1.0f, pangolin::Attach::Pix(200), 1.0f,
                    (float)m_image_width / m_image_height)
                .SetLock(pangolin::LockLeft, pangolin::LockTop);
        //.SetHandler(new pangolin::Handler3D(m_s_cam_objRecognition));
        m_d_cam_objRecognition.show = true;
    } else if (m_switch_window_flag == 2) {
        m_d_cam_slam.show = false;
        m_d_cam_objRecognition.show = false;
        m_d_cam_detector =
            pangolin::CreateDisplay()
                .SetBounds(
                    0, 1.0f, pangolin::Attach::Pix(200), 1.0f,
                    -1024.0f / 768.0f)
                //.SetLock(pangolin::LockLeft, pangolin::LockTop)
                .SetHandler(new pangolin::Handler3D(m_s_cam_detector));
        m_d_cam_detector.show = true;
    }
}

void Viewer::Run() {
    pangolin::CreateWindowAndBind(
        "ORB-SLAM3: Map Viewer", m_image_width + 200, m_image_height);
    // pangolin::CreateWindowAndBind("Viewer", w + 200, h);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    // Issue specific OpenGl we might need
    //    glEnable(GL_BLEND);
    //    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::function<void(void)> switch_win_callback =
        std::bind(&Viewer::SwitchWindow, this);
    pangolin::RegisterKeyPressCallback('s', switch_win_callback);

    DrawSLAMInit();
#ifdef OBJECTRECOGNITION
    DrawObjRecognitionInit();
#ifdef SUPERPOINT
#else
    DrawDetectorInit();
#endif
#endif

#ifdef OBJECT_BOX
    textModel = new Model("/home/zhangye/data1/objectRecognition/obj/box.obj");
#endif
#ifdef OBJECT_BAG
    textModel = new Model("/home/zhangye/data1/objectRecognition/obj/bag.obj");
#endif
#ifdef OBJECT_TOY
    textModel = new Model("/home/zhangye/data1/objectRecognition/obj/toy.obj");
#endif
    //    printf("OpenGL version supported by this platform (%s): \n",
    //    glGetString(GL_VERSION));

    Draw();
}

void Viewer::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop() {
    unique_lock<mutex> lock(mMutexStop);
    if (!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop() {
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if (mbFinishRequested)
        return false;
    else if (mbStopRequested) {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;
}

void Viewer::Release() {
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}
} // namespace ORB_SLAM3
