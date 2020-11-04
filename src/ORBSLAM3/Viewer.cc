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
#include "Visualizer/GlobalImageViewer.h"
#include "ORBSLAM3/Viewer.h"
#include "ORBSLAM3/ViewerCommon.h"
#include "include/Tools.h"
#include "mode.h"
namespace ORB_SLAM3 {

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
    switch_window_flag = false;
    image_width = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    image_height = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    imageTexture = pangolin::GlTexture();
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
    s_cam_objRecognition = pangolin::OpenGlRenderState();
    imageTexture = pangolin::GlTexture(
        image_width, image_height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
}

void Viewer::DrawSLAMInit() {
    pangolin::CreatePanel("menu").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(200));
    menuFollowCamera = std::make_unique<pangolin::Var<bool>>(
        "menu.Follow Camera", false, true);
    menuCamView =
        std::make_unique<pangolin::Var<bool>>("menu.Camera View", false, false);
    menuTopView =
        std::make_unique<pangolin::Var<bool>>("menu.Top View", false, false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    menuShowPoints =
        std::make_unique<pangolin::Var<bool>>("menu.Show Points", true, true);
    menuShowKeyFrames = std::make_unique<pangolin::Var<bool>>(
        "menu.Show KeyFrames", true, true);
    menuShowGraph =
        std::make_unique<pangolin::Var<bool>>("menu.Show Graph", false, true);
    menuShowCameraTrajectory = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Camera trajectory", true, true);
    menuShow3DObject =
        std::make_unique<pangolin::Var<bool>>("menu.Show 3DObject", true, true);
    menuShowMatched3DObject = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Matched 3DObject", true, true);
    menuShowInertialGraph = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Inertial Graph", true, true);
    menuReset =
        std::make_unique<pangolin::Var<bool>>("menu.Reset", false, false);
    menuStepByStep = std::make_unique<pangolin::Var<bool>>(
        "menu.Step By Step", false, true); // false, true
    menuStep = std::make_unique<pangolin::Var<bool>>("menu.Step", false, false);
    menuStop = std::make_unique<pangolin::Var<bool>>("menu.Stop", false, false);

    // Define Camera Render Object (for view / scene browsing)
    s_cam_slam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(
            1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(
            mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    d_cam_slam =
        pangolin::CreateDisplay()
            .SetBounds(
                0.0, 1.0, pangolin::Attach::Pix(200), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam_slam));
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

void Viewer::DrawMatchedMappoints() {
    std::vector<ObjRecognition::MapPointIndex> matchedMapPoint;
    std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds =
        m_pointCloud_model->GetPointClouds();
    std::vector<Eigen::Vector3f> matchedMapPointCoords;
    glColor3f(1.0f, 1.0f, 1.0f);
    glPointSize(9.0);
    glBegin(GL_POINTS);
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.rotate(m_Row.cast<float>());
    T.pretranslate(m_tow.cast<float>());
    ObjRecognition::GlobalPointCloudMatchViewer::GetMatchedMapPoint(
        matchedMapPoint);
    ObjRecognition::GlobalPointCloudMatchViewer::DrawMatchedMapPoint(
        pointClouds, T, matchedMapPoint, matchedMapPointCoords);
    for (int i = 0; i < matchedMapPointCoords.size(); i++) {
        glVertex3f(
            matchedMapPointCoords[i].x(), matchedMapPointCoords[i].y(),
            matchedMapPointCoords[i].z());
    }
    glEnd();
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
        *menuShowGraph = true;
    }

    if (mpTracker->m_objRecognition_mode_) {
        *menuShow3DObject = true;
        *menuShowCameraTrajectory = true;
        *menuShowMatched3DObject = true;
    }

#ifdef OBJECTRECOGNITION
    // for objrecognition
    float fx = ObjRecognition::CameraIntrinsic::GetInstance().FX();
    float fy = ObjRecognition::CameraIntrinsic::GetInstance().FY();
    float cx = ObjRecognition::CameraIntrinsic::GetInstance().CX();
    float cy = ObjRecognition::CameraIntrinsic::GetInstance().CY();

    pangolin::OpenGlMatrixSpec P = pangolin::ProjectionMatrixRDF_TopLeft(
        image_width, image_height, fx, fy, cx, cy, 0.001, 1000);

#endif
    cv::Mat im;
    int slam_status;
    int image_num = 0;
    while (true) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (*menuStepByStep && !bStepByStep) {
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        } else if (!(*menuStepByStep) && bStepByStep) {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if (*menuStep) {
            mpTracker->mbStep = true;
            *menuStep = false;
        }

        GetSLAMInfo(im, slam_status, image_num);

        if (!switch_window_flag) {
            d_cam_slam.show = true;
            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc, Ow, Twwp);
            if (mbStopTrack) {
                *menuStepByStep = true;
                mbStopTrack = false;
            }

            Tools::DrawTxt("IMAGE: " + std::to_string(image_num), 220, 10);
            if (!(*menuFollowCamera)) {
                cv::Mat cam_pos;
                mpMapDrawer->GetCurrentCameraPos(cam_pos);
                m_trajectory.push_back(cam_pos);
            }

            if (*menuFollowCamera && bFollow) {
                if (bCameraView)
                    s_cam_slam.Follow(Twc);
                else
                    s_cam_slam.Follow(Ow);
            } else if (*menuFollowCamera && !bFollow) {
                if (bCameraView) {
                    s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                        1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1,
                        1000));
                    s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                        mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0,
                        -1.0, 0.0));
                    s_cam_slam.Follow(Twc);
                } else {
                    s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                        1024, 768, 3000, 3000, 512, 389, 0.1, 1000));
                    s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                        0, 0.01, 10, 0, 0, 0, 0.0, 0.0, 1.0));
                    s_cam_slam.Follow(Ow);
                }
                bFollow = true;
            } else if (!(*menuFollowCamera) && bFollow) {
                bFollow = false;
            }

            if (*menuCamView) {
                *menuCamView = false;
                bCameraView = true;
                s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 10000));
                s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0,
                    0.0));
                s_cam_slam.Follow(Twc);
            }

            if (*menuTopView && mpMapDrawer->mpAtlas->isImuInitialized()) {
                *menuTopView = false;
                bCameraView = false;
                s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, 3000, 3000, 512, 389, 0.1, 10000));
                s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
                s_cam_slam.Follow(Ow);
            }

            d_cam_slam.Activate(s_cam_slam);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            pangolin::glDrawAxis(0.6f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
            pangolin::glDraw_z0(0.5f, 100);

            mpMapDrawer->DrawCurrentCamera(Twc);
            if (*menuShowKeyFrames || *menuShowGraph || *menuShowInertialGraph)
                mpMapDrawer->DrawKeyFrames(
                    *menuShowKeyFrames, *menuShowGraph, *menuShowInertialGraph);
            if (*menuShowPoints)
                mpMapDrawer->DrawMapPoints();

            if (*menuShowCameraTrajectory) {
                mpMapDrawer->DrawCameraTrajectory(m_trajectory);
            }

            if (*menuShow3DObject) {
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
                if (*menuShowMatched3DObject) {
                    DrawMatchedMappoints();
                }
            }

            if (*menuStop) {
                SetFinish();
                mpSystem->Shutdown();
            }

            if (*menuReset) {
                *menuShowGraph = true;
                *menuShowInertialGraph = true;
                *menuShowKeyFrames = true;
                *menuShowPoints = true;
                *menuLocalizationMode = false;
                bFollow = true;
                *menuFollowCamera = false;
                *menuShow3DObject = true;
                *menuShowMatched3DObject = true;
                *menuShowCameraTrajectory = true;
                // mpSystem->Reset();
                mpSystem->ResetActiveMap();
                *menuReset = false;
                *menuStop = false;
            }
        } else {

#ifdef OBJECTRECOGNITION

            glColor3f(1.0, 1.0, 1.0);
            cv::Mat Tcw;

            if (!im.empty()) {
                Tcw = Tcw_;
                cv::cvtColor(im, im, CV_GRAY2RGB);
                PrintSLAMStatusForViewer(slam_status, image_num, im);
                DrawImageTexture(imageTexture, im);

                d_cam_objRecognition.Activate(s_cam_objRecognition);
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
                eigen2cv(Two, Two_cv);
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

                if (result.state_buffer[0] == 0) {
                    // tracking good
                    // draw under slam camera coords
                    glPushMatrix();
                    glTwo.Multiply();
                    DrawBoundingboxInImage(boundingbox);
                    DrawPointCloudInImage(result.pointCloud_pos);
                    glPopMatrix();
                }
            }
#endif
        }

        mpFrameDrawer->DrawFrame(true);
        ObjRecognition::GlobalOcvViewer::DrawAllView();
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
    unique_lock<mutex> lock(mMutexPoseImage);
    img_from_objRecognition = img.clone();
    slam_state_from_objRecognition = slam_state;
    img_num = image_num;
    Tcw_ = camPos;
}

void Viewer::GetSLAMInfo(cv::Mat &img, int &state, int &image_num) {
    unique_lock<mutex> lock(mMutexPoseImage);
    img = img_from_objRecognition.clone();
    state = slam_state_from_objRecognition;
    image_num = img_num;
}

void Viewer::SwitchWindow() {
    if (switch_window_flag) {
        d_cam_objRecognition.show = false;
        d_cam_slam = pangolin::CreateDisplay()
                         .SetBounds(
                             0.0, 1.0, pangolin::Attach::Pix(200), 1.0,
                             -1024.0f / 768.0f)
                         .SetHandler(new pangolin::Handler3D(s_cam_slam));
        d_cam_slam.show = true;
    } else {
        d_cam_slam.show = false;
        d_cam_objRecognition =
            pangolin::CreateDisplay()
                .SetBounds(
                    0, 1.0f, pangolin::Attach::Pix(200), 1.0f,
                    (float)image_width / image_height)
                .SetLock(pangolin::LockLeft, pangolin::LockTop)
                .SetHandler(new pangolin::Handler3D(s_cam_objRecognition));
        d_cam_objRecognition.show = true;
    }
    switch_window_flag = !switch_window_flag;
}

void Viewer::Run() {
    pangolin::CreateWindowAndBind(
        "ORB-SLAM3: Map Viewer", image_width + 200, image_height);
    // pangolin::CreateWindowAndBind("Viewer", w + 200, h);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::function<void(void)> switch_win_callback =
        std::bind(&Viewer::SwitchWindow, this);
    pangolin::RegisterKeyPressCallback('s', switch_win_callback);

    DrawSLAMInit();
#ifdef OBJECTRECOGNITION
    DrawObjRecognitionInit();
#endif
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

void Viewer::SetTrackingPause() {
    mbStopTrack = true;
}

} // namespace ORB_SLAM3
