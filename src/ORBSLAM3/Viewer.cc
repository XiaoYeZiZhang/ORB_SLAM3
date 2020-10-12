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
    Eigen::Matrix3d Row, Eigen::Vector3d tow) {
    m_Row = Row;
    m_tow = tow;
}

void Viewer::DrawObjRecognitionInit() {
    int w = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    int h = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    // define projection and initial movelview matrix: default
    s_cam_objRecognition = pangolin::OpenGlRenderState();
    d_cam_objRecognition =
        pangolin::Display("image")
            .SetBounds(0, 1.0f, pangolin::Attach::Pix(200), 1.0f, (float)w / h)
            .SetLock(pangolin::LockLeft, pangolin::LockTop)
            .SetHandler(new pangolin::Handler3D(s_cam_objRecognition));
    d_cam_objRecognition.show = false;
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
    menuShowInertialGraph = std::make_unique<pangolin::Var<bool>>(
        "menu.Show Inertial Graph", true, true);
    menuLocalizationMode = std::make_unique<pangolin::Var<bool>>(
        "menu.Localization Mode", false, true);
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
    d_cam_slam.show = false;
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
    bool bLocalizationMode = false;
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
    }

#ifdef OBJECTRECOGNITION
    // for objrecognition
    int w = ObjRecognition::CameraIntrinsic::GetInstance().Width();
    int h = ObjRecognition::CameraIntrinsic::GetInstance().Height();
    float fx = ObjRecognition::CameraIntrinsic::GetInstance().FX();
    float fy = ObjRecognition::CameraIntrinsic::GetInstance().FY();
    float cx = ObjRecognition::CameraIntrinsic::GetInstance().CX();
    float cy = ObjRecognition::CameraIntrinsic::GetInstance().CY();
    pangolin::GlTexture imageTexture(
        w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::OpenGlMatrixSpec P = pangolin::ProjectionMatrixRDF_TopLeft(
        w, h, fx, fy, cx, cy, 0.001, 1000);
    cv::Mat im;
    int status;
#endif

    while (1) {
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

        if (!switch_window_flag) {
            d_cam_slam.show = true;
            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc, Ow, Twwp);
            if (mbStopTrack) {
                *menuStepByStep = true;
                mbStopTrack = false;
            }

            if (!(*menuFollowCamera)) {
                cv::Mat cam_pos;
                // TODO(zhangye): check the cam pos???
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
                /*s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10,
                0,0,0,0.0,0.0, 1.0));*/
                s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, 3000, 3000, 512, 389, 0.1, 10000));
                s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
                s_cam_slam.Follow(Ow);
            }

            /*if(menuSideView && mpMapDrawer->mpAtlas->isImuInitialized())
            {
                s_cam_slam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
                s_cam_slam.SetModelViewMatrix(pangolin::ModelViewLookAt(0.0,0.1,30.0,0,0,0,0.0,0.0,1.0));
                s_cam_slam.Follow(Twwp);
            }*/

            if (*menuLocalizationMode && !bLocalizationMode) {
                mpSystem->ActivateLocalizationMode();
                bLocalizationMode = true;
            } else if (!(*menuLocalizationMode) && bLocalizationMode) {
                mpSystem->DeactivateLocalizationMode();
                bLocalizationMode = false;
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
            }

            if (*menuStop) {
                SetFinish();
                mpSystem->Shutdown();
            }

            mpFrameDrawer->DrawFrame(true);
            ObjRecognition::GlobalOcvViewer::DrawAllView();

            if (*menuReset) {
                *menuShowGraph = true;
                *menuShowInertialGraph = true;
                *menuShowKeyFrames = true;
                *menuShowPoints = true;
                *menuLocalizationMode = false;
                if (bLocalizationMode)
                    mpSystem->DeactivateLocalizationMode();
                bLocalizationMode = false;
                bFollow = true;
                *menuFollowCamera = false;
                *menuShow3DObject = true;
                *menuShowCameraTrajectory = true;
                // mpSystem->Reset();
                mpSystem->ResetActiveMap();
                *menuReset = false;
                *menuStop = false;
            }
        } else {
#ifdef OBJECTRECOGNITION
            // draw for objectRecognition:
            d_cam_objRecognition.show = true;
            // Activate m_camera view
            d_cam_objRecognition.Activate(s_cam_objRecognition);
            glColor3f(1.0, 1.0, 1.0);
            cv::Mat Rwc, twc;
            cv::Mat Tcw;
            GetFrameAndState(im, status);
            if (!im.empty()) {
                mpMapDrawer->GetCurrentCameraPose(Rwc, twc, Tcw);
                if (!Rwc.empty() && !twc.empty()) {
                    m_camera.SetCamPos(
                        twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));
                }

                PrintStatusForViewer(status, im);
                cv::cvtColor(im, im, CV_GRAY2RGB);
                DrawImageTexture(imageTexture, im);

                // draw boundingbox:
                ObjRecognition::ObjRecogResult result =
                    ObjRecognitionExd::ObjRecongManager::Instance()
                        .GetObjRecognitionResult();

                Eigen::Matrix4d Two = Eigen::Matrix4d::Identity();
                Two << result.R_obj_buffer[0], result.R_obj_buffer[1],
                    result.R_obj_buffer[2], result.t_obj_buffer[0],
                    result.R_obj_buffer[3], result.R_obj_buffer[4],
                    result.R_obj_buffer[5], result.t_obj_buffer[1],
                    result.R_obj_buffer[6], result.R_obj_buffer[7],
                    result.R_obj_buffer[8], result.t_obj_buffer[2], 0, 0, 0, 1;

                // TODO(zhangye):  check the column priority??
                cv::Mat Two_cv;
                eigen2cv(Two, Two_cv);
                pangolin::OpenGlMatrix glTwo;
                glTwo.m[0] = Two_cv.at<float>(0, 0);
                glTwo.m[1] = Two_cv.at<float>(1, 0);
                glTwo.m[2] = Two_cv.at<float>(2, 0);
                glTwo.m[3] = 0.0;

                glTwo.m[4] = Two_cv.at<float>(0, 1);
                glTwo.m[5] = Two_cv.at<float>(1, 1);
                glTwo.m[6] = Two_cv.at<float>(2, 1);
                glTwo.m[7] = 0.0;

                glTwo.m[8] = Two_cv.at<float>(0, 2);
                glTwo.m[9] = Two_cv.at<float>(1, 2);
                glTwo.m[10] = Two_cv.at<float>(2, 2);
                glTwo.m[11] = 0.0;

                glTwo.m[12] = Two_cv.at<float>(0, 3);
                glTwo.m[13] = Two_cv.at<float>(1, 3);
                glTwo.m[14] = Two_cv.at<float>(2, 3);
                glTwo.m[15] = 1.0;

                std::vector<Eigen::Vector3d> boundingbox;
                for (size_t i = 0; i < 8; i++) {
                    boundingbox.emplace_back(result.bounding_box[i * 3]);
                    boundingbox.emplace_back(result.bounding_box[i * 3 + 1]);
                    boundingbox.emplace_back(result.bounding_box[i * 3 + 2]);
                }

                glClear(GL_DEPTH_BUFFER_BIT);
                // Load m_camera projection
                glMatrixMode(GL_PROJECTION);
                P.Load();
                // load model view matrix
                glMatrixMode(GL_MODELVIEW);
                // Load m_camera pose  set opengl coords, same as slam coords
                // view matrix Tcw
                LoadCameraPose(Tcw);
                // draw under slam camera coords

                // draw boundingbox
                glPushMatrix();
                // Two
                glTwo.Multiply();
                // TODO(zhangye): check the coords, now the result is wrong
                DrawBoundingboxInImage(boundingbox);
                glPopMatrix();
                DrawPointCloudInImage(result.pointCloud_pos);
            }
#endif
        }

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

void Viewer::SetFrameAndState(const cv::Mat &img, const int &state) {
    unique_lock<mutex> lock(mMutexPoseImage);
    img_from_objRecognition = img.clone();
    state_from_objRecognition = state;
}

void Viewer::GetFrameAndState(cv::Mat &img, int &state) {
    unique_lock<mutex> lock(mMutexPoseImage);
    img = img_from_objRecognition.clone();
    state = state_from_objRecognition;
}

void Viewer::SwitchWindow() {
    switch_window_flag = !switch_window_flag;
    if (!switch_window_flag) {
        d_cam_objRecognition.show = false;
    } else {
        d_cam_slam.show = false;
    }
}

void Viewer::Run() {
    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer", 1024, 768);
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
    // DrawObjRecognition();
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
