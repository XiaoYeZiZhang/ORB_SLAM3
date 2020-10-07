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
#include "include/ORBSLAM3/Viewer.h"

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

void Viewer::Run() {
    mbFinished = false;
    mbStopped = false;

    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer", 1024, 768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false, true);
    pangolin::Var<bool> menuCamView("menu.Camera View", false, false);
    pangolin::Var<bool> menuTopView("menu.Top View", false, false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", false, true);
    pangolin::Var<bool> menuShowCameraTrajectory(
        "menu.Show Camera trajectory", true, true);
    pangolin::Var<bool> menuShow3DObject("menu.Show 3DObject", true, true);
    pangolin::Var<bool> menuShowInertialGraph(
        "menu.Show Inertial Graph", true, true);
    pangolin::Var<bool> menuLocalizationMode(
        "menu.Localization Mode", false, true);
    pangolin::Var<bool> menuReset("menu.Reset", false, false);
    pangolin::Var<bool> menuStepByStep(
        "menu.Step By Step", false, true); // false, true
    pangolin::Var<bool> menuStep("menu.Step", false, false);
    pangolin::Var<bool> menuStop("menu.Stop", false, false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(
            1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(
            mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam =
        pangolin::CreateDisplay()
            .SetBounds(
                0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

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

    if (mpTracker->mSensor == mpSystem->MONOCULAR) {
        menuShowGraph = true;
    }

    if (mpTracker->m_objRecognition_mode_) {
        menuShow3DObject = true;
        menuShowCameraTrajectory = true;
    }

    while (1) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc, Ow, Twwp);

        if (mbStopTrack) {
            menuStepByStep = true;
            mbStopTrack = false;
        }

        if (!menuFollowCamera) {
            cv::Mat cam_pos;
            // TODO(zhangye): check the cam pos???
            mpMapDrawer->GetCurrentCameraPos(cam_pos);
            m_trajectory.push_back(cam_pos);
        }

        if (menuFollowCamera && bFollow) {
            if (bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        } else if (menuFollowCamera && !bFollow) {
            if (bCameraView) {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0,
                    0.0));
                s_cam.Follow(Twc);
            } else {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                    1024, 768, 3000, 3000, 512, 389, 0.1, 1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                    0, 0.01, 10, 0, 0, 0, 0.0, 0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        } else if (!menuFollowCamera && bFollow) {
            bFollow = false;
        }

        if (menuCamView) {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0,
                0.0));
            s_cam.Follow(Twc);
        }

        if (menuTopView && mpMapDrawer->mpAtlas->isImuInitialized()) {
            menuTopView = false;
            bCameraView = false;
            /*s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10,
            0,0,0,0.0,0.0, 1.0));*/
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                1024, 768, 3000, 3000, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(0, 0.01, 50, 0, 0, 0, 0.0, 0.0, 1.0));
            s_cam.Follow(Ow);
        }

        /*if(menuSideView && mpMapDrawer->mpAtlas->isImuInitialized())
        {
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0.0,0.1,30.0,0,0,0,0.0,0.0,1.0));
            s_cam.Follow(Twwp);
        }*/

        if (menuLocalizationMode && !bLocalizationMode) {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        } else if (!menuLocalizationMode && bLocalizationMode) {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        if (menuStepByStep && !bStepByStep) {
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        } else if (!menuStepByStep && bStepByStep) {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if (menuStep) {
            mpTracker->mbStep = true;
            menuStep = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        pangolin::glDrawAxis(0.6f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
        pangolin::glDraw_z0(0.5f, 100);

        mpMapDrawer->DrawCurrentCamera(Twc);
        if (menuShowKeyFrames || menuShowGraph || menuShowInertialGraph)
            mpMapDrawer->DrawKeyFrames(
                menuShowKeyFrames, menuShowGraph, menuShowInertialGraph);
        if (menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        if (menuShowCameraTrajectory) {
            mpMapDrawer->DrawCameraTrajectory(m_trajectory);
        }

        if (menuShow3DObject) {
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
                    Eigen::Vector3f p = pointClouds[i]->GetPose().cast<float>();
                    p = T.inverse() * p;
                    glVertex3f(p.x(), p.y(), p.z());
                }
            }
            glEnd();
        }

        if (menuStop) {
            SetFinish();
            mpSystem->Shutdown();
        }

        pangolin::FinishFrame();

        mpFrameDrawer->DrawFrame(true);
        ObjRecognition::GlobalOcvViewer::DrawAllView();

        if (menuReset) {
            menuShowGraph = true;
            menuShowInertialGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if (bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = false;
            menuShow3DObject = true;
            menuShowCameraTrajectory = true;
            // mpSystem->Reset();
            mpSystem->ResetActiveMap();
            menuReset = false;
            menuStop = false;
        }

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
