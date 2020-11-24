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

#include "include/ORBSLAM3/MapDrawer.h"
#include "include/ORBSLAM3/MapPoint.h"
#include "include/ORBSLAM3/KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <cxeigen.hpp>
#include "include/Tools.h"
#include "mode.h"

namespace ORB_SLAM3 {

MapDrawer::MapDrawer(Atlas *pAtlas, const string &strSettingPath)
    : mpAtlas(pAtlas) {
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
}

MapDrawer::MapDrawer(
    Atlas *pAtlas, Atlas *pAtlas_superpoint, const string &strSettingPath)
    : mpAtlas(pAtlas), mpAtlas_superpoint(pAtlas_superpoint) {
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
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings) {
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if (!node.empty()) {
        mKeyFrameSize = node.real();
    } else {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if (!node.empty()) {
        mKeyFrameLineWidth = node.real();
    } else {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is "
                     "not a real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if (!node.empty()) {
        mGraphLineWidth = node.real();
    } else {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not "
                     "a real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if (!node.empty()) {
        mPointSize = node.real();
    } else {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if (!node.empty()) {
        mCameraSize = node.real();
    } else {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a "
                     "real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if (!node.empty()) {
        mCameraLineWidth = node.real();
    } else {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is "
                     "not a real number*"
                  << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints_SuperPoint(
    const std::vector<double> &boundingbox_p_corner,
    const std::set<MapPoint *> &mappoint_picked, const Eigen::Matrix4d &Twp) {
#ifdef SUPERPOINT
    int covisualize_keyframe_num = 3;
#else
    int covisualize_keyframe_num = 0;
#endif
    const vector<MapPoint *> &vpMPs =
        mpAtlas_superpoint->GetAllMapPoints(covisualize_keyframe_num);
    const vector<MapPoint *> &vpRefMPs =
        mpAtlas_superpoint->GetReferenceMapPoints();

    set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty())
        return;

    // mappoints:
    glPointSize(mPointSize);
    glBegin(GL_POINTS);

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
        if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        if (mappoint_picked.count(vpMPs[i])) {
            glColor3f(0.0, 0.0, 1.0);
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        } else {
            Eigen::Vector3d mappoint_pos_w;
            cv::cv2eigen(pos, mappoint_pos_w);
            Eigen::Vector4d mappoint_pos_w_4_4 = Eigen::Vector4d(
                mappoint_pos_w(0), mappoint_pos_w(1), mappoint_pos_w(2), 1.0);
            Eigen::Vector4d mappoint_pos_p_4_4 =
                Twp.inverse() * mappoint_pos_w_4_4;
            Eigen::Vector3d mappoint_p = Eigen::Vector3d(
                mappoint_pos_p_4_4(0) / mappoint_pos_p_4_4(3),
                mappoint_pos_p_4_4(1) / mappoint_pos_p_4_4(3),
                mappoint_pos_p_4_4(2) / mappoint_pos_p_4_4(3));

            if (mappoint_p(0) > boundingbox_p_corner[0] &&
                mappoint_p(1) > boundingbox_p_corner[1] &&
                mappoint_p(2) > boundingbox_p_corner[2] &&
                mappoint_p(0) < boundingbox_p_corner[3] &&
                mappoint_p(1) < boundingbox_p_corner[4] &&
                mappoint_p(2) < boundingbox_p_corner[5]) {
                glColor3f(1.0, 0.0, 0.0);
                glVertex3f(
                    pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
            } else {
                glColor3f(0.0, 1.0, 0.0);
                glVertex3f(
                    pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
            }
        }
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 1.0);

    for (set<MapPoint *>::iterator sit = spRefMPs.begin(),
                                   send = spRefMPs.end();
         sit != send; sit++) {
        if ((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }

    glEnd();
}

void MapDrawer::DrawMapPoints() {
    const vector<MapPoint *> &vpMPs = mpAtlas->GetAllMapPoints();
    const vector<MapPoint *> &vpRefMPs = mpAtlas->GetReferenceMapPoints();

    set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty())
        return;

    // mappoints:
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
        if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 1.0);

    for (set<MapPoint *>::iterator sit = spRefMPs.begin(),
                                   send = spRefMPs.end();
         sit != send; sit++) {
        if ((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }

    glEnd();
}

void MapDrawer::DrawKeyFrames(
    const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph) {
    const float &w = mKeyFrameSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    const vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();

    if (bDrawKF) {
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            if (!pKF->GetParent()) // It is the first KF in the map
            {
                // first kf: red
                glLineWidth(mKeyFrameLineWidth * 5);
                glColor3f(1.0f, 0.0f, 0.0f);
                glBegin(GL_LINES);

                // cout << "Initial KF: " <<
                // mpAtlas->GetCurrentMap()->GetOriginKF()->mnId << endl; cout
                // << "Parent KF: " << vpKFs[i]->mnId << endl;
            } else {
                glLineWidth(mKeyFrameLineWidth);
                // glColor3f(0.0f,0.0f,1.0f);
                glColor3f(
                    mfFrameColors[index_color][0],
                    mfFrameColors[index_color][1],
                    mfFrameColors[index_color][2]);
                glBegin(GL_LINES);
            }

            glVertex3f(0, 0, 0);
            glVertex3f(w, h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, h, z);

            glVertex3f(w, h, z);
            glVertex3f(w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(-w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(w, h, z);

            glVertex3f(-w, -h, z);
            glVertex3f(w, -h, z);
            glEnd();

            glPopMatrix();

            // Draw lines with Loop and Merge candidates
            /*glLineWidth(mGraphLineWidth);
            glColor4f(1.0f,0.6f,0.0f,1.0f);
            glBegin(GL_LINES);
            cv::Mat Ow = pKF->GetCameraCenter();
            const vector<KeyFrame*> vpLoopCandKFs = pKF->mvpLoopCandKFs;
            if(!vpLoopCandKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vpLoopCandKFs.begin(),
            vend=vpLoopCandKFs.end(); vit!=vend; vit++)
                {
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }
            const vector<KeyFrame*> vpMergeCandKFs = pKF->mvpMergeCandKFs;
            if(!vpMergeCandKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator
            vit=vpMergeCandKFs.begin(), vend=vpMergeCandKFs.end(); vit!=vend;
            vit++)
                {
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }*/

            glEnd();
        }
    }

    if (bDrawGraph) {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for (size_t i = 0; i < vpKFs.size(); i++) {
            // Covisibility Graph
            const vector<KeyFrame *> vCovKFs =
                vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if (!vCovKFs.empty()) {
                for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(),
                                                        vend = vCovKFs.end();
                     vit != vend; vit++) {
                    if ((*vit)->mnId < vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(
                        Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                    glVertex3f(
                        Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame *pParent = vpKFs[i]->GetParent();
            if (pParent) {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                glVertex3f(
                    Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(),
                                           send = sLoopKFs.end();
                 sit != send; sit++) {
                if ((*sit)->mnId < vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                glVertex3f(
                    Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
            }
        }

        glEnd();
    }

    if (bDrawInertialGraph && mpAtlas->isImuInitialized()) {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f, 0.0f, 0.0f, 0.6f);
        glBegin(GL_LINES);

        // Draw inertial links
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKFi = vpKFs[i];
            cv::Mat Ow = pKFi->GetCameraCenter();
            KeyFrame *pNext = pKFi->mNextKF;
            if (pNext) {
                cv::Mat Owp = pNext->GetCameraCenter();
                glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
                glVertex3f(
                    Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
            }
        }

        glEnd();
    }

    vector<Map *> vpMaps = mpAtlas->GetAllMaps();

    if (bDrawKF) {
        for (Map *pMap : vpMaps) {
            if (pMap == mpAtlas->GetCurrentMap())
                continue;

            vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();

            for (size_t i = 0; i < vpKFs.size(); i++) {
                KeyFrame *pKF = vpKFs[i];
                cv::Mat Twc = pKF->GetPoseInverse().t();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf(Twc.ptr<GLfloat>(0));

                if (!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth * 5);
                    glColor3f(1.0f, 0.0f, 0.0f);
                    glBegin(GL_LINES);
                } else {
                    glLineWidth(mKeyFrameLineWidth);
                    // glColor3f(0.0f,0.0f,1.0f);
                    glColor3f(
                        mfFrameColors[index_color][0],
                        mfFrameColors[index_color][1],
                        mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0, 0, 0);
                glVertex3f(w, h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, -h, z);
                glVertex3f(0, 0, 0);
                glVertex3f(-w, h, z);

                glVertex3f(w, h, z);
                glVertex3f(w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(-w, -h, z);

                glVertex3f(-w, h, z);
                glVertex3f(w, h, z);

                glVertex3f(-w, -h, z);
                glVertex3f(w, -h, z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) {
    const float &w = mCameraSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw) {
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::DrawCameraTrajectory(const std::vector<cv::Mat> &trajectory) {
    glPointSize(mPointSize);
    glColor3f(1.0, 1.0, 0.0);
    glBegin(GL_LINE_STRIP);
    for (const auto &p : trajectory) {
        if (p.cols > 0 && p.rows > 0) {
            glVertex3f(p.at<float>(0), p.at<float>(1), p.at<float>(2));
        }
    }
    glEnd();
}

void MapDrawer::GetCurrentCameraPos(cv::Mat &cam_pos) {
    if (!mCameraPose.empty()) {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
        }
        cam_pos = twc;
    } else {
        cam_pos = cv::Mat();
    }
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(
    pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw) {
    if (!mCameraPose.empty()) {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
        }

        Tools::ChangeCV44ToGLMatrixFloat(Rwc, M);
        MOw.SetIdentity();
        MOw.m[12] = twc.at<float>(0);
        MOw.m[13] = twc.at<float>(1);
        MOw.m[14] = twc.at<float>(2);
    } else {
        M.SetIdentity();
        MOw.SetIdentity();
    }
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(
    pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw,
    pangolin::OpenGlMatrix &MTwwp) {
    if (!mCameraPose.empty()) {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        cv::Mat Rwwp(3, 3, CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
        }

        Tools::ChangeCV44ToGLMatrixFloat(Rwc, M);
        MOw.SetIdentity();
        MOw.m[12] = twc.at<float>(0);
        MOw.m[13] = twc.at<float>(1);
        MOw.m[14] = twc.at<float>(2);

        MTwwp.SetIdentity();
        Tools::ChangeCV44ToGLMatrixFloat(Rwwp, MTwwp);
    } else {
        M.SetIdentity();
        MOw.SetIdentity();
        MTwwp.SetIdentity();
    }
}

} // namespace ORB_SLAM3
