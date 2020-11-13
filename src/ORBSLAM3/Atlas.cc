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

#include "include/ORBSLAM3/Atlas.h"
#include "include/ORBSLAM3/Viewer.h"

#include "GeometricCamera.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "ObjectRecognition/Utility/Camera.h"
#include "include/Tools.h"
#include "mode.h"

namespace ORB_SLAM3 {

Atlas::Atlas() {
    mpCurrentMap = static_cast<Map *>(NULL);
}

Atlas::Atlas(int initKFid) : mnLastInitKFidMap(initKFid), mHasViewer(false) {
    mpCurrentMap = static_cast<Map *>(NULL);
    CreateNewMap();
}

Atlas::~Atlas() {
    for (std::set<Map *>::iterator it = mspMaps.begin(), end = mspMaps.end();
         it != end;) {
        Map *pMi = *it;

        if (pMi) {
            delete pMi;
            pMi = static_cast<Map *>(NULL);

            it = mspMaps.erase(it);
        } else
            ++it;
    }
}

void Atlas::CreateNewMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    VLOG(5) << "ORBSLAM3: Creation of new map with id: " << Map::nNextId;
    if (mpCurrentMap) {
        VLOG(5) << "ORBSLAM3: Exits current map " << endl;
        if (!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() +
                                1; // The init KF is the next of current maximum

        mpCurrentMap->SetStoredMap();
        VLOG(5) << "ORBSLAM3: Saved map with ID: " << mpCurrentMap->GetId();

        // if(mHasViewer)
        //    mpViewer->AddMapToCreateThumbnail(mpCurrentMap);
    }
    VLOG(5) << "ORBSLAM3: Creation of new map with last KF id: "
            << mnLastInitKFidMap;

    mpCurrentMap = new Map(mnLastInitKFidMap);
    mpCurrentMap->SetCurrentMap();
    mspMaps.insert(mpCurrentMap);
}

void Atlas::ChangeMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexAtlas);
    VLOG(5) << "ORBSLAM3: Chage to map with id: " << pMap->GetId();
    if (mpCurrentMap) {
        mpCurrentMap->SetStoredMap();
    }

    mpCurrentMap = pMap;
    mpCurrentMap->SetCurrentMap();
}

unsigned long int Atlas::GetLastInitKFid() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mnLastInitKFidMap;
}

void Atlas::SetViewer(Viewer *pViewer) {
    mpViewer = pViewer;
    mHasViewer = true;
}

void Atlas::AddKeyFrame_superpoint(KeyFrame *pKF) {
    Map *pMapKF = GetCurrentMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddKeyFrame(KeyFrame *pKF) {
    Map *pMapKF = pKF->GetMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddMapPoint(MapPoint *pMP) {
    Map *pMapMP = pMP->GetMap();
    pMapMP->AddMapPoint(pMP);
}

void Atlas::AddCamera(GeometricCamera *pCam) {
    mvpCameras.push_back(pCam);
}

void Atlas::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs) {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetReferenceMapPoints(vpMPs);
}

void Atlas::InformNewBigChange() {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->InformNewBigChange();
}

int Atlas::GetLastBigChangeIdx() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetLastBigChangeIdx();
}

long unsigned int Atlas::MapPointsInMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->MapPointsInMap();
}

long unsigned Atlas::KeyFramesInMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->KeyFramesInMap();
}

std::vector<KeyFrame *> Atlas::GetAllKeyFrames() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllKeyFrames();
}

std::vector<MapPoint *> Atlas::GetAllMapPoints(const int covis_keyframe_num) {
    unique_lock<mutex> lock(mMutexAtlas);
    if (covis_keyframe_num) {
        std::vector<MapPoint *> good_mappoints =
            mpCurrentMap->GetAllMapPoints(covis_keyframe_num);
        return good_mappoints;
    } else {
        return mpCurrentMap->GetAllMapPoints();
    }
}

std::vector<MapPoint *> Atlas::GetReferenceMapPoints() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetReferenceMapPoints();
}

vector<Map *> Atlas::GetAllMaps() {
    unique_lock<mutex> lock(mMutexAtlas);
    struct compFunctor {
        inline bool operator()(Map *elem1, Map *elem2) {
            return elem1->GetId() < elem2->GetId();
        }
    };
    vector<Map *> vMaps(mspMaps.begin(), mspMaps.end());
    sort(vMaps.begin(), vMaps.end(), compFunctor());
    return vMaps;
}

int Atlas::CountMaps() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mspMaps.size();
}

void Atlas::clearMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->clear();
}

void Atlas::clearAtlas() {
    unique_lock<mutex> lock(mMutexAtlas);
    /*for(std::set<Map*>::iterator it=mspMaps.begin(), send=mspMaps.end();
    it!=send; it++)
    {
        (*it)->clear();
        delete *it;
    }*/
    mspMaps.clear();
    mpCurrentMap = static_cast<Map *>(NULL);
    mnLastInitKFidMap = 0;
}

Map *Atlas::GetCurrentMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    if (!mpCurrentMap)
        CreateNewMap();
    while (mpCurrentMap->IsBad())
        usleep(3000);

    return mpCurrentMap;
}

void Atlas::SetMapBad(Map *pMap) {
    mspMaps.erase(pMap);
    pMap->SetBad();

    mspBadMaps.insert(pMap);
}

void Atlas::RemoveBadMaps() {
    /*for(Map* pMap : mspBadMaps)
    {
        delete pMap;
        pMap = static_cast<Map*>(NULL);
    }*/
    mspBadMaps.clear();
}

bool Atlas::isInertial() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->IsInertial();
}

void Atlas::SetInertialSensor() {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetInertialSensor();
}

void Atlas::SetImuInitialized() {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetImuInitialized();
}

bool Atlas::isImuInitialized() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->isImuInitialized();
}

void Atlas::GetBoundingBoxCoordsRange() {
    m_bbx_xmin = 10000;
    m_bbx_ymin = 10000;
    m_bbx_zmin = 10000;
    m_bbx_xmax = -10000;
    m_bbx_ymax = -10000;
    m_bbx_zmax = -10000;

    for (int i = 0; i < m_boundingbox_w_.size(); i++) {
        if (m_boundingbox_w_[i](0) < m_bbx_xmin) {
            m_bbx_xmin = m_boundingbox_w_[i](0);
        }
        if (m_boundingbox_w_[i](0) > m_bbx_xmax) {
            m_bbx_xmax = m_boundingbox_w_[i](0);
        }

        if (m_boundingbox_w_[i](1) < m_bbx_ymin) {
            m_bbx_ymin = m_boundingbox_w_[i](1);
        }
        if (m_boundingbox_w_[i](1) > m_bbx_ymax) {
            m_bbx_ymax = m_boundingbox_w_[i](1);
        }

        if (m_boundingbox_w_[i](2) < m_bbx_zmin) {
            m_bbx_zmin = m_boundingbox_w_[i](2);
        }
        if (m_boundingbox_w_[i](2) > m_bbx_zmax) {
            m_bbx_zmax = m_boundingbox_w_[i](2);
        }
    }
}

bool Atlas::MappointInBoundingbox(const cv::Mat &mappoint_pos) {
    if (mappoint_pos.at<float>(0) >= m_bbx_xmin &&
        mappoint_pos.at<float>(0) <= m_bbx_xmax &&
        mappoint_pos.at<float>(1) >= m_bbx_ymin &&
        mappoint_pos.at<float>(1) <= m_bbx_ymax &&
        mappoint_pos.at<float>(2) >= m_bbx_zmin &&
        mappoint_pos.at<float>(2) <= m_bbx_zmax) {
        return true;
    } else {
        return false;
    }
}

long long Atlas::GetMemSizeFor3DObject(
    const std::string &version, const bool is_superpoint) {
    // already set boundingbox
    GetBoundingBoxCoordsRange();
    m_3dobject_version_ = version;

    long long nTotalSize = 0;
    if (!mspMaps.empty()) {
        // version
        nTotalSize += m_3dobject_version_.size();
        // timestamp
        nTotalSize += sizeof(double);
        // width + height
        nTotalSize += sizeof(int) * 2;
        // fx, fy, cx, cy
        nTotalSize += sizeof(double) * 4;

        // bounding box + scale
        nTotalSize += 27 * sizeof(double);

        VLOG(0) << "getmemsize1" << nTotalSize;
        nTotalSize += sizeof(unsigned int);
        std::vector<Map *> saved_map;
        struct compFunctor {
            inline bool operator()(Map *elem1, Map *elem2) {
                return elem1->GetId() < elem2->GetId();
            }
        };
        std::copy(
            mspMaps.begin(), mspMaps.end(), std::back_inserter(saved_map));
        sort(saved_map.begin(), saved_map.end(), compFunctor());

#ifdef SUPERPOINT
        int covisualize_keyframe_num = 8;
#else
        int covisualize_keyframe_num = 5;
#endif
        for (Map *pMi : saved_map) {
            for (MapPoint *pMPi :
                 pMi->GetAllMapPoints(covisualize_keyframe_num)) {
                cv::Mat tmpPos = pMPi->GetWorldPos();
                if (MappointInBoundingbox(tmpPos)) {
                    m_saved_mappoint_for_3dobject_.emplace_back(pMPi);
                    nTotalSize += pMPi->GetMemSizeFor3DObject(is_superpoint);
                }
            }
        }

        if (m_saved_mappoint_for_3dobject_.empty()) {
            LOG(ERROR) << "There is no mappoint in boundingbox, exit";
            return 0;
        }

        VLOG(0) << "getmemsize2" << nTotalSize;
        if (!is_superpoint) {
            for (Map *pMi : saved_map) {
                for (KeyFrame *pKFi : pMi->GetAllKeyFrames()) {

                    m_saved_keyframe_for_3dobject_.emplace_back(pKFi);
                }
            }
        }

        nTotalSize += sizeof(unsigned int);
        for (auto &item : m_saved_keyframe_for_3dobject_) {
            nTotalSize += item->GetMemSizeFor3DObject(is_superpoint);
            VLOG(5) << "get mem id: " << item->mnId << " " << nTotalSize;
        }
        VLOG(0) << "getmemsize3" << nTotalSize;
    }

    return nTotalSize;
}

bool Atlas::WriteToMemoryFor3DObject(
    const long long &mem_size, char *mem, const bool is_superpoint) {
    long long mem_pos = 0;
    const auto &camera_intrinsic =
        ObjRecognition::CameraIntrinsic::GetInstance();
    char version_str[sizeof(m_3dobject_version_)];
    Tools::PutDataToMem(
        mem + mem_pos, version_str, m_3dobject_version_.size(), mem_pos);
    double m_timestamp = 0.0;
    Tools::PutDataToMem(
        mem + mem_pos, &m_timestamp, sizeof(m_timestamp), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &camera_intrinsic.Width(), sizeof(int), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &camera_intrinsic.Height(), sizeof(int), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &camera_intrinsic.FX(), sizeof(double), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &camera_intrinsic.FY(), sizeof(double), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &camera_intrinsic.CX(), sizeof(double), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &camera_intrinsic.CY(), sizeof(double), mem_pos);

    double bounding_box[24];
    for (int i = 0; i < m_boundingbox_w_.size(); i++) {
        bounding_box[i * 3] = m_boundingbox_w_[i](0);
        bounding_box[i * 3 + 1] = m_boundingbox_w_[i](1);
        bounding_box[i * 3 + 2] = m_boundingbox_w_[i](2);
    }

    Tools::PutDataToMem(
        mem + mem_pos, bounding_box, 24 * sizeof(double), mem_pos);
    // box_scale
    Eigen::Vector3d m_box_scale = Eigen::Vector3d::Zero();
    Tools::PutDataToMem(
        mem + mem_pos, m_box_scale.data(), 3 * sizeof(double), mem_pos);

    // mappoint size:
    unsigned int nMPs = 0;
    unsigned int nKFs = 0;

    std::set<GeometricCamera *> spCams(mvpCameras.begin(), mvpCameras.end());
    VLOG(5) << "ORBSLAM3: There are " << spCams.size()
            << " cameras in the atlas";
    nMPs = m_saved_mappoint_for_3dobject_.size();
    nKFs = m_saved_keyframe_for_3dobject_.size();

    VLOG(10) << "write to memory for mappoints: " << nMPs;
    VLOG(10) << "write to memory for keyframes: " << nKFs;
    VLOG(5) << "writememsize1" << mem_pos;
    Tools::PutDataToMem(mem + mem_pos, &nMPs, sizeof(nMPs), mem_pos);

    Eigen::Matrix4d m_object_Two = Eigen::Matrix4d::Identity();
    for (MapPoint *pMPi : m_saved_mappoint_for_3dobject_) {
        pMPi->WriteToMemoryFor3DObject(
            mem_pos, mem, m_object_Two, is_superpoint);
    }
    VLOG(0) << "saved mappoint num: " << nMPs;
    VLOG(5) << "writememsize2" << mem_pos;

    // keyframe size:
    Tools::PutDataToMem(mem + mem_pos, &nKFs, sizeof(nKFs), mem_pos);

    int num = 0;
    for (KeyFrame *pKFi : m_saved_keyframe_for_3dobject_) {
        pKFi->WriteToMemoryFor3DObject(
            mem_pos, mem, m_object_Two, is_superpoint);
        VLOG(0) << "write mem id: " << num++ << " " << mem_pos;
    }

    VLOG(0) << "save keyframes num: " << nKFs;
    VLOG(0) << "writememsize3" << mem_pos;
    return mem_pos == mem_size;
}

void Atlas::SetScanBoundingbox_W(
    const std::vector<Eigen::Vector3d> &boundingbox) {
    m_boundingbox_w_.clear();
    m_boundingbox_w_ = boundingbox;
}

void Atlas::PreSave() {
    if (mpCurrentMap) {
        if (!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() +
                                1; // The init KF is the next of current maximum
    }

    struct compFunctor {
        inline bool operator()(Map *elem1, Map *elem2) {
            return elem1->GetId() < elem2->GetId();
        }
    };
    std::copy(
        mspMaps.begin(), mspMaps.end(), std::back_inserter(mvpBackupMaps));
    sort(mvpBackupMaps.begin(), mvpBackupMaps.end(), compFunctor());

    std::set<GeometricCamera *> spCams(mvpCameras.begin(), mvpCameras.end());
    VLOG(10) << "ORBSLAM3: There are " << spCams.size()
             << " cameras in the atlas";
    for (Map *pMi : mvpBackupMaps) {
        VLOG(10) << "ORBSLAM3: Pre-save of map " << pMi->GetId();
        pMi->PreSave(spCams);
    }
    VLOG(10) << "ORBSLAM3: Maps stored";
    for (GeometricCamera *pCam : mvpCameras) {
        VLOG(10) << "ORBSLAM3: Pre-save of camera " << pCam->GetId();
        if (pCam->GetType() == pCam->CAM_PINHOLE) {
            mvpBackupCamPin.push_back((Pinhole *)pCam);
        } else if (pCam->GetType() == pCam->CAM_FISHEYE) {
            mvpBackupCamKan.push_back((KannalaBrandt8 *)pCam);
        }
    }
}

void Atlas::PostLoad() {
    mvpCameras.clear();
    map<unsigned int, GeometricCamera *> mpCams;
    for (Pinhole *pCam : mvpBackupCamPin) {
        // mvpCameras.push_back((GeometricCamera*)pCam);
        mvpCameras.push_back(pCam);
        mpCams[pCam->GetId()] = pCam;
    }
    for (KannalaBrandt8 *pCam : mvpBackupCamKan) {
        // mvpCameras.push_back((GeometricCamera*)pCam);
        mvpCameras.push_back(pCam);
        mpCams[pCam->GetId()] = pCam;
    }

    mspMaps.clear();
    unsigned long int numKF = 0, numMP = 0;
    map<long unsigned int, KeyFrame *> mpAllKeyFrameId;
    for (Map *pMi : mvpBackupMaps) {
        VLOG(10) << "ORBSLAM3: Map id:" << pMi->GetId();
        mspMaps.insert(pMi);
        map<long unsigned int, KeyFrame *> mpKeyFrameId;
        pMi->PostLoad(mpKeyFrameDB, mpORBVocabulary, mpKeyFrameId, mpCams);
        mpAllKeyFrameId.insert(mpKeyFrameId.begin(), mpKeyFrameId.end());
        numKF += pMi->GetAllKeyFrames().size();
        numMP += pMi->GetAllMapPoints().size();
    }

    VLOG(10) << "ORBSLAM3: Number KF:" << numKF << "; number MP:" << numMP;
    mvpBackupMaps.clear();
}

void Atlas::SetKeyFrameDababase(KeyFrameDatabase *pKFDB) {
    mpKeyFrameDB = pKFDB;
}

KeyFrameDatabase *Atlas::GetKeyFrameDatabase() {
    return mpKeyFrameDB;
}

void Atlas::SetORBVocabulary(ORBVocabulary *pORBVoc) {
    mpORBVocabulary = pORBVoc;
}

ORBVocabulary *Atlas::GetORBVocabulary() {
    return mpORBVocabulary;
}

long unsigned int Atlas::GetNumLivedKF() {
    unique_lock<mutex> lock(mMutexAtlas);
    long unsigned int num = 0;
    for (Map *mMAPi : mspMaps) {
        num += mMAPi->GetAllKeyFrames().size();
    }

    return num;
}

long unsigned int Atlas::GetNumLivedMP() {
    unique_lock<mutex> lock(mMutexAtlas);
    long unsigned int num = 0;
    for (Map *mMAPi : mspMaps) {
        num += mMAPi->GetAllMapPoints().size();
    }

    return num;
}

} // namespace ORB_SLAM3
