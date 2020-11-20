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

#include "include/ORBSLAM3/MapPoint.h"
#include "include/ORBSLAM3/ORBmatcher.h"
#include "include/Tools.h"
#include "ORBSLAM3/SuperPointMatcher.h"
#include <mutex>

namespace ORB_SLAM3 {

long unsigned int MapPoint::nNextId = 0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint()
    : mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
      mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
      mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
      mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
      mpReplaced(static_cast<MapPoint *>(NULL)) {
    mpReplaced = static_cast<MapPoint *>(NULL);
}

MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap)
    : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
      mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
      mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
      mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF),
      mnVisible(1), mnFound(1), mbBad(false),
      mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0),
      mfMaxDistance(0), mpMap(pMap), mnOriginMapId(pMap->GetId()) {
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    mbTrackInViewR = false;
    mbTrackInView = false;

    // MapPoints can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

MapPoint::MapPoint(
    const double invDepth, cv::Point2f uv_init, KeyFrame *pRefKF,
    KeyFrame *pHostKF, Map *pMap)
    : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
      mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
      mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
      mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF),
      mnVisible(1), mnFound(1), mbBad(false),
      mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0),
      mfMaxDistance(0), mpMap(pMap), mnOriginMapId(pMap->GetId()) {
    mInvDepth = invDepth;
    mInitU = (double)uv_init.x;
    mInitV = (double)uv_init.y;
    mpHostKF = pHostKF;

    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    // Worldpos is not set
    // MapPoints can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

MapPoint::MapPoint(
    const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF)
    : mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0),
      mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
      mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
      mnCorrectedReference(0), mnBAGlobalForKF(0),
      mpRefKF(static_cast<KeyFrame *>(NULL)), mnVisible(1), mnFound(1),
      mbBad(false), mpReplaced(NULL), mpMap(pMap),
      mnOriginMapId(pMap->GetId()) {
    Pos.copyTo(mWorldPos);
    cv::Mat Ow;
    if (pFrame->Nleft == -1 || idxF < pFrame->Nleft) {
        Ow = pFrame->GetCameraCenter();
    } else {
        cv::Mat Rwl = pFrame->mRwc;
        cv::Mat tlr = pFrame->mTlr.col(3);
        cv::Mat twl = pFrame->mOw;

        Ow = Rwl * tlr + twl;
    }
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = (pFrame->Nleft == -1)
                          ? pFrame->mvKeysUn[idxF].octave
                          : (idxF < pFrame->Nleft)
                                ? pFrame->mvKeys[idxF].octave
                                : pFrame->mvKeysRight[idxF].octave;
    const float levelScaleFactor = pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal() {
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame *MapPoint::GetReferenceKeyFrame() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

void MapPoint::AddObservation(
    KeyFrame *pKF, int idx, const bool is_superpoint) {
    unique_lock<mutex> lock(mMutexFeatures);
    tuple<int, int> indexes;

    if (mObservations.count(pKF)) {
        indexes = mObservations[pKF];
    } else {
        indexes = tuple<int, int>(-1, -1);
    }

    if (pKF->NLeft != -1 && idx >= pKF->NLeft) {
        get<1>(indexes) = idx;
    } else {
        get<0>(indexes) = idx;
    }

    mObservations[pKF] = indexes;

    if (is_superpoint) {
        nObs++;
    } else {
        if (!pKF->mpCamera2 && pKF->mvuRight[idx] >= 0)
            nObs += 2;
        else
            nObs++;
    }
}

void MapPoint::EraseObservation(KeyFrame *pKF, bool is_superpoint) {
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF)) {
            // int idx = mObservations[pKF];
            tuple<int, int> indexes = mObservations[pKF];
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (!is_superpoint) {
                if (leftIndex != -1) {
                    if (!pKF->mpCamera2 && pKF->mvuRight[leftIndex] >= 0)
                        nObs -= 2;
                    else
                        nObs--;
                }
                if (rightIndex != -1) {
                    nObs--;
                }
            } else {
                if (leftIndex != -1) {
                    nObs--;
                }
            }
            mObservations.erase(pKF);
            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag(is_superpoint);
}

std::map<KeyFrame *, std::tuple<int, int>> MapPoint::GetObservations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag(bool is_superpoint) {
    map<KeyFrame *, tuple<int, int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(),
                                                    mend = obs.end();
         mit != mend; mit++) {
        KeyFrame *pKF = mit->first;
        int leftIndex = get<0>(mit->second), rightIndex = get<1>(mit->second);
        if (!is_superpoint) {
            if (leftIndex != -1) {
                pKF->EraseMapPointMatch(leftIndex);
            }
            if (rightIndex != -1) {
                pKF->EraseMapPointMatch(rightIndex);
            }
        } else {
            if (leftIndex != -1) {
                pKF->EraseMapPointMatch(leftIndex, true);
            }
        }
    }

    mpMap->EraseMapPoint(this);
}

MapPoint *MapPoint::GetReplaced() {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint *pMP) {
    if (pMP->mnId == this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame *, tuple<int, int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for (map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(),
                                                    mend = obs.end();
         mit != mend; mit++) {
        // Replace measurement in keyframe
        KeyFrame *pKF = mit->first;

        tuple<int, int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if (!pMP->IsInKeyFrame(pKF)) {
            if (leftIndex != -1) {
                pKF->ReplaceMapPointMatch(leftIndex, pMP);
                pMP->AddObservation(pKF, leftIndex);
            }
            if (rightIndex != -1) {
                pKF->ReplaceMapPointMatch(rightIndex, pMP);
                pMP->AddObservation(pKF, rightIndex);
            }
        } else {
            if (leftIndex != -1) {
                pKF->EraseMapPointMatch(leftIndex);
            }
            if (rightIndex != -1) {
                pKF->EraseMapPointMatch(rightIndex);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad() {
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapPoint::IncreaseVisible(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapPoint::IncreaseFound(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

float MapPoint::GetFoundRatio() {
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors(bool is_superpoint) {
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;
    map<KeyFrame *, tuple<int, int>> observations;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad)
            return;
        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(),
                                                    mend = observations.end();
         mit != mend; mit++) {
        KeyFrame *pKF = mit->first;
        if (!pKF->isBad()) {
            tuple<int, int> indexes = mit->second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (leftIndex != -1) {
                if (is_superpoint) {
                    vDescriptors.push_back(
                        pKF->mDescriptors_superpoint.row(leftIndex));
                } else {
                    vDescriptors.push_back(pKF->mDescriptors.row(leftIndex));
                }
            }

            if (!is_superpoint) {
                if (rightIndex != -1) {
                    vDescriptors.push_back(pKF->mDescriptors.row(rightIndex));
                }
            }
        }
    }

    if (vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++) {
            int distij;
            if (is_superpoint) {
                distij = ORB_SLAM3::SuperPointMatcher::DescriptorDistance(
                    vDescriptors[i], vDescriptors[j]);
            } else {
                distij = ORBmatcher::DescriptorDistance(
                    vDescriptors[i], vDescriptors[j]);
            }
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for (size_t i = 0; i < N; i++) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];

        if (median < BestMedian) {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

tuple<int, int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return tuple<int, int>(-1, -1);
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth(bool is_suerpoint) {
    map<KeyFrame *, tuple<int, int>> observations;
    KeyFrame *pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos.clone();
    }

    if (observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int n = 0;
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(),
                                                    mend = observations.end();
         mit != mend; mit++) {
        KeyFrame *pKF = mit->first;

        tuple<int, int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

        if (leftIndex != -1) {
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }
        if (!is_suerpoint) {
            if (rightIndex != -1) {
                cv::Mat Owi = pKF->GetRightCameraCenter();
                cv::Mat normali = mWorldPos - Owi;
                normal = normal + normali / cv::norm(normali);
                n++;
            }
        }
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);
    tuple<int, int> indexes = observations[pRefKF];
    int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    int level;

    if (pRefKF->NLeft == -1) {
        if (is_suerpoint) {
            level = pRefKF->mvKeysUn_superpoint[leftIndex].octave;
        } else {
            level = pRefKF->mvKeysUn[leftIndex].octave;
        }
    } else if (leftIndex != -1) {
        if (is_suerpoint) {
            level = pRefKF->mvKeys_superpoint[leftIndex].octave;
        } else {
            level = pRefKF->mvKeys[leftIndex].octave;
        }
    } else {
        if (!is_suerpoint) {
            level = pRefKF->mvKeysRight[rightIndex - pRefKF->NLeft].octave;
        }
    }

    // const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    float levelScaleFactor;
    if (is_suerpoint) {
        levelScaleFactor = pRefKF->mvScaleFactors_suerpoint[level];
    } else {
        levelScaleFactor = pRefKF->mvScaleFactors[level];
    }

    int nLevels;
    if (is_suerpoint) {
        nLevels = pRefKF->mnScaleLevels_suerpoint;
    } else {
        nLevels = pRefKF->mnScaleLevels;
    }

    if (is_suerpoint) {
        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance_superpoint = dist * levelScaleFactor;
            mfMinDistance_superpoint =
                mfMaxDistance_superpoint /
                pRefKF->mvScaleFactors_suerpoint[nLevels - 1];
            mNormalVector_superpoint = normal / n;
        }

    } else {
        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist * levelScaleFactor;
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            mNormalVector = normal / n;
        }
    }
}

void MapPoint::SetNormalVector(cv::Mat &normal) {
    unique_lock<mutex> lock3(mMutexPos);
    mNormalVector = normal;
}

float MapPoint::GetMinDistanceInvariance() {
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance() {
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF) {
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame *pF) {
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}

void MapPoint::PrintObservations() {
    cout << "MP_OBS: MP " << mnId << endl;
    for (map<KeyFrame *, tuple<int, int>>::iterator mit = mObservations.begin(),
                                                    mend = mObservations.end();
         mit != mend; mit++) {
        KeyFrame *pKFi = mit->first;
        tuple<int, int> indexes = mit->second;
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        cout << "--OBS in KF " << pKFi->mnId << " in map "
             << pKFi->GetMap()->GetId() << endl;
    }
}

Map *MapPoint::GetMap() {
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MapPoint::UpdateMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

long long MapPoint::GetMemSizeFor3DObject(
    const unsigned int start_sfm_keyframe_id, const int &descriptor_len,
    bool is_superpoint) {
    m_start_sfm_keyframe_id = start_sfm_keyframe_id;
    long long total_size = 0;
    total_size += sizeof(mnId);
    // position
    total_size += 3 * sizeof(double);
    // deps size
    total_size += sizeof(int);
    total_size += sizeof(mpRefKF->mnId);
    VLOG(5) << "mappoint 0: " << total_size;
    if (is_superpoint) {
        total_size += descriptor_len * sizeof(float);
    } else {
        total_size += descriptor_len * sizeof(uchar);
    }

    VLOG(5) << "mappoint 1: " << total_size;
    // ref_kf
    auto obs = std::move(GetObservations());
    total_size += sizeof(unsigned int);
    m_obs_for_sfm = 0;
    for (auto &curOb : obs) {
        if (curOb.first->mnId > start_sfm_keyframe_id) {
            total_size += sizeof(curOb.first);
            total_size += sizeof(get<0>(curOb.second));
            m_obs_for_sfm++;
        }
    }

    total_size += sizeof(mnVisible);
    total_size += sizeof(mnFound);
    VLOG(5) << "mappoint 2: " << total_size;
    return total_size;
}

void MapPoint::WriteToMemoryFor3DObject(
    long long &mem_pos, char *mem, const int &descriptor_len,
    bool is_superpoint) {
    Tools::PutDataToMem(mem + mem_pos, &mnId, sizeof(mnId), mem_pos);
    Eigen::Vector3d pos = Eigen::Vector3d(
        mWorldPos.at<float>(0), mWorldPos.at<float>(1), mWorldPos.at<float>(2));
    Tools::PutDataToMem(mem + mem_pos, &(pos(0)), sizeof(double), mem_pos);
    Tools::PutDataToMem(mem + mem_pos, &(pos(1)), sizeof(double), mem_pos);
    Tools::PutDataToMem(mem + mem_pos, &(pos(2)), sizeof(double), mem_pos);

    int deps_size = 1;
    Tools::PutDataToMem(mem + mem_pos, &deps_size, sizeof(deps_size), mem_pos);
    Tools::PutDataToMem(
        mem + mem_pos, &mpRefKF->mnId, sizeof(mpRefKF->mnId), mem_pos);
    auto obs = GetObservations();
    cv::Mat desp = GetDescriptor();
    if (is_superpoint) {
        Tools::PutDataToMem(
            mem + mem_pos, desp.data, descriptor_len * sizeof(float), mem_pos);
    } else {
        Tools::PutDataToMem(
            mem + mem_pos, desp.data, descriptor_len * sizeof(uchar), mem_pos);
    }

    unsigned int obSize = m_obs_for_sfm;
    Tools::PutDataToMem(mem + mem_pos, &obSize, sizeof(obSize), mem_pos);

    for (auto &curOb : obs) {
        if (curOb.first->mnId > m_start_sfm_keyframe_id) {
            Tools::PutDataToMem(
                mem + mem_pos, &(curOb.first->mnId), sizeof(curOb.first->mnId),
                mem_pos);
            Tools::PutDataToMem(
                mem + mem_pos, &get<0>(curOb.second),
                sizeof(get<0>(curOb.second)), mem_pos);
        }
    }

    Tools::PutDataToMem(mem + mem_pos, &mnVisible, sizeof(mnVisible), mem_pos);
    Tools::PutDataToMem(mem + mem_pos, &mnFound, sizeof(mnFound), mem_pos);
}

void MapPoint::PreSave(set<KeyFrame *> &spKF, set<MapPoint *> &spMP) {
    mBackupReplacedId = -1;
    if (mpReplaced && spMP.find(mpReplaced) != spMP.end())
        mBackupReplacedId = mpReplaced->mnId;

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
    // Save the id and position in each KF who view it
    for (std::map<KeyFrame *, std::tuple<int, int>>::const_iterator
             it = mObservations.begin(),
             end = mObservations.end();
         it != end; ++it) {
        KeyFrame *pKFi = it->first;
        if (spKF.find(pKFi) != spKF.end()) {
            mBackupObservationsId1[it->first->mnId] = get<0>(it->second);
            mBackupObservationsId2[it->first->mnId] = get<1>(it->second);
        } else {
            EraseObservation(pKFi);
        }
    }

    // Save the id of the reference KF
    if (spKF.find(mpRefKF) != spKF.end()) {
        mBackupRefKFId = mpRefKF->mnId;
    }
}

void MapPoint::PostLoad(
    map<long unsigned int, KeyFrame *> &mpKFid,
    map<long unsigned int, MapPoint *> &mpMPid) {
    mpRefKF = mpKFid[mBackupRefKFId];
    if (!mpRefKF) {
        cout << "MP without KF reference " << mBackupRefKFId
             << "; Num obs: " << nObs << endl;
    }
    mpReplaced = static_cast<MapPoint *>(NULL);
    if (mBackupReplacedId >= 0) {
        map<long unsigned int, MapPoint *>::iterator it =
            mpMPid.find(mBackupReplacedId);
        if (it != mpMPid.end())
            mpReplaced = it->second;
    }

    mObservations.clear();

    for (map<long unsigned int, int>::const_iterator
             it = mBackupObservationsId1.begin(),
             end = mBackupObservationsId1.end();
         it != end; ++it) {
        KeyFrame *pKFi = mpKFid[it->first];
        map<long unsigned int, int>::const_iterator it2 =
            mBackupObservationsId2.find(it->first);
        std::tuple<int, int> indexes = tuple<int, int>(it->second, it2->second);
        if (pKFi) {
            mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId1.clear();
    mBackupObservationsId2.clear();
}

} // namespace ORB_SLAM3
