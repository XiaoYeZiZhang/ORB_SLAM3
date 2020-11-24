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

#include "include/ORBSLAM3/KeyFrame.h"
#include "include/ORBSLAM3/Converter.h"
#include "include/ORBSLAM3/ImuTypes.h"
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <include/CameraModels/Pinhole.h>
#include <glog/logging.h>
#include "include/Tools.h"
#include "ObjectRecognition/Utility/Camera.h"
#include "mode.h"

namespace ORB_SLAM3 {

long unsigned int KeyFrame::nNextId = 0;

KeyFrame::KeyFrame()
    : mnFrameId(0), mTimeStamp(0), mnGridCols(FRAME_GRID_COLS),
      mnGridRows(FRAME_GRID_ROWS), mfGridElementWidthInv(0),
      mfGridElementHeightInv(0), mnTrackReferenceForFrame(0),
      mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
      mnBALocalForMerge(0), mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0),
      mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
      fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0),
      mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0),
      mPlaceRecognitionScore(0), mbf(0), mb(0), mThDepth(0), N(0),
      mvKeys(static_cast<vector<cv::KeyPoint>>(NULL)),
      mvKeysUn(static_cast<vector<cv::KeyPoint>>(NULL)),
      mvuRight(static_cast<vector<float>>(NULL)),
      mvDepth(static_cast<vector<float>>(NULL)), /*mDescriptors(NULL),*/
      /*mBowVec(NULL), mFeatVec(NULL),*/ mnScaleLevels(0), mfScaleFactor(0),
      mfLogScaleFactor(0), mvScaleFactors(0), mvLevelSigma2(0),
      mvInvLevelSigma2(0), mnMinX(0), mnMinY(0), mnMaxX(0), mnMaxY(0),
      /*mK(NULL),*/ mPrevKF(static_cast<KeyFrame *>(NULL)),
      mNextKF(static_cast<KeyFrame *>(NULL)), mbFirstConnection(true),
      mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
      mHalfBaseline(0), mbCurrentPlaceRecognition(false), mbHasHessian(false),
      mnMergeCorrectedForKF(0), NLeft(0), NRight(0), mnNumberOfOpt(0),
      mnNumberOfOpt_Superpoint(0), N_superpoint(-1) {
}

void KeyFrame::UndistortKeyPoints() {
    if (mDistCoef.at<float>(0) == 0.0) {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F);

    for (int i = 0; i < N; i++) {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(
        mat, mat, static_cast<Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(),
        mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++) {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)
    : bImu(pMap->isImuInitialized()), mnFrameId(F.mnId),
      mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS),
      mnGridRows(FRAME_GRID_ROWS),
      mfGridElementWidthInv(F.mfGridElementWidthInv),
      mfGridElementHeightInv(F.mfGridElementHeightInv),
      mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0),
      mnBAFixedForKF(0), mnBALocalForMerge(0), mnLoopQuery(0), mnLoopWords(0),
      mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
      mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0),
      mPlaceRecognitionScore(0), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy),
      invfx(F.invfx), invfy(F.invfy), mbf(F.mbf), mb(F.mb),
      mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
      mvuRight(F.mvuRight), mvDepth(F.mvDepth),
      mDescriptors(F.mDescriptors.clone()), mBowVec(F.mBowVec),
      mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels),
      mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor),
      mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
      mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY),
      mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK(F.mK), mPrevKF(NULL),
      mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated),
      mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
      mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true),
      mpParent(NULL), mDistCoef(F.mDistCoef), mbNotErase(false),
      mnDataset(F.mnDataset), mbToBeErased(false), mbBad(false),
      mHalfBaseline(F.mb / 2), mpMap(pMap), mbCurrentPlaceRecognition(false),
      mNameFile(F.mNameFile), mbHasHessian(false), mnMergeCorrectedForKF(0),
      mpCamera(F.mpCamera), mpCamera2(F.mpCamera2),
      mvLeftToRightMatch(F.mvLeftToRightMatch),
      mvRightToLeftMatch(F.mvRightToLeftMatch), mTlr(F.mTlr.clone()),
      mvKeysRight(F.mvKeysRight), NLeft(F.Nleft), NRight(F.Nright),
      mTrl(F.mTrl), mnNumberOfOpt(0), mnNumberOfOpt_Superpoint(0),
      N_superpoint(-1) {

    imgLeft = F.imgLeft.clone();
    imgRight = F.imgRight.clone();

    mnId = nNextId++;

    mGrid.resize(mnGridCols);
    if (F.Nleft != -1)
        mGridRight.resize(mnGridCols);
    for (int i = 0; i < mnGridCols; i++) {
        mGrid[i].resize(mnGridRows);
        if (F.Nleft != -1)
            mGridRight[i].resize(mnGridRows);
        for (int j = 0; j < mnGridRows; j++) {
            mGrid[i][j] = F.mGrid[i][j];
            if (F.Nleft != -1) {
                mGridRight[i][j] = F.mGridRight[i][j];
            }
        }
    }

    if (F.mVw.empty())
        Vw = cv::Mat::zeros(3, 1, CV_32F);
    else
        Vw = F.mVw.clone();

    mImuBias = F.mImuBias;
    SetPose(F.mTcw);

    mnOriginMapId = pMap->GetId();
}

void KeyFrame::ComputeBoW() {
    if (mBowVec.empty() || mFeatVec.empty()) {
        vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from
        // leaves up) We assume the vocabulary tree has 6 levels, change the 4
        // otherwise
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void KeyFrame::ComputeBoW_SuperPoint(
    const ORB_SLAM3::SUPERPOINTVocabulary *superpoint_voc) {
    if (mBowVec_superpoint.empty() || mFeatVec_superpoint.empty()) {
        vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(mDescriptors_superpoint);
        superpoint_voc->transform(
            vCurrentDesc, mBowVec_superpoint, mFeatVec_superpoint, 4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_) {
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc * tcw;
    if (!mImuCalib.Tcb.empty())
        Owb = Rwc * mImuCalib.Tcb.rowRange(0, 3).col(3) + Ow;

    Twc = cv::Mat::eye(4, 4, Tcw.type());
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(Twc.rowRange(0, 3).col(3));
    cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
    Cw = Twc * center;
}

void KeyFrame::SetVelocity(const cv::Mat &Vw_) {
    unique_lock<mutex> lock(mMutexPose);
    Vw_.copyTo(Vw);
}

cv::Mat KeyFrame::GetPose() {
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter() {
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}

cv::Mat KeyFrame::GetImuPosition() {
    unique_lock<mutex> lock(mMutexPose);
    return Owb.clone();
}

cv::Mat KeyFrame::GetImuRotation() {
    unique_lock<mutex> lock(mMutexPose);
    return Twc.rowRange(0, 3).colRange(0, 3) *
           mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3);
}

cv::Mat KeyFrame::GetImuPose() {
    unique_lock<mutex> lock(mMutexPose);
    return Twc * mImuCalib.Tcb;
}

cv::Mat KeyFrame::GetRotation() {
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}

cv::Mat KeyFrame::GetTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0, 3).col(3).clone();
}

cv::Mat KeyFrame::GetVelocity() {
    unique_lock<mutex> lock(mMutexPose);
    return Vw.clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight) {
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        else if (mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles() {
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                        mend = mConnectedKeyFrameWeights.end();
         mit != mend; mit++)
        vPairs.push_back(make_pair(mit->second, mit->first));

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame *> lKFs;
    list<int> lWs;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
        if (!vPairs[i].second->isBad()) {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame *> KeyFrame::GetConnectedKeyFrames() {
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame *> s;
    for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin();
         mit != mConnectedKeyFrameWeights.end(); mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames() {
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) {
    unique_lock<mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame *>(
            mvpOrderedConnectedKeyFrames.begin(),
            mvpOrderedConnectedKeyFrames.begin() + N);
}

vector<KeyFrame *> KeyFrame::GetCovisiblesByWeight(const int &w) {
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty()) {
        return vector<KeyFrame *>();
    }

    vector<int>::iterator it = upper_bound(
        mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
        KeyFrame::weightComp);

    if (it == mvOrderedWeights.end() && mvOrderedWeights.back() < w) {
        return vector<KeyFrame *>();
    } else {
        int n = it - mvOrderedWeights.begin();
        return vector<KeyFrame *>(
            mvpOrderedConnectedKeyFrames.begin(),
            mvpOrderedConnectedKeyFrames.begin() + n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF) {
    unique_lock<mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

int KeyFrame::GetNumberMPs() {
    unique_lock<mutex> lock(mMutexFeatures);
    int numberMPs = 0;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++) {
        if (!mvpMapPoints[i])
            continue;
        numberMPs++;
    }
    return numberMPs;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

void KeyFrame::AddSuperpointMapPoint(MapPoint *pMP, const size_t &idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints_superpoint[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx, const bool is_superpoint) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (is_superpoint) {
        mvpMapPoints_superpoint[idx] = static_cast<MapPoint *>(NULL);
    } else {
        mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }
}

void KeyFrame::EraseMapPointMatch(MapPoint *pMP, const bool is_superpoint) {
    tuple<size_t, size_t> indexes = pMP->GetIndexInKeyFrame(this);
    size_t leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if (!is_superpoint) {
        if (leftIndex != -1)
            mvpMapPoints[leftIndex] = static_cast<MapPoint *>(NULL);
        if (rightIndex != -1)
            mvpMapPoints[rightIndex] = static_cast<MapPoint *>(NULL);
    } else {
        if (leftIndex != -1)
            mvpMapPoints_superpoint[leftIndex] = static_cast<MapPoint *>(NULL);
    }
}

void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint *pMP) {
    mvpMapPoints[idx] = pMP;
}

set<MapPoint *> KeyFrame::GetMapPoints() {
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint *> s;
    for (size_t i = 0, iend = mvpMapPoints.size(); i < iend; i++) {
        if (!mvpMapPoints[i])
            continue;
        MapPoint *pMP = mvpMapPoints[i];
        if (!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs) {
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    for (int i = 0; i < N; i++) {
        MapPoint *pMP = mvpMapPoints[i];
        if (pMP) {
            if (!pMP->isBad()) {
                if (bCheckObs) {
                    if (mvpMapPoints[i]->Observations() >= minObs)
                        nPoints++;
                } else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint *> KeyFrame::GetMapPointMatches() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

vector<MapPoint *> KeyFrame::GetMapPointMatches_SuperPoint() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints_superpoint;
}

MapPoint *KeyFrame::GetMapPoint(const size_t &idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

MapPoint *KeyFrame::GetSuperpointMapPoint(const size_t &idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints_superpoint[idx];
}

void KeyFrame::UpdateConnections(bool upParent) {
    map<KeyFrame *, int> KFcounter;

    vector<MapPoint *> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    // For all map points in keyframe check in which other keyframes are they
    // seen Increase counter for those keyframes
    for (vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end();
         vit != vend; vit++) {
        MapPoint *pMP = *vit;

        if (!pMP)
            continue;

        if (pMP->isBad())
            continue;

        map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

        for (map<KeyFrame *, tuple<int, int>>::iterator
                 mit = observations.begin(),
                 mend = observations.end();
             mit != mend; mit++) {
            if (mit->first->mnId == mnId || mit->first->isBad() ||
                mit->first->GetMap() != mpMap)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum
    // counter
    int nmax = 0;
    KeyFrame *pKFmax = NULL;
    int th = 15;

    vector<pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(KFcounter.size());
    if (!upParent)
        VLOG(5) << "ORBSLAM3: UPDATE_CONN: current KF " << mnId;
    for (map<KeyFrame *, int>::iterator mit = KFcounter.begin(),
                                        mend = KFcounter.end();
         mit != mend; mit++) {
        if (!upParent)
            VLOG(5) << "  UPDATE_CONN: KF " << mit->first->mnId
                    << " ; num matches: " << mit->second;
        if (mit->second > nmax) {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th) {
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->AddConnection(this, mit->second);
        }
    }

    if (vPairs.empty()) {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(this, nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFrame *> lKFs;
    list<int> lWs;
    for (size_t i = 0; i < vPairs.size(); i++) {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames =
            vector<KeyFrame *>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        //        if(mbFirstConnection && mnId!=mpMap->GetInitKFid())
        //        {
        //            mpParent = mvpOrderedConnectedKeyFrames.front();
        //            mpParent->AddChild(this);
        //            mbFirstConnection = false;
        //        }

        if (mbFirstConnection && mnId != mpMap->GetInitKFid()) {
            /*if(!mpParent || mpParent->GetParent() != this)
            {
                KeyFrame* pBestParent = static_cast<KeyFrame*>(NULL);
                for(KeyFrame* pKFi : mvpOrderedConnectedKeyFrames)
                {
                    if(pKFi->GetParent() || pKFi->mnId == mpMap->GetInitKFid())
                    {
                        pBestParent = pKFi;
                        break;
                    }
                }
                if(!pBestParent)
                {
                    cout << "It can't be a covisible KF without Parent" << endl
            << endl; return;
                }
                mpParent = pBestParent;
                mpParent->AddChild(this);
                mbFirstConnection = false;
            }*/
            // cout << "udt.conn.id: " << mnId << endl;
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}

void KeyFrame::AddChild(KeyFrame *pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    //    if(!mpParent && mpParent != this)
    //        mpParent->EraseChild(this);
    if (pKF == this) {
        LOG(FATAL) << "ORBSLAM3 ERROR: Change parent KF, the parent and child "
                      "are the same KF";
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame *> KeyFrame::GetChilds() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame *KeyFrame::GetParent() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::SetFirstConnection(bool bFirst) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection = bFirst;
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame *> KeyFrame::GetLoopEdges() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::AddMergeEdge(KeyFrame *pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFrame *> KeyFrame::GetMergeEdges() {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

void KeyFrame::SetNotErase() {
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase() {
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mspLoopEdges.empty()) {
            mbNotErase = false;
        }
    }

    if (mbToBeErased) {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag() {
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mnId == mpMap->GetInitKFid()) {
            return;
        } else if (mbNotErase) {
            mbToBeErased = true;
            return;
        }
        if (!mpParent) {
            // cout << "KF.BADFLAG-> There is not parent, but it is not the
            // first KF in the map" << endl; cout << "KF.BADFLAG-> KF: " << mnId
            // << "; first KF: " << mpMap->GetInitKFid() << endl;
        }
    }
    // std::cout << "KF.BADFLAG-> Erasing KF..." << std::endl;

    for (map<KeyFrame *, int>::iterator mit = mConnectedKeyFrameWeights.begin(),
                                        mend = mConnectedKeyFrameWeights.end();
         mit != mend; mit++) {
        mit->first->EraseConnection(this);
    }
    // std::cout << "KF.BADFLAG-> Connection erased..." << std::endl;

    for (size_t i = 0; i < mvpMapPoints.size(); i++) {
        if (mvpMapPoints[i]) {
            mvpMapPoints[i]->EraseObservation(this);
            // nDeletedPoints++;
        }
    }
    // cout << "nDeletedPoints: " << nDeletedPoints << endl;
    // std::cout << "KF.BADFLAG-> Observations deleted..." << std::endl;

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame *> sParentCandidates;
        if (mpParent)
            sParentCandidates.insert(mpParent);
        // std::cout << "KF.BADFLAG-> Initially there are " <<
        // sParentCandidates.size() << " candidates" << std::endl;

        // Assign at each iteration one children with a parent (the pair with
        // highest covisibility weight) Include that children as new parent
        // candidate for the rest
        while (!mspChildrens.empty()) {
            bool bContinue = false;

            int max = -1;
            KeyFrame *pC;
            KeyFrame *pP;

            for (set<KeyFrame *>::iterator sit = mspChildrens.begin(),
                                           send = mspChildrens.end();
                 sit != send; sit++) {
                KeyFrame *pKF = *sit;
                if (pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame *> vpConnected =
                    pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++) {
                    for (set<KeyFrame *>::iterator
                             spcit = sParentCandidates.begin(),
                             spcend = sParentCandidates.end();
                         spcit != spcend; spcit++) {
                        if (vpConnected[i]->mnId == (*spcit)->mnId) {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w > max) {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }
            // std::cout << "KF.BADFLAG-> Find most similar children" <<
            // std::endl;

            if (bContinue) {
                if (pC->mnId == pP->mnId) {
                    /*cout << "ERROR: The parent and son can't be the same KF.
                    ID: " << pC->mnId << endl; cout << "Current KF: " << mnId <<
                    endl; cout << "Parent of the map: " << endl;*/
                }
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            } else
                break;
        }
        // std::cout << "KF.BADFLAG-> Apply change of parent to children" <<
        // std::endl;

        // If a children has no covisibility links with any parent candidate,
        // assign to the original parent of this KF
        if (!mspChildrens.empty()) {
            for (set<KeyFrame *>::iterator sit = mspChildrens.begin();
                 sit != mspChildrens.end(); sit++) {
                (*sit)->ChangeParent(mpParent);
            }
        }
        // std::cout << "KF.BADFLAG-> Apply change to its parent" << std::endl;

        if (mpParent) {
            mpParent->EraseChild(this);
            mTcp = Tcw * mpParent->GetPoseInverse();
        } else {
            // cout << "Error: KF haven't got a parent, it is imposible reach
            // this code point without him" << endl;
        }
        mbBad = true;
    }

    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad() {
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame *pKF) {
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF)) {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea_Superpoint(
    const float &x, const float &y, const float &r) {
    vector<size_t> vIndices;
    vIndices.reserve(N_superpoint);
    for (int k = 0; k < mvKeysUn_superpoint.size(); k++) {
        const float distx = mvKeysUn_superpoint[k].pt.x - x;
        const float disty = mvKeysUn_superpoint[k].pt.y - y;

        if (fabs(distx) < r && fabs(disty) < r) {
            vIndices.emplace_back(k);
        }
    }

    return vIndices;
}

vector<size_t> KeyFrame::GetFeaturesInArea(
    const float &x, const float &y, const float &r, const bool bRight) const {
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX =
        max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
        return vIndices;

    const int nMaxCellX =
        min((int)mnGridCols - 1,
            (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY =
        max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
        return vIndices;

    const int nMaxCellY =
        min((int)mnGridRows - 1,
            (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            const vector<size_t> vCell =
                (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const cv::KeyPoint &kpUn =
                    (NLeft == -1)
                        ? mvKeysUn[vCell[j]]
                        : (!bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const {
    return (x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i) {
    const float z = mvDepth[i];
    if (z > 0) {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc +
               Twc.rowRange(0, 3).col(3);
    } else
        return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q) {
    vector<MapPoint *> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2, 3);
    for (int i = 0; i < N; i++) {
        if (mvpMapPoints[i]) {
            MapPoint *pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(), vDepths.end());

    return vDepths[(vDepths.size() - 1) / q];
}

void KeyFrame::SetNewBias(const IMU::Bias &b) {
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if (mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

cv::Mat KeyFrame::GetGyroBias() {
    unique_lock<mutex> lock(mMutexPose);
    return (cv::Mat_<float>(3, 1) << mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

cv::Mat KeyFrame::GetAccBias() {
    unique_lock<mutex> lock(mMutexPose);
    return (cv::Mat_<float>(3, 1) << mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias() {
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

Map *KeyFrame::GetMap() {
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::UpdateMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void KeyFrame::SetMap_SuperPoint(Map *pMap_SuperPoint) {
    unique_lock<mutex> lock(mMutexMap);
    mpMap_Superpoint = pMap_SuperPoint;
}

Map *KeyFrame::GetMap_SuperPoint() {
    unique_lock<mutex> lock(mMutexMap);
    return mpMap_Superpoint;
}

// TODO(zhangye): need more variables to reset???
void KeyFrame::SetKeyPoints(std::vector<cv::KeyPoint> &keypoints) {
    mvKeys = keypoints;
    N = mvKeys.size();
    UndistortKeyPoints();
    int originN = mvpMapPoints.size();
    mvpMapPoints.resize(N);
    for (int i = originN; i < mvpMapPoints.size(); i++) {
        mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
    }
}

void KeyFrame::SetKeyPoints_Superpoints() {
    VLOG(0) << "keyframe superpoint keypoint size: "
            << mvKeys_superpoint.size();
    N_superpoint = mvKeys_superpoint.size();
    VLOG(5) << "mappoint id, keypoint num: " << mnId << " " << N_superpoint;
    mvKeysUn_superpoint = mvKeys_superpoint;
    mvpMapPoints_superpoint.resize(N_superpoint);
    for (int i = 0; i < mvpMapPoints_superpoint.size(); i++) {
        mvpMapPoints_superpoint[i] = static_cast<MapPoint *>(NULL);
    }
}

void KeyFrame::SetDesps(const cv::Mat &desps) {
    mDescriptors = desps;
}

long long KeyFrame::GetMemSizeFor3DObject(
    const int start_sfm_keyframe_id, const int &descriptor_len,
    const bool is_superpoint) {
    long long totalSize = 0;
    totalSize += sizeof(mnId);
    VLOG(5) << "getmem key 1: " << totalSize;

#ifdef SAVE_CONNECT_FOR_DETECTOR
    // connect keyframe num
    saved_connected_keyframes_for3DObject.clear();
    vector<KeyFrame *> covis_keyframes = GetBestCovisibilityKeyFrames(15);
    for (auto keyframe : covis_keyframes) {
        if (start_sfm_keyframe_id == -1) {
            saved_connected_keyframes_for3DObject.emplace_back(keyframe);
        } else {
            if (keyframe->mnId > (long unsigned int)start_sfm_keyframe_id) {
                saved_connected_keyframes_for3DObject.emplace_back(keyframe);
            }
        }
    }
    long unsigned int connect_kfs_size =
        saved_connected_keyframes_for3DObject.size();
    totalSize += sizeof(connect_kfs_size);
    totalSize += ((sizeof(mnId)) * connect_kfs_size);

    if (!is_superpoint) {
        for (auto mappoint : mvpMapPoints) {
            if (mappoint) {
                saved_connected_mappoints_for3DObject.emplace_back(mappoint);
            }
        }

        long unsigned int connect_mappoint_size =
            saved_connected_mappoints_for3DObject.size();
        totalSize += sizeof(connect_mappoint_size);
        totalSize += sizeof(long unsigned int) * connect_mappoint_size;
    } else {
        for (auto mappoint : mvpMapPoints_superpoint) {
            if (mappoint) {
                saved_connected_mappoints_for3DObject.emplace_back(mappoint);
            }
        }
        long unsigned int connect_mappoint_size =
            saved_connected_mappoints_for3DObject.size();
        totalSize += sizeof(connect_mappoint_size);
        totalSize += sizeof(long unsigned int) * connect_mappoint_size;
    }
#endif

    unsigned int nKpts = mvKeys.size();
    if (is_superpoint) {
        nKpts = mvKeys_superpoint.size();
    }

    totalSize += sizeof(nKpts);
    VLOG(5) << "getmem key 2: " << totalSize;

    if (nKpts > 0) {
        const unsigned int PerKeyPointSize =
            sizeof(float) * 5 + 2 * sizeof(int);
        totalSize += nKpts * PerKeyPointSize;
        VLOG(5) << "getmem key 3: " << totalSize;
        if (is_superpoint) {
            totalSize += nKpts * descriptor_len * sizeof(float);
        } else {
            totalSize += nKpts * 32 * sizeof(uchar);
        }
        VLOG(5) << "getmem key 4: " << totalSize;
    }

    // Tco
    totalSize += sizeof(double) * 7;
    VLOG(5) << "getmem key 5: " << totalSize;
    // upload image data
    totalSize += sizeof(char) *
                 ObjRecognition::CameraIntrinsic::GetInstance().Width() *
                 ObjRecognition::CameraIntrinsic::GetInstance().Height();
    VLOG(5) << "getmem key 6: " << totalSize;
    return totalSize;
}

void KeyFrame::WriteToMemoryFor3DObject(
    long long &mem_pos, char *mem, const Eigen::Matrix4d &Two,
    const bool is_superpoint) {
    VLOG(10) << "keyframe id: " << mnId;
    Tools::PutDataToMem(mem + mem_pos, &mnId, sizeof(mnId), mem_pos);

#ifdef SAVE_CONNECT_FOR_DETECTOR
    // connect keyframe id
    long unsigned int connect_kfs_size =
        saved_connected_keyframes_for3DObject.size();
    Tools::PutDataToMem(
        mem + mem_pos, &connect_kfs_size, sizeof(connect_kfs_size), mem_pos);
    for (auto keyframe : saved_connected_keyframes_for3DObject) {
        Tools::PutDataToMem(
            mem + mem_pos, &(keyframe->mnId), sizeof(keyframe->mnId), mem_pos);
    }

    // associated mappoint id:
    long unsigned int connect_mappoints_num =
        saved_connected_mappoints_for3DObject.size();
    Tools::PutDataToMem(
        mem + mem_pos, &connect_mappoints_num, sizeof(connect_mappoints_num),
        mem_pos);
    for (auto mappoint : saved_connected_mappoints_for3DObject) {
        Tools::PutDataToMem(
            mem + mem_pos, &(mappoint->mnId), sizeof(mappoint->mnId), mem_pos);
    }

#endif

    VLOG(5) << "write mem key 1:" << sizeof(mnId);
    if (is_superpoint) {
        Tools::PackSUPERPOINTFeatures(
            mvKeys_superpoint, mDescriptors_superpoint, mem_pos, mem);
    } else {
        Tools::PackORBFeatures(mvKeys, mDescriptors, mem_pos, mem);
    }

    Eigen::Vector3d Tcw;
    Eigen::Matrix3d Rcw;

    Eigen::Matrix4d Tcw_4_4;
    cv::Mat Tcw_mat = GetPose();
    cv::cv2eigen(Tcw_mat, Tcw_4_4);

    Rcw = Tcw_4_4.block<3, 3>(0, 0);
    Tcw = Tcw_4_4.block<3, 1>(0, 3);

    Eigen::Matrix3d Rco = Rcw * Two.block<3, 3>(0, 0);
    Eigen::Vector3d tco = Rcw * Two.block<3, 1>(0, 3) + Tcw;
    auto init = mem_pos;
    Tools::PackCamCWToMem(tco, Rco, mem_pos, mem);
    VLOG(5) << "write mem key 5:" << mem_pos - init;

    VLOG(10) << "size: "
             << ObjRecognition::CameraIntrinsic::GetInstance().Width() << " "
             << ObjRecognition::CameraIntrinsic::GetInstance().Height();
    Tools::PutDataToMem(
        mem + mem_pos, imgLeft.data,
        sizeof(char) * ObjRecognition::CameraIntrinsic::GetInstance().Width() *
            ObjRecognition::CameraIntrinsic::GetInstance().Height(),
        mem_pos);
    VLOG(5) << "write mem key 6:"
            << sizeof(char) *
                   ObjRecognition::CameraIntrinsic::GetInstance().Width() *
                   ObjRecognition::CameraIntrinsic::GetInstance().Height();
}

void KeyFrame::PreSave(
    set<KeyFrame *> &spKF, set<MapPoint *> &spMP,
    set<GeometricCamera *> &spCam) {
    // Save the id of each MapPoint in this KF, there can be null pointer in the
    // vector
    mvBackupMapPointsId.clear();
    mvBackupMapPointsId.reserve(N);
    for (int i = 0; i < N; ++i) {

        if (mvpMapPoints[i] &&
            spMP.find(mvpMapPoints[i]) !=
                spMP.end()) // Checks if the element is not null
            mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
        else // If the element is null his value is -1 because all the id are
             // positives
            mvBackupMapPointsId.push_back(-1);
    }
    // cout << "KeyFrame: ID from MapPoints stored" << endl;
    // Save the id of each connected KF with it weight
    mBackupConnectedKeyFrameIdWeights.clear();
    for (std::map<KeyFrame *, int>::const_iterator
             it = mConnectedKeyFrameWeights.begin(),
             end = mConnectedKeyFrameWeights.end();
         it != end; ++it) {
        if (spKF.find(it->first) != spKF.end())
            mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
    }
    // cout << "KeyFrame: ID from connected KFs stored" << endl;
    // Save the parent id
    mBackupParentId = -1;
    if (mpParent && spKF.find(mpParent) != spKF.end())
        mBackupParentId = mpParent->mnId;
    // cout << "KeyFrame: ID from Parent KF stored" << endl;
    // Save the id of the childrens KF
    mvBackupChildrensId.clear();
    mvBackupChildrensId.reserve(mspChildrens.size());
    for (KeyFrame *pKFi : mspChildrens) {
        if (spKF.find(pKFi) != spKF.end())
            mvBackupChildrensId.push_back(pKFi->mnId);
    }
    // cout << "KeyFrame: ID from Children KFs stored" << endl;
    // Save the id of the loop edge KF
    mvBackupLoopEdgesId.clear();
    mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
    for (KeyFrame *pKFi : mspLoopEdges) {
        if (spKF.find(pKFi) != spKF.end())
            mvBackupLoopEdgesId.push_back(pKFi->mnId);
    }
    // cout << "KeyFrame: ID from Loop KFs stored" << endl;
    // Save the id of the merge edge KF
    mvBackupMergeEdgesId.clear();
    mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
    for (KeyFrame *pKFi : mspMergeEdges) {
        if (spKF.find(pKFi) != spKF.end())
            mvBackupMergeEdgesId.push_back(pKFi->mnId);
    }
    // cout << "KeyFrame: ID from Merge KFs stored" << endl;

    // Camera data
    mnBackupIdCamera = -1;
    if (mpCamera && spCam.find(mpCamera) != spCam.end())
        mnBackupIdCamera = mpCamera->GetId();
    // cout << "KeyFrame: ID from Camera1 stored; " << mnBackupIdCamera << endl;

    mnBackupIdCamera2 = -1;
    if (mpCamera2 && spCam.find(mpCamera2) != spCam.end())
        mnBackupIdCamera2 = mpCamera2->GetId();
    // cout << "KeyFrame: ID from Camera2 stored; " << mnBackupIdCamera2 <<
    // endl;

    // Inertial data
    mBackupPrevKFId = -1;
    if (mPrevKF && spKF.find(mPrevKF) != spKF.end())
        mBackupPrevKFId = mPrevKF->mnId;
    // cout << "KeyFrame: ID from Prev KF stored" << endl;
    mBackupNextKFId = -1;
    if (mNextKF && spKF.find(mNextKF) != spKF.end())
        mBackupNextKFId = mNextKF->mnId;
    // cout << "KeyFrame: ID from NextKF stored" << endl;
    if (mpImuPreintegrated)
        mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
    // cout << "KeyFrame: Imu Preintegrated stored" << endl;
}

void KeyFrame::PostLoad(
    map<long unsigned int, KeyFrame *> &mpKFid,
    map<long unsigned int, MapPoint *> &mpMPid,
    map<unsigned int, GeometricCamera *> &mpCamId) {
    // Rebuild the empty variables

    // Pose
    SetPose(Tcw);

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints.clear();
    mvpMapPoints.resize(N);
    for (int i = 0; i < N; ++i) {
        if (mvBackupMapPointsId[i] != -1)
            mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
        else
            mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
    }

    // Conected KeyFrames with him weight
    mConnectedKeyFrameWeights.clear();
    for (map<long unsigned int, int>::const_iterator
             it = mBackupConnectedKeyFrameIdWeights.begin(),
             end = mBackupConnectedKeyFrameIdWeights.end();
         it != end; ++it) {
        KeyFrame *pKFi = mpKFid[it->first];
        mConnectedKeyFrameWeights[pKFi] = it->second;
    }

    // Restore parent KeyFrame
    if (mBackupParentId >= 0)
        mpParent = mpKFid[mBackupParentId];

    // KeyFrame childrens
    mspChildrens.clear();
    for (vector<long unsigned int>::const_iterator
             it = mvBackupChildrensId.begin(),
             end = mvBackupChildrensId.end();
         it != end; ++it) {
        mspChildrens.insert(mpKFid[*it]);
    }

    // Loop edge KeyFrame
    mspLoopEdges.clear();
    for (vector<long unsigned int>::const_iterator
             it = mvBackupLoopEdgesId.begin(),
             end = mvBackupLoopEdgesId.end();
         it != end; ++it) {
        mspLoopEdges.insert(mpKFid[*it]);
    }

    // Merge edge KeyFrame
    mspMergeEdges.clear();
    for (vector<long unsigned int>::const_iterator
             it = mvBackupMergeEdgesId.begin(),
             end = mvBackupMergeEdgesId.end();
         it != end; ++it) {
        mspMergeEdges.insert(mpKFid[*it]);
    }

    // Camera data
    if (mnBackupIdCamera >= 0) {
        mpCamera = mpCamId[mnBackupIdCamera];
    }
    if (mnBackupIdCamera2 >= 0) {
        mpCamera2 = mpCamId[mnBackupIdCamera2];
    }

    // Inertial data
    if (mBackupPrevKFId != -1) {
        mPrevKF = mpKFid[mBackupPrevKFId];
    }
    if (mBackupNextKFId != -1) {
        mNextKF = mpKFid[mBackupNextKFId];
    }
    mpImuPreintegrated = &mBackupImuPreintegrated;

    // Remove all backup container
    mvBackupMapPointsId.clear();
    mBackupConnectedKeyFrameIdWeights.clear();
    mvBackupChildrensId.clear();
    mvBackupLoopEdgesId.clear();

    UpdateBestCovisibles();

    // ComputeSceneMedianDepth();
}

bool KeyFrame::ProjectPointDistort(
    MapPoint *pMP, cv::Point2f &kp, float &u, float &v) {

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);

    // 3D in camera coordinates
    cv::Mat Pc = Rcw * P + tcw;
    float &PcX = Pc.at<float>(0);
    float &PcY = Pc.at<float>(1);
    float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f) {
        VLOG(5) << "ORBSLAM3: Negative depth: " << PcZ;
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f / PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    // cout << "c";

    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if (mDistCoef.total() == 5) {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool KeyFrame::ProjectPointUnDistort(
    MapPoint *pMP, cv::Point2f &kp, float &u, float &v) {

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    // 3D in camera coordinates
    cv::Mat Pc = Rcw * P + tcw;
    float &PcX = Pc.at<float>(0);
    float &PcY = Pc.at<float>(1);
    float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f) {
        VLOG(5) << "ORBSLAM3: Negative depth: " << PcZ;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    // cout << "c";

    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    kp = cv::Point2f(u, v);

    return true;
}

cv::Mat KeyFrame::GetRightPose() {
    unique_lock<mutex> lock(mMutexPose);

    cv::Mat Rrl = mTlr.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat Rlw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Mat Rrw = Rrl * Rlw;

    cv::Mat tlw = Tcw.rowRange(0, 3).col(3).clone();
    cv::Mat trl = -Rrl * mTlr.rowRange(0, 3).col(3);

    cv::Mat trw = Rrl * tlw + trl;

    cv::Mat Trw;
    cv::hconcat(Rrw, trw, Trw);

    return Trw.clone();
}

cv::Mat KeyFrame::GetRightPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat Rlw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Mat Rwr = (Rrl * Rlw).t();

    cv::Mat Rwl = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat tlr = mTlr.rowRange(0, 3).col(3);
    cv::Mat twl = GetCameraCenter();

    cv::Mat twr = Rwl * tlr + twl;

    cv::Mat Twr;
    cv::hconcat(Rwr, twr, Twr);

    return Twr.clone();
}

cv::Mat KeyFrame::GetRightPoseInverseH() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat Rlw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Mat Rwr = (Rrl * Rlw).t();

    cv::Mat Rwl = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat tlr = mTlr.rowRange(0, 3).col(3);
    cv::Mat twl = Ow.clone();

    cv::Mat twr = Rwl * tlr + twl;

    cv::Mat Twr;
    cv::hconcat(Rwr, twr, Twr);
    cv::Mat h(1, 4, CV_32F, cv::Scalar(0.0f));
    h.at<float>(3) = 1.0f;
    cv::vconcat(Twr, h, Twr);

    return Twr.clone();
}

cv::Mat KeyFrame::GetRightCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rwl = Tcw.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat tlr = mTlr.rowRange(0, 3).col(3);
    cv::Mat twl = Ow.clone();

    cv::Mat twr = Rwl * tlr + twl;

    return twr.clone();
}

cv::Mat KeyFrame::GetRightRotation() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat Rlw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
    cv::Mat Rrw = Rrl * Rlw;

    return Rrw.clone();
}

cv::Mat KeyFrame::GetRightTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    cv::Mat Rrl = mTlr.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat tlw = Tcw.rowRange(0, 3).col(3).clone();
    cv::Mat trl = -Rrl * mTlr.rowRange(0, 3).col(3);

    cv::Mat trw = Rrl * tlw + trl;

    return trw.clone();
}

void KeyFrame::SetORBVocabulary(ORBVocabulary *pORBVoc) {
    mpORBvocabulary = pORBVoc;
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase *pKFDB) {
    mpKeyFrameDB = pKFDB;
}

} // namespace ORB_SLAM3
