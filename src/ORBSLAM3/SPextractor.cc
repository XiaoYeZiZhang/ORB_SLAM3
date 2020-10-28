//
// Created by root on 2020/10/21.
//
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "ORBSLAM3/SPextractor.h"
#include "ORBSLAM3/SuperPoint.h"
#include <glog/logging.h>
#include <torch/script.h>
#include <chrono>
using namespace cv;
using namespace std;
using namespace std::chrono;

namespace ORB_SLAM3 {

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

const float factorPI = (float)(CV_PI / 180.f);

void ExtractorNode_sp::DivideNode(
    ExtractorNode_sp &n1, ExtractorNode_sp &n2, ExtractorNode_sp &n3,
    ExtractorNode_sp &n4) {
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    // Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++) {
        const cv::KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x) {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        } else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}

SPextractor::SPextractor(
    int _nfeatures, float _scaleFactor, int _nlevels, float _iniThFAST,
    float _minThFAST, bool _is_use_cuda)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
      iniThFAST(_iniThFAST), minThFAST(_minThFAST), is_use_cuda(_is_use_cuda) {
    model = make_shared<SuperPoint>();
    torch::Device device(torch::kCUDA);
    model->to(device);

    traced_module = torch::jit::load(
        "/home/zhangye/data1/traced_superpoint_model_test_save_time.pt");
    //    traced_module =
    //        torch::jit::load("/home/zhangye/data1/traced_superpoint_model.pt");
    traced_module.to(at::kCUDA);

    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++) {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++) {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale =
        nfeatures * (1 - factor) /
        (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++) {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
}

vector<cv::KeyPoint> SPextractor::DistributeOctTree(
    const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
    const int &maxX, const int &minY, const int &maxY, const int &N,
    const int &level) {
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode_sp> lNodes;

    vector<ExtractorNode_sp *> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++) {
        ExtractorNode_sp ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode_sp>::iterator lit = lNodes.begin();

    while (lit != lNodes.end()) {
        if (lit->vKeys.size() == 1) {
            lit->bNoMore = true;
            lit++;
        } else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode_sp *>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish) {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end()) {
            if (lit->bNoMore) {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            } else {
                // If more than one point, subdivide
                ExtractorNode_sp n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0) {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(
                            make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0) {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(
                            make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0) {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(
                            make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0) {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(
                            make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize) {
            bFinish = true;
        } else if (((int)lNodes.size() + nToExpand * 3) > N) {

            while (!bFinish) {

                prevSize = lNodes.size();

                vector<pair<int, ExtractorNode_sp *>>
                    vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(
                    vPrevSizeAndPointerToNode.begin(),
                    vPrevSizeAndPointerToNode.end());
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0;
                     j--) {
                    ExtractorNode_sp n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(
                        n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0) {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(
                                make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0) {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(
                                make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0) {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(
                                make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0) {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(
                                make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for (list<ExtractorNode_sp>::iterator lit = lNodes.begin();
         lit != lNodes.end(); lit++) {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint *pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void KeyPointsFilterByPixelsMask(
    std::vector<cv::KeyPoint> &inOutKeyPoints, const cv::Mat &mask,
    int minBorderX, int minBorderY, float scaleFactor) {
    if (mask.empty() || inOutKeyPoints.empty()) {
        return;
    }

    if (scaleFactor <= 0.0f) {
        LOG(ERROR) << __FUNCTION__ << ": invlid scaleFactor = " << scaleFactor;
        return;
    }

    const float invScaleFactor = 1.0f / scaleFactor;

    for (cv::KeyPoint &cur : inOutKeyPoints) {
        cur.pt.x += minBorderX;
        cur.pt.y += minBorderY;
        cur.pt *= scaleFactor;
    }

    KeyPointsFilter::runByPixelsMask(inOutKeyPoints, mask);

    for (cv::KeyPoint &cur : inOutKeyPoints) {
        cur.pt *= invScaleFactor;
        cur.pt.x -= minBorderX;
        cur.pt.y -= minBorderY;
    }
}

void SPextractor::ComputeKeyPointsWithMask(
    vector<vector<KeyPoint>> &allKeypoints, cv::Mat &_desc,
    const cv::Mat &mask) {
    allKeypoints.resize(nlevels);

    vector<cv::Mat> vDesc;

    const float W = 30;

    for (int level = 0; level < nlevels; ++level) {
        SPDetector detector(model, traced_module);
        detector.detect(mvImagePyramid[level], is_use_cuda);

        const int minBorderX = 0; // EDGE_THRESHOLD - 3;
        const int minBorderY = 0; // minBorderX;
        const int maxBorderX =
            mvImagePyramid[level].cols - 0; // EDGE_THRESHOLD + 3;
        const int maxBorderY =
            mvImagePyramid[level].rows - 0; // EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        auto start = high_resolution_clock::now();
        vector<cv::KeyPoint> vKeysCell;

        detector.getKeyPoints(
            iniThFAST, minBorderX, maxBorderX, minBorderY, maxBorderY,
            vKeysCell, true);

        if (vKeysCell.empty()) {
            detector.getKeyPoints(
                minThFAST, minBorderX, maxBorderX, minBorderY, maxBorderY,
                vKeysCell, true);
        }

        if (!vKeysCell.empty()) {
            for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
                 vit != vKeysCell.end(); vit++) {
                vToDistributeKeys.push_back(*vit);
            }
        }

        VLOG(5) << "Time taken by filter keypoints: "
                << (duration_cast<microseconds>(
                        high_resolution_clock::now() - start))
                           .count() /
                       1000.0
                << " ms" << std::endl;

        start = high_resolution_clock::now();
        vector<KeyPoint> &keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        KeyPointsFilterByPixelsMask(
            vToDistributeKeys, mask, minBorderX, minBorderY,
            mvScaleFactor[level]);
        keypoints = vToDistributeKeys;
        KeyPointsFilter::retainBest(keypoints, mnFeaturesPerLevel[level]);

        VLOG(5) << "time for distribute oct tree: "
                << (duration_cast<microseconds>(
                        high_resolution_clock::now() - start))
                           .count() /
                       1000.0
                << " ms" << std::endl;

        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++) {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }

        start = high_resolution_clock::now();
        cv::Mat desc;
        detector.computeDescriptors(keypoints, desc, is_use_cuda);
        vDesc.push_back(desc);

        VLOG(5) << "time for compute descriptors"
                << (duration_cast<microseconds>(
                        high_resolution_clock::now() - start))
                           .count() /
                       1000.0
                << " ms" << std::endl;
    }

    cv::vconcat(vDesc, _desc);
}

void SPextractor::ComputeKeyPointsOctTree(
    vector<vector<KeyPoint>> &allKeypoints, cv::Mat &_desc) {
    allKeypoints.resize(nlevels);

    vector<cv::Mat> vDesc;

    const float W = 30;

    for (int level = 0; level < nlevels; ++level) {
        SPDetector detector(model, traced_module);
        detector.detect(mvImagePyramid[level], is_use_cuda);

        const int minBorderX = 0; // EDGE_THRESHOLD - 3;
        const int minBorderY = 0; // minBorderX;
        const int maxBorderX =
            mvImagePyramid[level].cols - 0; // EDGE_THRESHOLD + 3;
        const int maxBorderY =
            mvImagePyramid[level].rows - 0; // EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        //        auto start = high_resolution_clock::now();
        //        for(int i=0; i<nRows; i++)
        //        {
        //            const float iniY =minBorderY+i*hCell;
        //            float maxY = iniY+hCell+6;
        //
        //            if(iniY>=maxBorderY-3)
        //                continue;
        //            if(maxY>maxBorderY)
        //                maxY = maxBorderY;
        //
        //            for(int j=0; j<nCols; j++)
        //            {
        //                const float iniX =minBorderX+j*wCell;
        //                float maxX = iniX+wCell+6;
        //                if(iniX>=maxBorderX-6)
        //                    continue;
        //                if(maxX>maxBorderX)
        //                    maxX = maxBorderX;
        //
        //                auto start1 = high_resolution_clock::now();
        //                vector<cv::KeyPoint> vKeysCell;
        //
        //                detector.getKeyPoints(iniThFAST, iniX, maxX, iniY,
        //                maxY, vKeysCell, true);
        //
        //                if(vKeysCell.empty())
        //                {
        //                    detector.getKeyPoints(minThFAST, iniX, maxX, iniY,
        //                    maxY, vKeysCell, true);
        //                }
        //
        //                if(!vKeysCell.empty())
        //                {
        //                    for(vector<cv::KeyPoint>::iterator
        //                    vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
        //                    {
        //                        (*vit).pt.x+=j*wCell;
        //                        (*vit).pt.y+=i*hCell;
        //                        vToDistributeKeys.push_back(*vit);
        //                    }
        //                }
        //                auto stop1 = high_resolution_clock::now();
        //                auto duration1 = duration_cast<microseconds>(stop1 -
        //                start1); std::cout << "Time taken by get keypoints
        //                every cell: "
        //                          << duration1.count()/1000.0 << " ms" <<
        //                          std::endl;
        //
        //            }
        //        }
        //        auto stop = high_resolution_clock::now();
        //        auto duration = duration_cast<microseconds>(stop - start);
        //        std::cout << "Time taken by get keypoints: "
        //                  << duration.count()/1000.0 << " ms" << std::endl;
        auto start = high_resolution_clock::now();
        vector<cv::KeyPoint> vKeysCell;

        detector.getKeyPoints(
            iniThFAST, minBorderX, maxBorderX, minBorderY, maxBorderY,
            vKeysCell, true);

        if (vKeysCell.empty()) {
            detector.getKeyPoints(
                minThFAST, minBorderX, maxBorderX, minBorderY, maxBorderY,
                vKeysCell, true);
        }

        if (!vKeysCell.empty()) {
            for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
                 vit != vKeysCell.end(); vit++) {
                vToDistributeKeys.push_back(*vit);
            }
        }

        VLOG(5) << "Time taken by filter keypoints: "
                << (duration_cast<microseconds>(
                        high_resolution_clock::now() - start))
                           .count() /
                       1000.0
                << " ms" << std::endl;

        start = high_resolution_clock::now();
        vector<KeyPoint> &keypoints = allKeypoints[level];
        // keypoints.reserve(nfeatures);
        //        keypoints = vToDistributeKeys;
        keypoints = DistributeOctTree(
            vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY,
            mnFeaturesPerLevel[level], level);

        VLOG(5) << "time for distribute oct tree: "
                << (duration_cast<microseconds>(
                        high_resolution_clock::now() - start))
                           .count() /
                       1000.0
                << " ms" << std::endl;

        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++) {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }

        start = high_resolution_clock::now();
        cv::Mat desc;
        detector.computeDescriptors(keypoints, desc, is_use_cuda);
        vDesc.push_back(desc);

        VLOG(5) << "time for compute descriptors"
                << (duration_cast<microseconds>(
                        high_resolution_clock::now() - start))
                           .count() /
                       1000.0
                << " ms" << std::endl;
    }

    cv::vconcat(vDesc, _desc);
}

void SPextractor::operator()(
    InputArray _image, const cv::Mat &mask, vector<KeyPoint> &_keypoints,
    OutputArray _descriptors) {
    if (_image.empty()) {
        return;
    }

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    Mat descriptors;

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    vector<vector<KeyPoint>> allKeypoints;
    cv::Mat mask_zero = mask.clone();
    mask_zero = cv::Scalar::all(0);
    cv::Mat dst;
    cv::bitwise_xor(mask_zero, mask, dst);
    // use mask if not all zero
    if (cv::countNonZero(dst) > 0) {
        VLOG(5) << "compute with mask";
        ComputeKeyPointsWithMask(allKeypoints, descriptors, mask);
    } else {
        VLOG(5) << "compute without mask";
        ComputeKeyPointsOctTree(allKeypoints, descriptors);
    }

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if (nkeypoints == 0)
        _descriptors.release();
    else {
        _descriptors.create(nkeypoints, 256, CV_32F);
        descriptors.copyTo(_descriptors.getMat());
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level) {
        vector<KeyPoint> &keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // // preprocess the resized image
        // Mat workingMat = mvImagePyramid[level].clone();
        // GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2,
        // BORDER_REFLECT_101);

        // // Compute the descriptors
        // Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        // computeDescriptors(workingMat, keypoints, desc, pattern);

        // offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0) {
            float scale = mvScaleFactor[level]; // getScale(level, firstLevel,
                                                // scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                            keypointEnd = keypoints.end();
                 keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void SPextractor::ComputePyramid(cv::Mat image) {
    for (int level = 0; level < nlevels; ++level) {
        float scale = mvInvScaleFactor[level];
        Size sz(
            cvRound((float)image.cols * scale),
            cvRound((float)image.rows * scale));
        Size wholeSize(
            sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] =
            temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if (level != 0) {
            resize(
                mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0,
                INTER_LINEAR);

            copyMakeBorder(
                mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                EDGE_THRESHOLD, EDGE_THRESHOLD,
                BORDER_REFLECT_101 + BORDER_ISOLATED);
        } else {
            copyMakeBorder(
                image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                EDGE_THRESHOLD, BORDER_REFLECT_101);
        }
    }
}
} // namespace ORB_SLAM3
