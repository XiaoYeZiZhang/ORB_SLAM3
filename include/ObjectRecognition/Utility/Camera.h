//
// Created by zhangye on 2020/9/15.
//

#ifndef OBJECTRECOGNITION_ORBSLAM3_CAMERA_H
#define OBJECTRECOGNITION_ORBSLAM3_CAMERA_H
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
namespace ObjRecognition {

class CameraIntrinsic {
public:
    ~CameraIntrinsic() {
    }
    static CameraIntrinsic &GetInstance() {
        static CameraIntrinsic instance;
        return instance;
    }

    void SetParameters(
        const double &_fx, const double &_fy, const double &_cx,
        const double &_cy, const int &_width, const double &_height) {
        mFX = _fx;
        mFY = _fy;
        mCX = _cx;
        mCY = _cy;
        mWidth = _width;
        mHeight = _height;
        mCVK = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
        mCVK.at<double>(0, 0) = mFX;
        mCVK.at<double>(1, 1) = mFY;
        mCVK.at<double>(0, 2) = mCX;
        mCVK.at<double>(1, 2) = mCY;
        mCVK.at<double>(2, 2) = 1;
        mEigenK << mFX, 0, mCX, 0, mFY, mCY, 0, 0, 1;
    }

    const cv::Mat &GetCVK() {
        return mCVK;
    }

    const Eigen::Matrix3d &GetEigenK() {
        return mEigenK;
    }

    const int &Width() const {
        return mWidth;
    }

    const int &Height() const {
        return mHeight;
    }

    const double &FX() const {
        return mFX;
    }
    const double &FY() const {
        return mFY;
    }
    const double &CX() const {
        return mCX;
    }
    const double &CY() const {
        return mCY;
    }

private:
    CameraIntrinsic() {
    }
    cv::Mat mCVK;
    Eigen::Matrix3d mEigenK;
    double mFX;
    double mFY;
    double mCX;
    double mCY;
    int mWidth;
    int mHeight;
};

} // namespace ObjRecognition

#endif // OBJECTRECOGNITION_ORBSLAM3_CAMERA_H
