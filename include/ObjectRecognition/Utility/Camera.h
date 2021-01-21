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
        m_fx = _fx;
        m_fy = _fy;
        m_cx = _cx;
        m_cy = _cy;
        m_width = _width;
        m_height = _height;
        m_CV_K = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
        m_CV_K.at<double>(0, 0) = m_fx;
        m_CV_K.at<double>(1, 1) = m_fy;
        m_CV_K.at<double>(0, 2) = m_cx;
        m_CV_K.at<double>(1, 2) = m_cy;
        m_CV_K.at<double>(2, 2) = 1;
        m_Eigen_K << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1;
    }

    const cv::Mat &GetCVK() {
        return m_CV_K;
    }

    const Eigen::Matrix3d &GetEigenK() {
        return m_Eigen_K;
    }

    const int &Width() const {
        return m_width;
    }

    const int &Height() const {
        return m_height;
    }

    const double &FX() const {
        return m_fx;
    }
    const double &FY() const {
        return m_fy;
    }
    const double &CX() const {
        return m_cx;
    }
    const double &CY() const {
        return m_cy;
    }

private:
    CameraIntrinsic() {
    }
    cv::Mat m_CV_K;
    Eigen::Matrix3d m_Eigen_K;
    double m_fx;
    double m_fy;
    double m_cx;
    double m_cy;
    int m_width;
    int m_height;
};

} // namespace ObjRecognition

#endif // OBJECTRECOGNITION_ORBSLAM3_CAMERA_H
