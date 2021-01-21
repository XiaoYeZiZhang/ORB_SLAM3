#ifndef ORB_SLAM3_FRAME_H
#define ORB_SLAM3_FRAME_H
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
namespace ObjRecognition {
// for detector and tracker
class FrameForObjRecognition {
public:
    long unsigned int m_frmIndex;
    Eigen::Matrix3d m_Rcw;
    Eigen::Vector3d m_tcw;
    cv::Mat m_img;
    std::vector<cv::KeyPoint> m_kpts;
    cv::Mat m_desp;
};

class CallbackImage {
public:
    unsigned char *data;
    int width;
    int height;
};

class CallbackFrame {
public:
    CallbackFrame() {
        width = 0;
        height = 0;
        id = 0;
    }
    long unsigned int id;
    unsigned char *data;
    int width;
    int height;
    double R[3][3];
    double t[3];
};

} // namespace ObjRecognition

#endif // ORB_SLAM3_FRAME_H
