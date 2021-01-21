#ifndef ORB_SLAM3_FRAMEBASE_H
#define ORB_SLAM3_FRAMEBASE_H
#include <Eigen/Core>
#include <mutex>
#include <opencv2/core/mat.hpp>
namespace ObjRecognition {
class FrameBase {
public:
    int m_frame_index = 0;
    cv::Mat m_raw_image;
    cv::Mat m_desp;
    std::vector<cv::KeyPoint> m_kpts;

private:
    Eigen::Matrix3d m_Rcw;
    Eigen::Vector3d m_tcw;
    Eigen::Matrix3d m_Rco;
    Eigen::Vector3d m_tco;
    Eigen::Matrix3d m_Rwo;
    Eigen::Vector3d m_two;
    std::mutex m_mutex_pose;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_FRAMEBASE_H
