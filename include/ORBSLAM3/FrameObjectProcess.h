#ifndef BACKEND_FRAMEOBJECTPROCESS_H
#define BACKEND_FRAMEOBJECTPROCESS_H
#include <memory>
#include <opencv2/opencv.hpp>
#include "Utility/Parameters.h"
#include "KeyFrame.h"
#include "Atlas.h"
namespace ORB_SLAM3 {

class FrameObjectProcess {
public:
    void ProcessFrame(ORB_SLAM3::KeyFrame *&pKF);
    void SetBoundingBox(const std::vector<Eigen::Vector3d> &boundingbox);
    void Reset();
    static FrameObjectProcess *GetInstance() {
        static FrameObjectProcess m_Instance;
        return &m_Instance;
    }

private:
    FrameObjectProcess();
    ~FrameObjectProcess() = default;
    FrameObjectProcess(const FrameObjectProcess &);
    FrameObjectProcess &operator=(const FrameObjectProcess &);
    std::vector<Eigen::Vector3d> m_obj_corner_points;
    cv::Ptr<cv::ORB> m_orb_detector;
};

} // namespace ORB_SLAM3

#endif // BACKEND_FRAMEOBJECTPROCESS_H
