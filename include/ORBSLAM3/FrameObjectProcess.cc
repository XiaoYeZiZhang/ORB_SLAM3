#include <Eigen/Core>
#include <cxeigen.hpp>
#include "Utility/Camera.h"
#include "FrameObjectProcess.h"
#include "glog/logging.h"
#include "include/Tools.h"
#include "mode.h"
#include "Visualizer/GlobalImageViewer.h"

namespace ORB_SLAM3 {

FrameObjectProcess::FrameObjectProcess() {
    // TODO(zhangye): check the parameters
    // 1000, 1.2, 8, 20
    m_orb_detector = cv::ORB::create(
        1000, Parameters::GetInstance().KORBExtractor_scaleFactor,
        Parameters::GetInstance().KORBExtractor_nlevels);
    m_orb_detector->setScoreType(cv::ORB::FAST_SCORE);
    m_orb_detector->setFastThreshold(
        Parameters::GetInstance().KORBExtractor_fastInitThreshold);
    m_obj_corner_points = std::vector<Eigen::Vector3d>();
}

void FrameObjectProcess::ProcessFrame(ORB_SLAM3::KeyFrame *&pKF) {

    if (!pKF) {
        LOG(FATAL) << "keyframe to process is null";
    }

    if (m_obj_corner_points.empty()) {
        return;
    }

    cv::Mat img = pKF->imgLeft.clone();
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;

    cv::Mat Tcw = pKF->GetPose();
    Eigen::Matrix4d Tcw_eigen;
    cv::cv2eigen(Tcw, Tcw_eigen);
    tcw = Tcw_eigen.block<3, 1>(0, 3);
    Rcw = Tcw_eigen.block<3, 3>(0, 0);

    Eigen::Matrix3d camera_intrinsic = Eigen::Matrix3d::Identity();
    camera_intrinsic(0, 0) =
        ObjRecognition::CameraIntrinsic::GetInstance().FX();
    camera_intrinsic(1, 1) =
        ObjRecognition::CameraIntrinsic::GetInstance().FY();
    camera_intrinsic(0, 2) =
        ObjRecognition::CameraIntrinsic::GetInstance().CX();
    camera_intrinsic(1, 2) =
        ObjRecognition::CameraIntrinsic::GetInstance().CY();

    cv::Mat mask;
    Tools::GetBoundingBoxMask(
        img, camera_intrinsic, Rcw, tcw, m_obj_corner_points, mask);

    std::vector<cv::KeyPoint> keypoints_new;
    cv::Mat descriptor_new;
    m_orb_detector->detectAndCompute(img, mask, keypoints_new, descriptor_new);

    VLOG(5) << "[STObject] KF id:" << pKF->mnId
            << ", object detect orb feature "
            << static_cast<int>(keypoints_new.size());

    std::string featureInfo = std::string("2d feature num: ");
    featureInfo += std::to_string(keypoints_new.size()) + " ";
    cv::Mat show = img.clone();

    cv::cvtColor(show, show, CV_GRAY2BGR);
    cv::drawKeypoints(show, keypoints_new, show);

    cv::Mat img_txt;
    std::stringstream s;
    s << featureInfo;
    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);
    img_txt = cv::Mat(show.rows + textSize.height + 10, show.cols, show.type());
    show.copyTo(img_txt.rowRange(0, show.rows).colRange(0, show.cols));
    img_txt.rowRange(show.rows, img_txt.rows) =
        cv::Mat::zeros(textSize.height + 10, show.cols, show.type());
    cv::putText(
        img_txt, s.str(), cv::Point(5, img_txt.rows - 5),
        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);

    ObjRecognition::GlobalOcvViewer::UpdateView(
        "New Keypoints to Extract:", img_txt);

    // TODO(zhangye): keys and keysun???
    std::vector<cv::KeyPoint> keypoints_old = pKF->mvKeys;
    cv::Mat descriptor_old = pKF->mDescriptors.clone();

    keypoints_old.insert(
        keypoints_old.end(), keypoints_new.begin(), keypoints_new.end());
    for (size_t i = 0; i < descriptor_new.rows; i++) {
        descriptor_old.push_back(descriptor_new.row(i));
    }

    pKF->SetKeyPoints(keypoints_old);
    pKF->SetDesps(descriptor_old);
} // FrameObjectProcess::ProcessFrame

void FrameObjectProcess::AddObjectModel(const int obj_id) {
    VLOG(5) << "[STObject] add the object id: " << obj_id;
} // FrameObjectProcess::AddObjectModel()

void FrameObjectProcess::SetBoundingBox(
    const std::vector<Eigen::Vector3d> &boundingbox) {
    m_obj_corner_points.clear();
    m_obj_corner_points = boundingbox;
}
void FrameObjectProcess::Reset() {
    m_obj_corner_points.clear();
}

} // namespace ORB_SLAM3