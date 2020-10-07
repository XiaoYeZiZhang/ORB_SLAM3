#include <Eigen/Core>
#include <cxeigen.hpp>
#include "Utility/Camera.h"
#include "FrameObjectProcess.h"
#include "glog/logging.h"
#include "QuickHull.h"
#include "Visualizer/GlobalImageViewer.h"

namespace ORB_SLAM3 {

FrameObjectProcess::FrameObjectProcess() {
    // TODO(zhangye): check the parameters
    m_orb_detector = cv::ORB::create(
        1000, Parameters::GetInstance().KORBExtractor_scaleFactor,
        Parameters::GetInstance().KORBExtractor_nlevels);
    m_orb_detector->setScoreType(cv::ORB::FAST_SCORE);
    m_orb_detector->setFastThreshold(
        Parameters::GetInstance().KORBExtractor_fastThreathold);
    m_obj_corner_points = std::vector<Eigen::Vector3d>();
}

static bool InBorder(
    const Eigen::Vector2d &pt, const int &xMin, const int &yMin,
    const int &xMax, const int &yMax) {
    int imgX = static_cast<int>(pt(0));
    int imgY = static_cast<int>(pt(1));
    return xMin <= imgX && imgX < xMax && yMin <= imgY && imgY < yMax;
}

static int PointInImageBorder(std::vector<cv::Point> box_proj_result) {
    int point_in_image_count = 0;

    for (const auto &it : box_proj_result) {
        if (InBorder(
                Eigen::Vector2d(it.x, it.y), 0, 0,
                ObjRecognition::CameraIntrinsic::GetInstance().Width(),
                ObjRecognition::CameraIntrinsic::GetInstance().Height())) {
            point_in_image_count++;
        }
    }
    return point_in_image_count;
}

static void GetBoundingBoxMask(
    const cv::Mat &img, const Eigen::Matrix3d &K,
    const Eigen::Matrix3d &resultObjR, const Eigen::Vector3d &resultObjT,
    const std::vector<Eigen::Vector3d> &mapPointBoundingBox, cv::Mat &mask) {

    if (mapPointBoundingBox.size() < 8) {
        return;
    }

    mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    // mask.setTo(255, mask == 0);
    // return;
    std::vector<cv::Point> boxProjResult;
    for (int i = 0; i < mapPointBoundingBox.size(); i++) {
        Eigen::Vector3d p =
            K * (resultObjR * mapPointBoundingBox[i] + resultObjT);
        cv::Point pResult;
        pResult.x = p(0) / p(2);
        pResult.y = p(1) / p(2);
        boxProjResult.emplace_back(pResult);
    }

    QuickHull::Polygon polygon_input, polygon_result;
    for (auto const &it : boxProjResult) {
        QuickHull::Point point(0, 0);
        point.x = it.x;
        point.y = it.y;
        polygon_input.push_back(point);
    }

    polygon_result = QuickHull::quickHull(polygon_input);

    if (polygon_result.size() < 3) {
        VLOG(5) << "[STObject] QuickHull result size is too small";
        return;
    }

    if (PointInImageBorder(boxProjResult) < 1) {
        VLOG(5) << "[STObject] QuickHull corner point projected point"
                   " none in the image ";
        return;
    }

    std::vector<cv::Point> boxProjResultShow;
    std::swap(boxProjResultShow, boxProjResult);
    VLOG(5) << "[STObject] QuickHull result size "
            << static_cast<int>(polygon_result.size());
    for (auto const &it : polygon_result) {
        cv::Point point;
        point.x = it.x;
        point.y = it.y;
        boxProjResult.push_back(point);
    }

    std::vector<std::vector<cv::Point>> pts;
    pts.push_back(boxProjResult);
    cv::fillPoly(mask, pts, cv::Scalar(255));

    // show:
    //    cv::Mat maskShow = mask.clone();
    //    cv::cvtColor(maskShow, maskShow, cv::COLOR_GRAY2BGR);
    //    for (const auto &it : boxProjResultShow) {
    //        cv::drawMarker(maskShow, it, cv::Scalar(0, 0, 255));
    //    }
    // GlobalOcvViewer::UpdateView("2D bounding box mask", maskShow);

} // void GetBoundingBoxMask()

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
    GetBoundingBoxMask(
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

    VLOG(0) << "here!!!!!!!!!!!!!!!!!!!";
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
    // m_obj_corner_points = GetBoundingBox();
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