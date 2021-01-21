#ifndef ORB_SLAM3_DETECTORPOINTCLOUD_H
#define ORB_SLAM3_DETECTORPOINTCLOUD_H

#include "PointCloudObject.h"
#include "RecognitionBase.h"
#include "Object.h"
#include "DetectorFrame.h"
#include "PoseSolver.h"
namespace ObjRecognition {
class PointCloudObjDetector : public RecognitionBase {
public:
    PointCloudObjDetector();
    ~PointCloudObjDetector();

    void SetPointCloudObj(const std::shared_ptr<Object> &pObj);
    void SetVoc(const std::shared_ptr<DBoW3::Vocabulary> &pVoc);
    void Process(const std::shared_ptr<FrameForObjRecognition> &frm);
    void Reset();
    void Clear();
    bool Load(const long long &mem_size, const char *mem);
    bool Save(long long &mem_size, char **mem);

private:
    void PreProcess(
        const std::shared_ptr<ObjRecognition::FrameForObjRecognition> &frm);
    std::vector<PS::MatchSet2D> Find2DMatches(
        const std::vector<KeyFrame::Ptr> &allKFs,
        std::vector<KeyFrame::Ptr> &kf_mathceds);
    PS::MatchSet3D Find3DMatch();
    PS::MatchSet3D
    Find3DMatchByConnection(const std::vector<KeyFrame::Ptr> &kf_mathceds);

    bool PoseSolver(
        const PS::MatchSet3D &matches_3d,
        const std::vector<PS::MatchSet2D> &matches_2d,
        std::vector<int> &inliers_3d,
        std::vector<std::vector<int>> &inliers_2d);
    void PnPResultHandle();
    void ResultRecord();
    void ShowDetectResult();
    void DrawTextInfo(const cv::Mat &img, cv::Mat &img_txt);
    float ComputeAverageReProjError(const std::vector<int> &inliers_3d);

private:
    std::shared_ptr<Object> m_obj;
    std::shared_ptr<DetectorFrame> m_frame_cur;
    Eigen::Matrix3d m_Rcw_cur;
    Eigen::Vector3d m_tcw_cur;
    Eigen::Matrix3d m_Rwo_cur;
    Eigen::Vector3d m_two_cur;

    Eigen::Matrix3d m_Rco_cur;
    Eigen::Vector3d m_tco_cur_scale;
    Eigen::Vector3d m_tco_cur;

    int m_knn_match_num;
    int m_pnp_inliers_num;
    int m_pnp_inliers_3d_num;
    int m_pnp_inliers_2d_num;
    bool m_pnp_solver_result;
    ObjRecogState m_detect_state;

    std::shared_ptr<DBoW3::Vocabulary> m_voc;

    float m_reproj_error;

    bool m_has_good_result;

    Eigen::Matrix3d m_last_detectionGood_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d m_last_detectionGood_t = Eigen::Vector3d::Zero();

    Eigen::Matrix3d m_last_slam_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d m_last_slam_t = Eigen::Vector3d::Zero();

    std::vector<MapPoint::Ptr> m_associated_mappoints_vector;
    double m_scale;
    size_t m_scale_num;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_DETECTORPOINTCLOUD_H
