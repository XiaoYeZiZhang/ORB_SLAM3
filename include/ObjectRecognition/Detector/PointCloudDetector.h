//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_POINTCLOUDDETECTOR_H
#define ORB_SLAM3_POINTCLOUDDETECTOR_H

#include "Struct/PointCloudObject.h"
#include "Utility/RecognitionBase.h"
#include "Struct/Object.h"
#include "DetectorFrame.h"
#include "PoseSolver/PoseSolver.h"
namespace ObjRecognition {

typedef long unsigned int KeyFrameIndex;

class PointCloudObjDetector : public RecognitionBase {
public:
    PointCloudObjDetector();
    ~PointCloudObjDetector();

    void SetPointCloudObj(const std::shared_ptr<Object> &pObj);
    void SetVoc(const std::shared_ptr<DBoW3::Vocabulary> &pVoc);
    void Process(const std::shared_ptr<FrameData> &frm);
    void Reset();
    void Clear();
    bool Load(const int &mem_size, const char *mem);
    bool Save(int &mem_size, char **mem);
    void SetInfo();
    int GetInfo(std::string &info);

private:
    void PreProcess(const std::shared_ptr<ObjRecognition::FrameData> &frm);

    std::vector<PS::MatchSet2D>
    Find2DMatches(const std::vector<KeyFrame::Ptr> &allKFs);
    PS::MatchSet3D Find3DMatch();

    void PoseOptimize(const std::vector<int> &inliers_3d);
    bool PoseSolver(
        const PS::MatchSet3D &matches_3d,
        const std::vector<PS::MatchSet2D> &matches_2d,
        std::vector<int> &inliers_3d,
        std::vector<std::vector<int>> &inliers_2d);

    void PnPResultHandle();

    void ResultRecord();

    void ShowDetectResult();

private:
    std::shared_ptr<Object> mObj;
    std::shared_ptr<DetectorFrame> m_frame_cur;
    Eigen::Matrix3d Rcw_cur_;
    Eigen::Vector3d tcw_cur_;
    Eigen::Matrix3d Rwo_cur_;

    Eigen::Vector3d two_cur_;
    Eigen::Matrix3d Rco_cur_;
    Eigen::Vector3d tco_cur_;

    int knn_match_num_;
    int pnp_inliers_num_;
    int pnp_inliers_3d_num_;
    int pnp_inliers_2d_num_;
    bool pnp_solver_result_;
    ObjRecogState detect_state_;

    std::shared_ptr<DBoW3::Vocabulary> voc_;

    std::string info_;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_POINTCLOUDDETECTOR_H
