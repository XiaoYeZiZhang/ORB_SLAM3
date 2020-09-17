//
// Created by zhangye on 2020/9/16.
//
#include <glog/logging.h>
#include <Eigen/Dense>
#include "Optimizer/LBAOptimizer.h"
#include "ceres/covariance.h"
#include "ceres/crs_matrix.h"
#include "ceres/iteration_callback.h"
#include "ceres/jet.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/ordered_groups.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "Optimizer/Factor.h"
typedef long unsigned int MapPointIndex;
namespace ObjRecognition {

class IterationCallback : public ceres::IterationCallback {
public:
    explicit IterationCallback(
        bool log_to_stdout, int min_num_iterations_,
        double max_solver_time_in_seconds_)
        : log_to_stdout_(log_to_stdout),
          min_num_iterations(min_num_iterations_),
          max_solver_time_in_seconds(max_solver_time_in_seconds_) {
    }

    ~IterationCallback() {
    }

    ceres::CallbackReturnType
    operator()(const ceres::IterationSummary &summary) {

        if (log_to_stdout_) {
            char temp[256];
            snprintf(
                temp, sizeof(temp),
                "ceres report iter %d, cost %lf, "
                "cost_change %lf, trust_region_radius %lf, "
                "step_solver_time %lf s, sum time %lf s",
                summary.iteration, summary.cost, summary.cost_change,
                summary.trust_region_radius,
                summary.step_solver_time_in_seconds,
                summary.cumulative_time_in_seconds);
        }

        if (summary.iteration >= min_num_iterations &&
            summary.cumulative_time_in_seconds +
                    summary.iteration_time_in_seconds >
                max_solver_time_in_seconds) {
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    const bool log_to_stdout_;
    int min_num_iterations;
    double max_solver_time_in_seconds;
};

const int MIN_CAMERA_OPTIMIZE = 7;
LBAOptimizer::LBAOptimizer() {
    Reset();
}

void LBAOptimizer::Reset() {
    m_optimize_count = 0;
    m_fixed_camera_size = 0;
    m_optimize_camera_size = 0;
    m_max_solve_time = 0.0;
    para_Pose.clear();
    para_Pose_fix.clear();
    para_landmark_fix.clear();
    memset(covariance_xx, 1.0, 36 * sizeof(double));
}

bool LBAOptimizer::PoseCeresOptimization(
    const std::vector<cv::KeyPoint> &keyPoints,
    const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
    const std::map<int, MapPointIndex> &matches2dTo3d, const Eigen::Matrix3d &K,
    std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos) {

    bool bOptimized =
        OptimizeVOLBA(keyPoints, pointClouds, matches2dTo3d, K, Rcos, Tcos);
    return bOptimized;
}

bool LBAOptimizer::OptimizeVOLBA(
    const std::vector<cv::KeyPoint> &keyPoints,
    const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
    const std::map<int, MapPointIndex> &matches2dTo3d, const Eigen::Matrix3d &K,
    std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos) {
    if (Rcos.size() != Tcos.size()) {
        VLOG(0) << "Rs and Ts need to be optimized have no same size";
        return false;
    }

    m_optimize_camera_size = Rcos.size();
    m_fixed_camera_size = 0;
    m_max_solve_time = 0.2;
    return OptimizeVO(keyPoints, pointClouds, matches2dTo3d, K, Rcos, Tcos);
}

void LBAOptimizer::PrepareVODataForCeres(
    const std::vector<cv::KeyPoint> &keyPoints,
    const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
    const std::map<int, MapPointIndex> &matches2dTo3d, const Eigen::Matrix3d &K,
    std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos) {
    /// add optimize pose
    para_Pose.resize(m_optimize_camera_size);
    for (int i = 0; i < m_optimize_camera_size; i++) {
        Eigen::Matrix3d Rco = Rcos[i];
        Eigen::Vector3d Tco = Tcos[i];
        Eigen::Quaterniond qoc(Rco.inverse());
        Eigen::Vector3d Toc = -1 * Rco.transpose() * Tco;
        para_Pose[i][0] = Toc[0];
        para_Pose[i][1] = Toc[1];
        para_Pose[i][2] = Toc[2];

        para_Pose[i][3] = qoc.x();
        para_Pose[i][4] = qoc.y();
        para_Pose[i][5] = qoc.z();
        para_Pose[i][6] = qoc.w();
    }

    /// add landmarks
    para_landmark_fix.resize(matches2dTo3d.size());
    int index = 0;
    for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
         iter++) {
        // ???
        cv::Point3f mapPoint =
            TypeConverter::Eigen2CVPoint(pointClouds[iter->second]->GetPose());
        para_landmark_fix[index][0] = mapPoint.x;
        para_landmark_fix[index][1] = mapPoint.y;
        para_landmark_fix[index][2] = mapPoint.z;
        index++;
    }
}

void LBAOptimizer::WriteVOBackParams(
    std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos) {
    for (uint i = 0; i < m_optimize_camera_size; i++) {
        Eigen::Vector3d p;
        Eigen::Quaterniond q;
        p[0] = para_Pose[i][0];
        p[1] = para_Pose[i][1];
        p[2] = para_Pose[i][2];
        q.x() = para_Pose[i][3];
        q.y() = para_Pose[i][4];
        q.z() = para_Pose[i][5];
        q.w() = para_Pose[i][6];

        /// update params
        Eigen::Matrix3d Roc = q.toRotationMatrix();
        Eigen::Vector3d Toc = p;

        Rcos[i] = Roc.inverse();
        Tcos[i] = -1 * Roc.transpose() * Toc;
    }
}

bool LBAOptimizer::OptimizeVO(
    const std::vector<cv::KeyPoint> &keyPoints,
    const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
    const std::map<int, MapPointIndex> &matches2dTo3d, const Eigen::Matrix3d &K,
    std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos) {
    VLOG(25) << "begin optimize";
    bool result = false;
    int residual_num = 0;
    /// build problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);

    PrepareVODataForCeres(keyPoints, pointClouds, matches2dTo3d, K, Rcos, Tcos);
    VLOG(10) << "ceres optimization landmark size: " << matches2dTo3d.size();
    /// add landmarks
    for (int i = 0; i < matches2dTo3d.size(); i++) {
        problem.AddParameterBlock(para_landmark_fix[i].data(), LANDMARK_SIZE);
    }
    // add optimized camera pose
    for (int i = 0; i < m_optimize_camera_size; i++) {
        ceres::LocalParameterization *local_parameterization =
            new ObjRecognition::PoseLocalParameterizationLeftJacobians();
        problem.AddParameterBlock(
            para_Pose[i].data(), POSE_SIZE, local_parameterization);
    }

    const uint frame_num = m_optimize_camera_size;
    for (uint i_camera = 0; i_camera < frame_num; ++i_camera) {
        /// set pose ptr
        double *pose_ptr = nullptr;
        pose_ptr = para_Pose[i_camera].data();

        for (int i = 0; i < matches2dTo3d.size(); i++) {
            problem.SetParameterBlockConstant(para_landmark_fix[i].data());
        }
        int index = 0;
        for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
             iter++) {
            cv::Point2f coords = keyPoints[iter->first].pt;
            double CX = K(0, 2);
            double FX = K(0, 0);
            double FY = K(1, 1);
            double CY = K(1, 2);
            Eigen::Vector3d pt_norm;
            pt_norm(0) = (coords.x - CX) / FX;
            pt_norm(1) = (coords.y - CY) / FY;
            pt_norm(2) = 1.0;

            ProjectionFactorXYZLeftJacobians *f =
                new ProjectionFactorXYZLeftJacobians(pt_norm);
            problem.AddResidualBlock(
                f, loss_function, pose_ptr, para_landmark_fix[index].data());
            index++;
            ++residual_num;
        }
    }
#ifdef MOBILE_PLATFORM
#else
    auto err_before = CalculateLBADebugError(
        keyPoints, pointClouds, matches2dTo3d, K, Rcos, Tcos);
    VLOG(10) << "ceres optimization error before:"
             << err_before.reprojection_error;
//    STSLAMCommon::StatsCollector stats_collector_error_before_opt(
//        "detector result optimizetion projection error before");
//    stats_collector_error_before_opt.AddSample(err_before.reprojection_error);
#endif

    ceres::Solver::Options options;
#ifdef DESKTOP_PLATFORM
    options.linear_solver_type = ceres::SPARSE_SCHUR;
#else
    options.linear_solver_type = ceres::DENSE_SCHUR;
#endif
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;

    options.initial_trust_region_radius = 1e5;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    options.use_explicit_schur_complement = true;
    IterationCallback ceres_log(false, 1, m_max_solve_time);
    options.callbacks.push_back(&ceres_log);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result = summary.termination_type != ceres::TerminationType::FAILURE &&
             summary.IsSolutionUsable();
    VLOG(10) << "[LBA test] ceres final termination_type "
             << summary.termination_type;
    VLOG(10) << "[LBA test] ceres final IsSolutionUsable "
             << summary.IsSolutionUsable();
    VLOG(10) << "[LBA test] ceres final iterations "
             << summary.iterations.size();
    VLOG(10) << "[LBA test] ceres summary.message " << summary.message.c_str();
    VLOG(10) << "[LBA test] ceres init cost " << summary.initial_cost
             << "final cost " << summary.final_cost;
    /// write back optimize data
    WriteVOBackParams(Rcos, Tcos);

#ifdef MOBILE_PLATFORM
#else
    auto err_after = CalculateLBADebugError(
        keyPoints, pointClouds, matches2dTo3d, K, Rcos, Tcos);
    VLOG(10) << "ceres optimization error after:"
             << err_after.reprojection_error;
//    STSLAMCommon::StatsCollector stats_collector_error_after_opt(
//        "detector result optimizetion projection error after");
//    stats_collector_error_after_opt.AddSample(err_after.reprojection_error);
#endif

    VLOG(10) << "###LBA CERES SOLVER\n"
             << " fixed num: " << m_fixed_camera_size
             << " state number: " << m_optimize_camera_size
             << " landmark number: " << matches2dTo3d.size()
             << " residual number: " << residual_num
             << " iterations: " << summary.iterations.size()
             << " initial cost: " << summary.initial_cost
             << " final cost: " << summary.final_cost
             << " successful: " << (result ? "YES" : "NO");
    ++m_optimize_count;
    return result;
}

LBAOptimizer::LBADebugError LBAOptimizer::CalculateLBADebugError(
    const std::vector<cv::KeyPoint> &keyPoints,
    const std::vector<ObjRecognition::MapPoint::Ptr> &pointClouds,
    const std::map<int, MapPointIndex> &matches2dTo3d, const Eigen::Matrix3d &K,
    std::vector<Eigen::Matrix3d> &Rcos, std::vector<Eigen::Vector3d> &Tcos) {
    LBAOptimizer::LBADebugError ret_error;
#if MOBILE_PLATFORM
    return ret_error;
#endif
    int num_pts_all = 0, num_pts_curr = 0;
    double sum_rpe_all = 0, sum_rpe_curr = 0;

    for (int i_cam = 0; i_cam < m_optimize_camera_size; ++i_cam) {
        for (auto iter = matches2dTo3d.begin(); iter != matches2dTo3d.end();
             iter++) {
            cv::Point2f coords = keyPoints[iter->first].pt;
            cv::Point3f pointCloudCoords = TypeConverter::Eigen2CVPoint(
                pointClouds[iter->second]->GetPose());
            Eigen::Vector3d pt_o = Eigen::Vector3d(
                pointCloudCoords.x, pointCloudCoords.y, pointCloudCoords.z);
            Eigen::Matrix3d Rco = Rcos[i_cam];
            Eigen::Vector3d Tco = Tcos[i_cam];
            Eigen::Vector3d pt_c = K * (Rco * pt_o + Tco);
            Eigen::Vector2d pt_proj{pt_c.x() / pt_c.z(), pt_c.y() / pt_c.z()};
            Eigen::Vector2d pt_norm_imgCoords = {coords.x, coords.y};
            double rpe = (pt_norm_imgCoords - pt_proj).norm();
            if (i_cam == Rcos.size() - 1) {
                sum_rpe_curr += rpe;
                num_pts_curr++;
            }
            sum_rpe_all += rpe;
            num_pts_all++;
        }
    }
    ret_error.reprojection_error = sum_rpe_all / num_pts_all;
    ret_error.reprojection_error_latest_frm = sum_rpe_curr / num_pts_curr;
    return ret_error;
}
} // namespace ObjRecognition