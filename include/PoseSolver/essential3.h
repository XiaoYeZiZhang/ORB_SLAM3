#ifndef POSE_SOLVER_ESSENTIAL3_H_
#define POSE_SOLVER_ESSENTIAL3_H_

#include <Eigen/Dense>

#include <vector>

namespace PS {
bool Essential3(
    const std::vector<Eigen::Vector2d> &pts_ref,
    const std::vector<Eigen::Vector2d> &pts_cur,
    const Eigen::Matrix3d &ref_R_world,
    // Gravity direciton of current frame, expressed in world frame, assume
    // gravity direction in world coordinate is [0, 0, 1]
    const Eigen::Vector3d &W_gravity_dir,
    std::vector<Eigen::Matrix3d> *cur_R_ref,
    // NOTE The translation is scaleless, for every pair of (R, t), user should
    // use BOTH (R, t) and (R, -t) as candidate solution!
    std::vector<Eigen::Vector3d> *cur_t_ref);
}
#endif // POSE_SOLVER_ESSENTIAL3_H_
