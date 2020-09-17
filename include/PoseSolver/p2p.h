#ifndef POSE_SOLVER_P2P_H_
#define POSE_SOLVER_P2P_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
namespace PS {

bool P2P(
    const Eigen::Vector3d &obj_pt_1, const Eigen::Vector2d &img_pt_1,
    const Eigen::Vector3d &obj_pt_2, const Eigen::Vector2d &img_pt_2,
    // Assume gravity direction in world(obj) coordinate is [0, 0, 1]
    const Eigen::Vector3d &gravity_dir, std::vector<Eigen::Matrix3d> *C_R_W,
    std::vector<Eigen::Vector3d> *C_t_W);
}
#endif // POSE_SOLVER_P2P_H_
