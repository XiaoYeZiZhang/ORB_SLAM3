#include "PoseSolver.h"

#include <csetjmp>

#include <glog/logging.h>

namespace PS {

namespace internal {

std::jmp_buf jmp_buf;
void JmpOnFatal() {
    LOG(ERROR) << "Fatal error captured.";
    std::longjmp(jmp_buf, 1);
}

// Fix R, only solve the translation(DLT), leave C_T_W unchanged if failed.
bool SolveTranslation(const std::vector<MatchSet2D> &matches_2d, Pose *C_T_W) {
    DCHECK(C_T_W != nullptr);
    // For every pair of match (ref, cur), equaling the 3D point in world frame
    // W_R_ref * (d_ref * ref) + W_t_ref = W_R_cur * (d_cur * cur) + W_t_cur
    // => [W_R_ref * ref; -W_R_cur * cur] * [d_ref, d_cur] = W_t_cur - W_t_ref
    //    let n = (W_R_ref * ref).cross(W_R_cur * cur) * any-scale
    // => 0 = n^T * LHS = n^T * RHS
    // => n^T * W_t_cur = n^T * W_t_ref
    // At least 3 match pairs needed to solve W_t_cur

    if (matches_2d.size() < 2u) {
        VLOG(5) << "No enough match set to solve translation, at least 2 "
                   "required, only get "
                << matches_2d.size();
        return false;
    }

    int nr_matches = 0;
    for (const auto &ms : matches_2d)
        nr_matches += static_cast<int>(ms.size());
    if (nr_matches < 3) {
        VLOG(5) << "No enough matches to solve translation, at least 3 "
                   "required, only get "
                << nr_matches;
    }

    Eigen::Matrix3f AtA = Eigen::Matrix3f::Zero();
    Eigen::Vector3f Atb = Eigen::Vector3f::Zero();

    const Eigen::Matrix3f W_R_cur = C_T_W->m_R.transpose();
    for (const auto &ms : matches_2d) {
        const Eigen::Matrix3f W_R_ref = ms.m_T1.m_R.transpose();
        const Eigen::Vector3f W_t_ref = -W_R_ref * ms.m_T1.m_t;
        for (const auto &m : ms) {
            const Eigen::Vector2f &ref = m.m_x1, &cur = m.m_x2;
            // This natually weighting match paris by viewing direction
            // parallax.
            Eigen::Vector3f n =
                (W_R_ref * ref.homogeneous().normalized())
                    .cross(W_R_cur * cur.homogeneous().normalized());
            AtA += n * n.transpose();
            Atb += n.dot(W_t_ref) * n;
        }
    }

    Eigen::Vector3f W_t_cur = AtA.ldlt().solve(Atb);
    C_T_W->m_t = -C_T_W->m_R * W_t_cur;

    return true;
}
} // namespace internal

bool RansacAndRefineSuppressFatalError(
    const Options &options, const MatchSet3D &matches_3d,
    const std::vector<MatchSet2D> &matches_2d, Pose *C_T_W,
    std::vector<int> *inliers_3d, std::vector<std::vector<int>> *inliers_2d) {
    // WARNING! WARNIGN! WARNING!
    // TODO(chen): The workaround here may cause memory leak, since the dtor of
    // atomatic object in stacks will not not be called (not sure). But since
    // the FATAL is triggered very rarely, the leak can be ignored?
    google::InstallFailureFunction(internal::JmpOnFatal);
    if (setjmp(internal::jmp_buf))
        return false;

    return RansacAndRefine(
        options, matches_3d, matches_2d, C_T_W, inliers_3d, inliers_2d);
}

bool RansacAndRefine(
    const Options &options, const MatchSet3D &matches_3d,
    const std::vector<MatchSet2D> &matches_2d, Pose *C_T_W,
    std::vector<int> *inliers_3d, std::vector<std::vector<int>> *inliers_2d) {
    options.CheckValidity();
    CHECK_NOTNULL(C_T_W);
    CHECK_NOTNULL(inliers_3d)->clear();
    CHECK_NOTNULL(inliers_2d)->resize(matches_2d.size());
    for (auto &inliers : *inliers_2d)
        inliers.clear();

    if (!Ransac(
            options, matches_3d, matches_2d, C_T_W, inliers_3d, inliers_2d)) {
        VLOG(5) << "Ransac failed";
        return false;
    }

    MatchSet3D inlier_matches_3d;
    inlier_matches_3d.reserve(inliers_3d->size());
    for (int i : *inliers_3d)
        inlier_matches_3d.push_back(matches_3d[i]);

    std::vector<MatchSet2D> inlier_matches_2d;
    inlier_matches_2d.reserve(matches_2d.size());
    for (size_t i = 0; i < matches_2d.size(); i++) {
        const auto &inliers = (*inliers_2d)[i];
        if (inliers.empty())
            continue;
        inlier_matches_2d.resize(1u + inlier_matches_2d.size());
        inlier_matches_2d.back().m_T1 = matches_2d[i].m_T1;
        inlier_matches_2d.back().reserve(inliers.size());
        for (int j : inliers)
            inlier_matches_2d.back().push_back(matches_2d[i][j]);
    }

    if (inlier_matches_3d.empty() &&
        options
            .try_refine_translation_before_optimization_for_2d_only_matches) {
        VLOG(10) << "Refine translation before optimization, result = "
                 << (internal::SolveTranslation(inlier_matches_2d, C_T_W)
                         ? "succ"
                         : "fail");
    }

    const bool enable_gravity_dir_check =
        options.gravity_dir.norm() > std::numeric_limits<float>::epsilon();
    if (!Refine(
            options.focal_length, inlier_matches_3d, inlier_matches_2d, C_T_W,
            enable_gravity_dir_check ? &options.gravity_dir : nullptr)) {
        VLOG(5) << "Refine failed";
        return false;
    }

    return true;
}

} // namespace PS
