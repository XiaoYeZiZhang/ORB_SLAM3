#include "PoseSolver.h"

#include <cfloat>
#include <random>
#include <utility>
#include <glog/logging.h>
#include "Generator.h"

using Eigen::Matrix3f;
using Eigen::Vector2f;
using Eigen::Vector3f;

constexpr float kPureRotationTranslationTh = 0.05;
constexpr bool kPureRotationReprojectionErrorScale = 5;

namespace PS {
/*
 * Returns: #inliers = #3d_inliers + #2d_innliers
 */
int Evaluate(
    const float reproj_err, const MatchSet3D &M3D,
    const std::vector<MatchSet2D> &Ms2D, const Pose &T, std::vector<int> *I3D,
    std::vector<std::vector<int>> *I2D) {
    const float e2Max = reproj_err * reproj_err;
    CHECK_NOTNULL(I3D)->clear();
    CHECK_NOTNULL(I2D)->clear();

    const int N3D = static_cast<int>(M3D.size());
    for (int i = 0; i < N3D; ++i) {
        const Match3D &M = M3D[i];
        const Vector3f TX = T.m_R * M.m_X + T.m_t;
        if (TX.z() <= 0.0f) {
            continue;
        }
        const Vector2f x = Vector2f(TX.x(), TX.y()) / TX.z();
        const Vector2f e = x - M.m_x;
        const float e2 = e.squaredNorm();
        if (e2 < e2Max) {
            I3D->push_back(i);
        }
    }
    int SN = static_cast<int>(I3D->size());

    const int Ns2D = static_cast<int>(Ms2D.size());
    I2D->resize(Ns2D);
    for (int i = 0; i < Ns2D; ++i) {
        const MatchSet2D &M2D = Ms2D[i];
        const Pose T21 = T / M2D.m_T1;
        const bool isPureRotation =
            T21.m_t.squaredNorm() <
            kPureRotationTranslationTh * kPureRotationTranslationTh;
        std::vector<int> &I2Di = I2D->at(i);
        I2Di.clear();
        const int N2D = static_cast<int>(M2D.size());
        if (!isPureRotation) {
            const Matrix3f E = Eigen::SkewSymmetric(T21.m_t) * T21.m_R;
            for (int j = 0; j < N2D; ++j) {
                const Match2D &M = M2D[j];
                const Vector3f x1(M.m_x1.x(), M.m_x1.y(), 1.0f);
                const Vector3f l = E * x1;
                const float s2I = l.x() * l.x() + l.y() * l.y();
                if (s2I < FLT_EPSILON) {
                    continue;
                }
                const Vector3f x2(M.m_x2.x(), M.m_x2.y(), 1.0f);
                const float s2 = 1 / s2I, s = sqrtf(s2);
                const float d = x2.dot(l);
                const float e = s * d;
                const float e2 = e * e;
                if (e2 < e2Max) {
                    I2Di.push_back(j);
                }
            }
        } else {
            VLOG(5) << "Pure rotation detected.";
            for (int j = 0; j < N2D; ++j) {
                const Match2D &M = M2D[j];
                const Vector3f x1(M.m_x1.x(), M.m_x1.y(), 1.0f);
                const Vector3f l = T21.m_R * x1;
                if (l.z() < FLT_EPSILON) {
                    continue;
                }
                const float e2 = (M.m_x2 - l.hnormalized()).squaredNorm();
                if (e2 < kPureRotationReprojectionErrorScale *
                             kPureRotationReprojectionErrorScale * e2Max) {
                    I2Di.push_back(j);
                }
            }
        }
        SN += static_cast<int>(I2Di.size());
    }

    return SN;
}

std::unique_ptr<HypoGenerator>
MakeHypoGenerator(const MatchData &match_data, const Options &options) {
    const bool valid_gravity_dir =
        options.gravity_dir.norm() > std::numeric_limits<float>::epsilon();
    if (valid_gravity_dir)
        CHECK_LT(std::abs(options.gravity_dir.norm() - 1), 1E-3F);
    else
        CHECK(!options.enable_gravity_solver);

    bool enable_gravity_dir_check =
        valid_gravity_dir && !options.enable_gravity_solver;

    const int nr_views = static_cast<int>(match_data.matches_2d.size());
    std::unique_ptr<RoundRobinGenerator> rrhypo =
        std::make_unique<RoundRobinGenerator>();
    if (options.enable_2d_solver) {
        std::unique_ptr<ScaleFreeGenerator> scale_less_solver =
            options.enable_gravity_solver
                ? std::unique_ptr<ScaleFreeGenerator>(
                      MakeIfHasEnoughSupport<Essential3Generator>(
                          match_data, options.gravity_dir))
                : MakeIfHasEnoughSupport<Essential5Generator>(match_data);

        if (scale_less_solver) {
            std::unique_ptr<ScaleOnlyGenerator> scale_solver =

                options.prefer_pure_2d_solver
                    ? std::unique_ptr<ScaleOnlyGenerator>(
                          MakeIfHasEnoughSupport<ScaleSolver2D>(match_data))
                    : MakeIfHasEnoughSupport<ScaleSolver3D>(match_data);
            if (!scale_solver)
                scale_solver =
                    !options.prefer_pure_2d_solver
                        ? std::unique_ptr<ScaleOnlyGenerator>(
                              MakeIfHasEnoughSupport<ScaleSolver2D>(match_data))
                        : MakeIfHasEnoughSupport<ScaleSolver3D>(match_data);
            if (scale_solver)
                rrhypo->Add(
                    MakeIfHasEnoughSupport<HybridGenerator>(
                        std::move(scale_less_solver), std::move(scale_solver)),
                    nr_views);
        }
    }

    if (options.enable_3d_solver) {
        std::unique_ptr<HypoGenerator> hypo_3d =
            options.enable_gravity_solver
                ? std::unique_ptr<HypoGenerator>(
                      MakeIfHasEnoughSupport<P2PGenerator>(
                          match_data, options.gravity_dir))
                : MakeIfHasEnoughSupport<PnPGenerator>(match_data);
        if (hypo_3d)
            rrhypo->Add(std::move(hypo_3d), 1);
    }

    if (rrhypo->Empty()) {
        LOG(ERROR) << "No enough match data to solve camera pose.";
        return nullptr;
    }

    std::unique_ptr<HypoGenerator> hypo = std::move(rrhypo);
    if (enable_gravity_dir_check)
        hypo = std::make_unique<GravityCheckGenerator>(
            std::move(hypo), options.gravity_dir,
            options.gravity_dir_max_err_deg);

    return hypo;
}

bool Ransac(
    const Options &options, const MatchSet3D &matches_3d,
    const std::vector<MatchSet2D> &matches_2d, Pose *T,
    std::vector<int> *inliers_3d, std::vector<std::vector<int>> *inliers_2d) {
    CHECK_NOTNULL(inliers_3d)->clear();
    CHECK_NOTNULL(inliers_2d)->clear();
    inliers_2d->resize(matches_2d.size());

    const MatchData match_data(matches_2d, matches_3d);
    std::unique_ptr<HypoGenerator> hypo =
        MakeHypoGenerator(match_data, options);
    if (!hypo) {
        LOG(ERROR) << "Can not generate hypothesis, no enough matches.";
        return false;
    }

    int nr_observations_3d = static_cast<int>(matches_3d.size()),
        nr_observations_2d = 0;
    for (const auto &ms : matches_2d)
        nr_observations_2d += static_cast<int>(ms.size());
    int nr_observations = nr_observations_3d + nr_observations_2d;

    Pose T_iter;
    std::vector<int> inliers_3d_iter;
    std::vector<std::vector<int>> inliers_2d_iter;

    int iter, max_iter = options.ransac_iterations, max_nr_inliers = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        if (!options.callbacks.empty()) {
            IterationSummary summary;
            summary.nr_iterations = iter;
            summary.C_T_W = T;
            summary.matches_3d = &matches_3d;
            summary.matches_2d = &matches_2d;
            summary.inliers_3d = inliers_3d;
            summary.inliers_2d = inliers_2d;

            for (auto &cb : options.callbacks) {
                CallbackReturnType action = cb(summary);
                if (action == CallbackReturnType::ABORT) {
                    VLOG(5) << "User abort iteration";
                    return false;
                }
                if (action == CallbackReturnType::TERMINATE_SUCC) {
                    VLOG(5) << "User teminated iteration with succ";
                    return true;
                }

                CHECK_EQ(action, CallbackReturnType::CONTINUE);
            }
        }

        bool succ = hypo->RunOnce(&T_iter);

        if (!succ) {
            VLOG(10) << "Hypothesis generation failed, continue.";
            continue;
        }

        const int nr_inliers_iter = Evaluate(
            options.max_reproj_err, matches_3d, matches_2d, T_iter,
            &inliers_3d_iter, &inliers_2d_iter);

        if (nr_inliers_iter <= max_nr_inliers) {
            VLOG(10) << "#inliers = " << nr_inliers_iter << ", reject.";
            continue;
        }

        max_nr_inliers = nr_inliers_iter;
        *T = T_iter;
        std::swap(*inliers_3d, inliers_3d_iter);
        std::swap(*inliers_2d, inliers_2d_iter);

        if (nr_inliers_iter == nr_observations) {
            VLOG(10) << "#inlier_ratio = 100%, accept and break.";
            ++iter;
            break;
        }

        const int nr_inliers_3d = static_cast<int>(inliers_3d->size()),
                  nr_inliers_2d = max_nr_inliers - nr_inliers_3d;
        const double inlier_ratio =
                         static_cast<double>(max_nr_inliers) / nr_observations,
                     inlier_ratio_3d = static_cast<double>(nr_inliers_3d) /
                                       nr_observations_3d,
                     inlier_ratio_2d = static_cast<double>(nr_inliers_2d) /
                                       nr_observations_2d;

        const double succ_prob = hypo->SuccProb(
            nr_observations_3d == 0 ? 0 : inlier_ratio_3d,
            nr_observations_2d == 0 ? 0 : inlier_ratio_2d);

        max_iter = std::min(
            max_iter, static_cast<int>(std::ceil(
                          std::log(1.0 - options.ransac_confidence) /
                          std::log(1.0 - succ_prob))));

        // In case of overflow.
        if (max_iter < 0)
            max_iter = options.ransac_iterations;

        VLOG(10) << "total_#inliers = " << max_nr_inliers
                 << ", total_inlier_ratio = " << inlier_ratio
                 << ", #3D_inliers = " << nr_inliers_3d
                 << ", 3D_inlier_ratio = " << inlier_ratio_3d
                 << ", #2D_inliers = " << nr_inliers_2d
                 << ", 2D_inlier_ratio = " << inlier_ratio_2d
                 << ", succ_prob = " << succ_prob
                 << ", adjust iteration count to " << max_iter;
    }

    VLOG(5) << "Ransac runed " << iter << " iterations.";

    return iter < options.ransac_iterations;
}

} // namespace PS
