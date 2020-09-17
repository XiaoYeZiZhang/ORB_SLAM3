
#ifndef _POSE_SOLVER_H_
#define _POSE_SOLVER_H_

#include <atomic>
#include <memory>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Eigen>

#define CFG_VERBOSE

namespace PS {

extern int REFINE_MAX_ITERATIONS;
extern float REFINE_MIN_DEPTH;
extern float REFINE_STD_FEATURE;
extern float REFINE_STD_GRAVITY;
extern float REFINE_CONVERGE_ROTATION;
extern float REFINE_CONVERGE_TRANSLATION;

extern int REFINE_DL_MAX_ITERATIONS;
extern float REFINE_DL_RADIUS_INITIAL;
extern float REFINE_DL_RADIUS_MIN;
extern float REFINE_DL_RADIUS_MAX;
extern float REFINE_DL_GAIN_RATIO_MIN;
extern float REFINE_DL_GAIN_RATIO_MAX;
extern float REFINE_DL_RADIUS_FACTOR_INCREASE;
extern float REFINE_DL_RADIUS_FACTOR_DECREASE;

class Pose {
public:
    inline Pose operator/(const Pose &T) const {
        Pose dT;
        dT.m_R = m_R * T.m_R.transpose();
        dT.m_t = m_t - dT.m_R * T.m_t;
        return dT;
    }

public:
    Eigen::Matrix3f m_R;
    Eigen::Vector3f m_t;

    Pose() {
        m_R.setIdentity();
        m_t.setZero();
    }
    Pose(const Eigen::Matrix3f &R, const Eigen::Vector3f &t) : m_R(R), m_t(t) {
    }
};

// class Point3D : public Eigen::Vector3f {};
// class Point2D : public Eigen::Vector2f {};
typedef Eigen::Vector3f Point3D;
typedef Eigen::Vector2f Point2D;

class Match3D {
public:
    Match3D() {
    }
    Match3D(const Point3D &X, const Point2D &x) : m_X(X), m_x(x) {
    }
    Point3D m_X;
    Point2D m_x;
};

class Match2D {
public:
    Match2D() {
    }
    Match2D(const Point2D &x1, const Point2D &x2) : m_x1(x1), m_x2(x2) {
    }
    Point2D m_x1, m_x2;
};

class MatchSet3D : public std::vector<Match3D> {};

class MatchSet2D : public std::vector<Match2D> {
public:
    Pose m_T1;
};

struct IterationSummary {
    // Iterations has runed
    int nr_iterations = 0;
    // Current best estimations.
    const Pose *C_T_W = nullptr;
    const MatchSet3D *matches_3d = nullptr;
    const std::vector<MatchSet2D> *matches_2d = nullptr;
    const std::vector<int> *inliers_3d = nullptr;
    const std::vector<std::vector<int>> *inliers_2d = nullptr;
};

enum CallbackReturnType {
    ABORT,          // Abort the iteration, solver will return false
    TERMINATE_SUCC, // Abort the iteration, solver will return true
    CONTINUE        // Continue
};
using IterationCallback =
    std::function<CallbackReturnType(const IterationSummary &summary)>;

// Several convinient callbacks.
class ExternalBreakCallback {
public:
    /// (Semantics of) Breaker should be pointer type
    using Breaker = std::shared_ptr<std::atomic_bool>;
    // using Breaker = volatile bool *;

    explicit ExternalBreakCallback(const Breaker &breaker) : breaker_(breaker) {
        CHECK(breaker != nullptr);
    }

    CallbackReturnType operator()(const IterationSummary &) const {
        return static_cast<bool>(*breaker_) ? CallbackReturnType::ABORT
                                            : CallbackReturnType::CONTINUE;
    }

private:
    const Breaker breaker_;
};

class EarlyBreakBy3DInlierCounting {
public:
    EarlyBreakBy3DInlierCounting(
        int min_nr_3d_matches, int min_nr_3d_inlier_matches,
        double min_3d_inlier_ratio)
        : min_nr_3d_matches_(min_nr_3d_matches),
          min_nr_3d_inlier_matches_(min_nr_3d_inlier_matches),
          min_3d_inlier_ratio_(min_3d_inlier_ratio) {
    }

    CallbackReturnType operator()(const IterationSummary &summary) const {
        return Succ(summary) ? CallbackReturnType::TERMINATE_SUCC
                             : CallbackReturnType::CONTINUE;
    }

private:
    bool Succ(const IterationSummary &summary) const {
        return static_cast<int>(summary.matches_3d->size()) >=
                   min_nr_3d_matches_ &&
               static_cast<int>(summary.inliers_3d->size()) >=
                   min_nr_3d_inlier_matches_ &&
               summary.inliers_3d->size() >=
                   min_3d_inlier_ratio_ * summary.matches_3d->size();
    }

    const int min_nr_3d_matches_, min_nr_3d_inlier_matches_;
    const double min_3d_inlier_ratio_;
};

struct Options {
    float focal_length = -1;
    // max reprojection error (in normalized plane) for a match to be inlier.
    float max_reproj_err = -1;

    // Max ransac iterations.
    int ransac_iterations = 100;
    // ransac confidence
    double ransac_confidence = 0.99;

    // NOTE: Assume gravity direciotn in world frame is [0, 0, 1].
    // Default value [0, 0, 0] implies unknown gravity direction.
    Eigen::Vector3f gravity_dir = Eigen::Vector3f::Zero();
    // 180 degree will essentially disable the gravity direction check.
    float gravity_dir_max_err_deg = 180;

    // PnP based pose solver.
    bool enable_3d_solver = true;
    // Essential matrix based solver.
    bool enable_2d_solver = true;
    // When use 2d solver, we first compute scaleless pose via essential matrix,
    // then we can either sample a 3D match or a 2D match to solve the scale. If
    // prefer_pure_2d_solver, sampleing over 2D matches will be prefered.
    bool prefer_pure_2d_solver = true;
    // Use gravity direction as a constraint to solve the pose, this will reduce
    // the problem by 2DOF. Enable it if your gravity_dir is accurate enough,
    // else the gravity direciton check will be enabled when the gravity
    // direction is valid.
    bool enable_gravity_solver = false;

    // If only have 2D inlier matches, try solve translation (with rotation
    // fixed) by DLT before running optimization.
    bool try_refine_translation_before_optimization_for_2d_only_matches = false;

    std::vector<IterationCallback> callbacks;

    void CheckValidity() const {
        CHECK_NE(focal_length, -1) << "focal_length not set.";
        CHECK_NE(max_reproj_err, -1) << "max_reproj_err not set.";
        float max_reproj_err_pixel = max_reproj_err * focal_length;
        CHECK_LT(max_reproj_err_pixel, 50)
            << "max_reproj_error(" << max_reproj_err_pixel
            << " pixels) seems too large, NOTE the max_reproj_err should be "
               "normalized by focal length.";
    }
};

bool Ransac(
    const Options &options, const MatchSet3D &M3D,
    const std::vector<MatchSet2D> &Ms2D, Pose *T, std::vector<int> *I3D,
    std::vector<std::vector<int>> *I2D);

bool Refine(
    const float f, const MatchSet3D &M3D, const std::vector<MatchSet2D> &Ms2D,
    Pose *T, const Eigen::Vector3f *g = NULL,
    const float min_translation = 0.0f);

bool RansacAndRefine(
    const Options &options, const MatchSet3D &matches_3d,
    const std::vector<MatchSet2D> &matches_2d, Pose *C_T_W,
    std::vector<int> *inliers_3d, std::vector<std::vector<int>> *inliers_2d);

bool RansacAndRefineSuppressFatalError(
    const Options &options, const MatchSet3D &matches_3d,
    const std::vector<MatchSet2D> &matches_2d, Pose *C_T_W,
    std::vector<int> *inliers_3d, std::vector<std::vector<int>> *inliers_2d);

} // namespace PS

namespace Eigen {

inline Matrix3f SkewSymmetric(const Vector3f &a) {
    Matrix3f S;
    S(0, 0) = 0.0f;
    S(0, 1) = -a.z();
    S(0, 2) = a.y();
    S(1, 0) = a.z();
    S(1, 1) = 0.0f;
    S(1, 2) = -a.x();
    S(2, 0) = -a.y();
    S(2, 1) = a.x();
    S(2, 2) = 0.0f;
    return S;
}
} // namespace Eigen

#endif
