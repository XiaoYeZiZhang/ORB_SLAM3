
#ifndef _POSE_SOLVER_H_
#define _POSE_SOLVER_H_

#include <atomic>
#include <memory>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Eigen>

namespace PoseSolver {
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
    int nr_iterations = 0;
    const MatchSet3D *matches_3d = nullptr;
    const std::vector<MatchSet2D> *matches_2d = nullptr;
    const std::vector<int> *inliers_3d = nullptr;
    const std::vector<std::vector<int>> *inliers_2d = nullptr;
};

enum CallbackReturnType { ABORT, TERMINATE_SUCC, CONTINUE };
using IterationCallback =
    std::function<CallbackReturnType(const IterationSummary &summary)>;

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
    float max_reproj_err = -1;
    int ransac_iterations = 100;
    double ransac_confidence = 0.99;
    std::vector<IterationCallback> callbacks;
};

bool Ransac_Tracker(
    const Options &options, const MatchSet3D &M3D,
    const std::vector<MatchSet2D> &Ms2D, Pose *T, std::vector<int> *I3D,
    std::vector<std::vector<int>> *I2D);

bool Ransac_Detector(
    const Options &options, const MatchSet3D &M3D,
    const std::vector<MatchSet2D> &Ms2D, Pose *T, std::vector<int> *I3D,
    std::vector<std::vector<int>> *I2D);
} // namespace PoseSolver

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
