#include "essential3.h"
#include <glog/logging.h>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;
namespace PS {
namespace detail {
// p -- world frame
// q -- cur frame
// gravity dierction is undserstood to be unit Z
bool SolveRotationC(
    const Vector2d &p0, const Vector2d &p1, const Vector2d &p2,
    const Vector2d &q0, const Vector2d &q1, const Vector2d &q2,
    std::vector<Matrix3d> *C_R_W) {
    DCHECK_NOTNULL(C_R_W)->clear();
    C_R_W->reserve(6);

    const double x0 = p0.x(), y0 = p0.y(), x1 = p1.x(), y1 = p1.y(),
                 x2 = p2.x(), y2 = p2.y();
    const double u0 = q0.x(), v0 = q0.y(), u1 = q1.x(), v1 = q1.y(),
                 u2 = q2.x(), v2 = q2.y();

    double coeffs[7];
    coeffs[6] = u0 * u1 * v2 * y0 - u0 * u1 * v2 * y1 + u0 * u1 * y0 * y2 -
                u0 * u1 * y1 * y2 - u0 * u2 * v1 * y0 + u0 * u2 * v1 * y2 -
                u0 * u2 * y0 * y1 + u0 * u2 * y1 * y2 + u0 * v1 * v2 * x1 -
                u0 * v1 * v2 * x2 + u0 * v1 * x1 * y2 - u0 * v1 * x2 * y0 +
                u0 * v2 * x1 * y0 - u0 * v2 * x2 * y1 + u0 * x1 * y0 * y2 -
                u0 * x2 * y0 * y1 + u1 * u2 * v0 * y1 - u1 * u2 * v0 * y2 +
                u1 * u2 * y0 * y1 - u1 * u2 * y0 * y2 - u1 * v0 * v2 * x0 +
                u1 * v0 * v2 * x2 - u1 * v0 * x0 * y2 + u1 * v0 * x2 * y1 -
                u1 * v2 * x0 * y1 + u1 * v2 * x2 * y0 - u1 * x0 * y1 * y2 +
                u1 * x2 * y0 * y1 + u2 * v0 * v1 * x0 - u2 * v0 * v1 * x1 +
                u2 * v0 * x0 * y1 - u2 * v0 * x1 * y2 + u2 * v1 * x0 * y2 -
                u2 * v1 * x1 * y0 + u2 * x0 * y1 * y2 - u2 * x1 * y0 * y2 +
                v0 * v1 * x0 * x2 - v0 * v1 * x1 * x2 - v0 * v2 * x0 * x1 +
                v0 * v2 * x1 * x2 - v0 * x0 * x1 * y2 + v0 * x0 * x2 * y1 +
                v1 * v2 * x0 * x1 - v1 * v2 * x0 * x2 + v1 * x0 * x1 * y2 -
                v1 * x1 * x2 * y0 - v2 * x0 * x2 * y1 + v2 * x1 * x2 * y0;
    coeffs[5] =
        -2 * (u0 * u1 * v2 * x0 - u0 * u1 * v2 * x1 + u0 * u1 * x0 * y2 -
              u0 * u1 * x1 * y2 + u0 * u1 * x2 * y0 - u0 * u1 * x2 * y1 -
              u0 * u2 * v1 * x0 + u0 * u2 * v1 * x2 - u0 * u2 * x0 * y1 -
              u0 * u2 * x1 * y0 + u0 * u2 * x1 * y2 + u0 * u2 * x2 * y1 -
              u0 * v1 * v2 * y1 + u0 * v1 * v2 * y2 - u0 * v1 * x0 * x2 +
              u0 * v1 * x1 * x2 + u0 * v1 * y0 * y2 - u0 * v1 * y1 * y2 +
              u0 * v2 * x0 * x1 - u0 * v2 * x1 * x2 - u0 * v2 * y0 * y1 +
              u0 * v2 * y1 * y2 + u0 * x0 * x1 * y2 - u0 * x0 * x2 * y1 +
              u1 * u2 * v0 * x1 - u1 * u2 * v0 * x2 + u1 * u2 * x0 * y1 -
              u1 * u2 * x0 * y2 + u1 * u2 * x1 * y0 - u1 * u2 * x2 * y0 +
              u1 * v0 * v2 * y0 - u1 * v0 * v2 * y2 - u1 * v0 * x0 * x2 +
              u1 * v0 * x1 * x2 + u1 * v0 * y0 * y2 - u1 * v0 * y1 * y2 -
              u1 * v2 * x0 * x1 + u1 * v2 * x0 * x2 + u1 * v2 * y0 * y1 -
              u1 * v2 * y0 * y2 - u1 * x0 * x1 * y2 + u1 * x1 * x2 * y0 -
              u2 * v0 * v1 * y0 + u2 * v0 * v1 * y1 + u2 * v0 * x0 * x1 -
              u2 * v0 * x1 * x2 - u2 * v0 * y0 * y1 + u2 * v0 * y1 * y2 -
              u2 * v1 * x0 * x1 + u2 * v1 * x0 * x2 + u2 * v1 * y0 * y1 -
              u2 * v1 * y0 * y2 + u2 * x0 * x2 * y1 - u2 * x1 * x2 * y0 -
              v0 * v1 * x0 * y2 + v0 * v1 * x1 * y2 - v0 * v1 * x2 * y0 +
              v0 * v1 * x2 * y1 + v0 * v2 * x0 * y1 + v0 * v2 * x1 * y0 -
              v0 * v2 * x1 * y2 - v0 * v2 * x2 * y1 + v0 * x1 * y0 * y2 -
              v0 * x2 * y0 * y1 - v1 * v2 * x0 * y1 + v1 * v2 * x0 * y2 -
              v1 * v2 * x1 * y0 + v1 * v2 * x2 * y0 - v1 * x0 * y1 * y2 +
              v1 * x2 * y0 * y1 + v2 * x0 * y1 * y2 - v2 * x1 * y0 * y2);
    coeffs[4] =
        u0 * u1 * v2 * y0 - u0 * u1 * v2 * y1 + 4 * u0 * u1 * x0 * x2 -
        4 * u0 * u1 * x1 * x2 - u0 * u1 * y0 * y2 + u0 * u1 * y1 * y2 -
        u0 * u2 * v1 * y0 + u0 * u2 * v1 * y2 - 4 * u0 * u2 * x0 * x1 +
        4 * u0 * u2 * x1 * x2 + u0 * u2 * y0 * y1 - u0 * u2 * y1 * y2 +
        u0 * v1 * v2 * x1 - u0 * v1 * v2 * x2 + 4 * u0 * v1 * x0 * y2 -
        u0 * v1 * x1 * y2 + u0 * v1 * x2 * y0 - 4 * u0 * v1 * x2 * y1 -
        4 * u0 * v2 * x0 * y1 - u0 * v2 * x1 * y0 + 4 * u0 * v2 * x1 * y2 +
        u0 * v2 * x2 * y1 + u0 * x1 * y0 * y2 - u0 * x2 * y0 * y1 +
        u1 * u2 * v0 * y1 - u1 * u2 * v0 * y2 + 4 * u1 * u2 * x0 * x1 -
        4 * u1 * u2 * x0 * x2 - u1 * u2 * y0 * y1 + u1 * u2 * y0 * y2 -
        u1 * v0 * v2 * x0 + u1 * v0 * v2 * x2 + u1 * v0 * x0 * y2 -
        4 * u1 * v0 * x1 * y2 + 4 * u1 * v0 * x2 * y0 - u1 * v0 * x2 * y1 +
        u1 * v2 * x0 * y1 - 4 * u1 * v2 * x0 * y2 + 4 * u1 * v2 * x1 * y0 -
        u1 * v2 * x2 * y0 - u1 * x0 * y1 * y2 + u1 * x2 * y0 * y1 +
        u2 * v0 * v1 * x0 - u2 * v0 * v1 * x1 - u2 * v0 * x0 * y1 -
        4 * u2 * v0 * x1 * y0 + u2 * v0 * x1 * y2 + 4 * u2 * v0 * x2 * y1 +
        4 * u2 * v1 * x0 * y1 - u2 * v1 * x0 * y2 + u2 * v1 * x1 * y0 -
        4 * u2 * v1 * x2 * y0 + u2 * x0 * y1 * y2 - u2 * x1 * y0 * y2 -
        v0 * v1 * x0 * x2 + v0 * v1 * x1 * x2 + 4 * v0 * v1 * y0 * y2 -
        4 * v0 * v1 * y1 * y2 + v0 * v2 * x0 * x1 - v0 * v2 * x1 * x2 -
        4 * v0 * v2 * y0 * y1 + 4 * v0 * v2 * y1 * y2 - v0 * x0 * x1 * y2 +
        v0 * x0 * x2 * y1 - v1 * v2 * x0 * x1 + v1 * v2 * x0 * x2 +
        4 * v1 * v2 * y0 * y1 - 4 * v1 * v2 * y0 * y2 + v1 * x0 * x1 * y2 -
        v1 * x1 * x2 * y0 - v2 * x0 * x2 * y1 + v2 * x1 * x2 * y0;
    coeffs[3] =
        -4 * (u0 * u1 * v2 * x0 - u0 * u1 * v2 * x1 - u0 * u2 * v1 * x0 +
              u0 * u2 * v1 * x2 - u0 * v1 * v2 * y1 + u0 * v1 * v2 * y2 +
              u0 * x0 * x1 * y2 - u0 * x0 * x2 * y1 + u1 * u2 * v0 * x1 -
              u1 * u2 * v0 * x2 + u1 * v0 * v2 * y0 - u1 * v0 * v2 * y2 -
              u1 * x0 * x1 * y2 + u1 * x1 * x2 * y0 - u2 * v0 * v1 * y0 +
              u2 * v0 * v1 * y1 + u2 * x0 * x2 * y1 - u2 * x1 * x2 * y0 +
              v0 * x1 * y0 * y2 - v0 * x2 * y0 * y1 - v1 * x0 * y1 * y2 +
              v1 * x2 * y0 * y1 + v2 * x0 * y1 * y2 - v2 * x1 * y0 * y2);
    coeffs[2] =
        -u0 * u1 * v2 * y0 + u0 * u1 * v2 * y1 + 4 * u0 * u1 * x0 * x2 -
        4 * u0 * u1 * x1 * x2 - u0 * u1 * y0 * y2 + u0 * u1 * y1 * y2 +
        u0 * u2 * v1 * y0 - u0 * u2 * v1 * y2 - 4 * u0 * u2 * x0 * x1 +
        4 * u0 * u2 * x1 * x2 + u0 * u2 * y0 * y1 - u0 * u2 * y1 * y2 -
        u0 * v1 * v2 * x1 + u0 * v1 * v2 * x2 + 4 * u0 * v1 * x0 * y2 -
        u0 * v1 * x1 * y2 + u0 * v1 * x2 * y0 - 4 * u0 * v1 * x2 * y1 -
        4 * u0 * v2 * x0 * y1 - u0 * v2 * x1 * y0 + 4 * u0 * v2 * x1 * y2 +
        u0 * v2 * x2 * y1 - u0 * x1 * y0 * y2 + u0 * x2 * y0 * y1 -
        u1 * u2 * v0 * y1 + u1 * u2 * v0 * y2 + 4 * u1 * u2 * x0 * x1 -
        4 * u1 * u2 * x0 * x2 - u1 * u2 * y0 * y1 + u1 * u2 * y0 * y2 +
        u1 * v0 * v2 * x0 - u1 * v0 * v2 * x2 + u1 * v0 * x0 * y2 -
        4 * u1 * v0 * x1 * y2 + 4 * u1 * v0 * x2 * y0 - u1 * v0 * x2 * y1 +
        u1 * v2 * x0 * y1 - 4 * u1 * v2 * x0 * y2 + 4 * u1 * v2 * x1 * y0 -
        u1 * v2 * x2 * y0 + u1 * x0 * y1 * y2 - u1 * x2 * y0 * y1 -
        u2 * v0 * v1 * x0 + u2 * v0 * v1 * x1 - u2 * v0 * x0 * y1 -
        4 * u2 * v0 * x1 * y0 + u2 * v0 * x1 * y2 + 4 * u2 * v0 * x2 * y1 +
        4 * u2 * v1 * x0 * y1 - u2 * v1 * x0 * y2 + u2 * v1 * x1 * y0 -
        4 * u2 * v1 * x2 * y0 - u2 * x0 * y1 * y2 + u2 * x1 * y0 * y2 -
        v0 * v1 * x0 * x2 + v0 * v1 * x1 * x2 + 4 * v0 * v1 * y0 * y2 -
        4 * v0 * v1 * y1 * y2 + v0 * v2 * x0 * x1 - v0 * v2 * x1 * x2 -
        4 * v0 * v2 * y0 * y1 + 4 * v0 * v2 * y1 * y2 + v0 * x0 * x1 * y2 -
        v0 * x0 * x2 * y1 - v1 * v2 * x0 * x1 + v1 * v2 * x0 * x2 +
        4 * v1 * v2 * y0 * y1 - 4 * v1 * v2 * y0 * y2 - v1 * x0 * x1 * y2 +
        v1 * x1 * x2 * y0 + v2 * x0 * x2 * y1 - v2 * x1 * x2 * y0;
    coeffs[1] =
        -2 * (u0 * u1 * v2 * x0 - u0 * u1 * v2 * x1 - u0 * u1 * x0 * y2 +
              u0 * u1 * x1 * y2 - u0 * u1 * x2 * y0 + u0 * u1 * x2 * y1 -
              u0 * u2 * v1 * x0 + u0 * u2 * v1 * x2 + u0 * u2 * x0 * y1 +
              u0 * u2 * x1 * y0 - u0 * u2 * x1 * y2 - u0 * u2 * x2 * y1 -
              u0 * v1 * v2 * y1 + u0 * v1 * v2 * y2 + u0 * v1 * x0 * x2 -
              u0 * v1 * x1 * x2 - u0 * v1 * y0 * y2 + u0 * v1 * y1 * y2 -
              u0 * v2 * x0 * x1 + u0 * v2 * x1 * x2 + u0 * v2 * y0 * y1 -
              u0 * v2 * y1 * y2 + u0 * x0 * x1 * y2 - u0 * x0 * x2 * y1 +
              u1 * u2 * v0 * x1 - u1 * u2 * v0 * x2 - u1 * u2 * x0 * y1 +
              u1 * u2 * x0 * y2 - u1 * u2 * x1 * y0 + u1 * u2 * x2 * y0 +
              u1 * v0 * v2 * y0 - u1 * v0 * v2 * y2 + u1 * v0 * x0 * x2 -
              u1 * v0 * x1 * x2 - u1 * v0 * y0 * y2 + u1 * v0 * y1 * y2 +
              u1 * v2 * x0 * x1 - u1 * v2 * x0 * x2 - u1 * v2 * y0 * y1 +
              u1 * v2 * y0 * y2 - u1 * x0 * x1 * y2 + u1 * x1 * x2 * y0 -
              u2 * v0 * v1 * y0 + u2 * v0 * v1 * y1 - u2 * v0 * x0 * x1 +
              u2 * v0 * x1 * x2 + u2 * v0 * y0 * y1 - u2 * v0 * y1 * y2 +
              u2 * v1 * x0 * x1 - u2 * v1 * x0 * x2 - u2 * v1 * y0 * y1 +
              u2 * v1 * y0 * y2 + u2 * x0 * x2 * y1 - u2 * x1 * x2 * y0 +
              v0 * v1 * x0 * y2 - v0 * v1 * x1 * y2 + v0 * v1 * x2 * y0 -
              v0 * v1 * x2 * y1 - v0 * v2 * x0 * y1 - v0 * v2 * x1 * y0 +
              v0 * v2 * x1 * y2 + v0 * v2 * x2 * y1 + v0 * x1 * y0 * y2 -
              v0 * x2 * y0 * y1 + v1 * v2 * x0 * y1 - v1 * v2 * x0 * y2 +
              v1 * v2 * x1 * y0 - v1 * v2 * x2 * y0 - v1 * x0 * y1 * y2 +
              v1 * x2 * y0 * y1 + v2 * x0 * y1 * y2 - v2 * x1 * y0 * y2);
    coeffs[0] = -u0 * u1 * v2 * y0 + u0 * u1 * v2 * y1 + u0 * u1 * y0 * y2 -
                u0 * u1 * y1 * y2 + u0 * u2 * v1 * y0 - u0 * u2 * v1 * y2 -
                u0 * u2 * y0 * y1 + u0 * u2 * y1 * y2 - u0 * v1 * v2 * x1 +
                u0 * v1 * v2 * x2 + u0 * v1 * x1 * y2 - u0 * v1 * x2 * y0 +
                u0 * v2 * x1 * y0 - u0 * v2 * x2 * y1 - u0 * x1 * y0 * y2 +
                u0 * x2 * y0 * y1 - u1 * u2 * v0 * y1 + u1 * u2 * v0 * y2 +
                u1 * u2 * y0 * y1 - u1 * u2 * y0 * y2 + u1 * v0 * v2 * x0 -
                u1 * v0 * v2 * x2 - u1 * v0 * x0 * y2 + u1 * v0 * x2 * y1 -
                u1 * v2 * x0 * y1 + u1 * v2 * x2 * y0 + u1 * x0 * y1 * y2 -
                u1 * x2 * y0 * y1 - u2 * v0 * v1 * x0 + u2 * v0 * v1 * x1 +
                u2 * v0 * x0 * y1 - u2 * v0 * x1 * y2 + u2 * v1 * x0 * y2 -
                u2 * v1 * x1 * y0 - u2 * x0 * y1 * y2 + u2 * x1 * y0 * y2 +
                v0 * v1 * x0 * x2 - v0 * v1 * x1 * x2 - v0 * v2 * x0 * x1 +
                v0 * v2 * x1 * x2 + v0 * x0 * x1 * y2 - v0 * x0 * x2 * y1 +
                v1 * v2 * x0 * x1 - v1 * v2 * x0 * x2 - v1 * x0 * x1 * y2 +
                v1 * x1 * x2 * y0 + v2 * x0 * x2 * y1 - v2 * x1 * x2 * y0;

    std::vector<cv::Complex<double>> roots;
    solvePoly(cv::Mat(1, 7, CV_64F, coeffs), roots);

    for (size_t i = 0; i < roots.size(); i++) {
        if (std::abs(roots[i].im) > 1E-10)
            continue;
        double x = roots[i].re, c = (1 - x * x) / (1 + x * x),
               s = 2 * x / (1 + x * x);

        Matrix3d R;
        R << c, -s, 0, //
            s, c, 0,   //
            0, 0, 1;

        C_R_W->push_back(R);
    }

    return !C_R_W->empty();
}

bool SolveRotation(
    const Vector2d &p0, const Vector2d &p1, const Vector2d &p2,
    const Vector2d &q0, const Vector2d &q1, const Vector2d &q2,
    const Vector3d &gravity_dir /* in world frame */,
    std::vector<Matrix3d> *C_R_W) {
    DCHECK_NOTNULL(C_R_W)->clear();

    const Matrix3d unit_z_R_gravity_dir =
                       Eigen::Quaterniond::FromTwoVectors(
                           gravity_dir, Eigen::Vector3d::UnitZ())
                           .toRotationMatrix(),
                   gravity_dir_R_unit_z = unit_z_R_gravity_dir.transpose();

    // TODO(chen): The hnormalized operation maybe dangerous, figure out and
    // hangle these cases.

    // TODO(zhangye): Check eigen problem
    if (!SolveRotationC(
            p0, p1, p2,
            (unit_z_R_gravity_dir * Vector3d(q0(0), q0(1), 1.0)).head<2>() /
                (unit_z_R_gravity_dir * Vector3d(q0(0), q0(1), 1.0))(2),
            (unit_z_R_gravity_dir * Vector3d(q1(0), q1(1), 1.0)).head<2>() /
                (unit_z_R_gravity_dir * Vector3d(q1(0), q1(1), 1.0))(2),
            (unit_z_R_gravity_dir * Vector3d(q2(0), q2(1), 1.0)).head<2>() /
                (unit_z_R_gravity_dir * Vector3d(q2(0), q2(1), 1.0))(2),
            C_R_W))
        return false;

    for (Matrix3d &R : *C_R_W)
        R = gravity_dir_R_unit_z * R;

    return !C_R_W->empty();
}

bool SolveRotation(
    const Vector2d &p0, const Vector2d &p1, const Vector2d &p2,
    const Vector2d &q0, const Vector2d &q1, const Vector2d &q2,
    const Matrix3d &ref_R_world,
    const Vector3d &gravity_dir /* in world frame */,
    std::vector<Matrix3d> *cur_R_ref) {
    DCHECK_NOTNULL(cur_R_ref)->clear();

    // TODO(chen): The hnormalized operation maybe dangerous, figure out and
    // hangle these cases.
    // TODO(zhangye): check eigen problem
    if (!SolveRotation(
            (ref_R_world.transpose() * Vector3d(p0(0), p0(1), 1.0)).head<2>() /
                (ref_R_world.transpose() *
                 Vector3d(p0(0), p0(1), 1.0))(2), // hnormalized(),
            (ref_R_world.transpose() * Vector3d(p1(0), p1(1), 1.0)).head<2>() /
                (ref_R_world.transpose() *
                 Vector3d(p1(0), p1(1), 1.0))(2), //.hnormalized(),
            (ref_R_world.transpose() * Vector3d(p2(0), p2(1), 1.0)).head<2>() /
                (ref_R_world.transpose() * Vector3d(p2(0), p2(1), 1.0))(2),
            q0, q1, q2, gravity_dir, cur_R_ref))
        return false;

    for (Matrix3d &R : *cur_R_ref)
        R = R * ref_R_world.transpose();

    return true;
}

// Solve cur_t_ref, NOTE since it's unscaled, caller should ALSO USE negate of
// the returned translation.
Vector3d SolveTranslation(
    const Vector2d &p0, const Vector2d &p1, const Vector2d &q0,
    const Vector2d &q1, const Matrix3d &cur_R_ref) {
    Vector3d t =
        q0.homogeneous()
            .cross(cur_R_ref * p0.homogeneous())
            .cross(q1.homogeneous().cross(cur_R_ref * p1.homogeneous()));
    return t.norm() < 1E-6 ? Vector3d::Zero() : t.normalized();
}
} // namespace detail

bool Essential3(
    const std::vector<Vector2d> &pts_ref, const std::vector<Vector2d> &pts_cur,
    const Matrix3d &ref_R_world, const Vector3d &W_gravity_dir,
    std::vector<Matrix3d> *cur_R_ref, std::vector<Vector3d> *cur_t_ref) {
    CHECK_NOTNULL(cur_R_ref)->clear();
    CHECK_NOTNULL(cur_t_ref)->clear();
    CHECK_EQ(pts_ref.size(), 3u);
    CHECK_EQ(pts_cur.size(), 3u);

    if (!detail::SolveRotation(
            pts_ref[0], pts_ref[1], pts_ref[2], pts_cur[0], pts_cur[1],
            pts_cur[2], ref_R_world, W_gravity_dir, cur_R_ref))
        return false;

    CHECK(!cur_R_ref->empty());

    cur_t_ref->reserve(cur_R_ref->size());
    for (const Matrix3d &R : *cur_R_ref) {
        cur_t_ref->push_back(detail::SolveTranslation(
            pts_ref[0], pts_ref[1], pts_cur[0], pts_cur[1], R));
    }

    return true;
}
} // namespace PS
