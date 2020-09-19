#include "p2p.h"

#include <limits>

#include <glog/logging.h>

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

// #define _PS_DEBUG_P2P_
// Reject solution if points lies behind camera.
#ifdef _PS_DEBUG_P2P_
static constexpr bool kEnableDepthCheck = false;
#else
static constexpr bool kEnableDepthCheck = true;
#endif

namespace PS {
namespace detail {
// Gravity direction is assumed to be [0, 0, 1] if not specified.
// Cap letters like X, Y -- world points
// low case letters like x, y -- image points
bool SolveRotationC(
    const Vector3d &X, const Vector2d &x, const Vector3d &Y, const Vector2d &y,
    std::vector<Matrix3d> *C_R_W) {
    DCHECK_NOTNULL(C_R_W)->clear();
    C_R_W->reserve(2);

    Vector3d W_dir = (X - Y).normalized();
    if (W_dir.z() < 0)
        W_dir = -W_dir;
    double r2 = W_dir.hnormalized().squaredNorm();
    // circle equation: z = 1, x^2 + y^2 = r^2

    // TODO(chen): If <normal, UnitZ> ~= 1, the rotation is unsolvable, abort
    // and false the result. Note this is very unlikely in our case, especially
    // in a RANSAC framework.

    Eigen::Vector3d normal =
        x.homogeneous().cross(y.homogeneous()).normalized();
    // line equation: z = 1, <(x, y, z), normal> = 0

    // Interset the line and circle:
    //
    // x ~ dx, y ~ dy
    // => x^2 + y^2 - r^2 = 0
    //    x * n.x + y * n.y + n.z = 0
    // => (n.y/n.x * y + n.z/n.x)^2 + y^2 - r^2 = 0;
    // => a * y^2 + b * y + c = 0
    bool swap_xy = std::abs(normal.x()) < std::abs(normal.y());
    if (swap_xy)
        std::swap(normal.x(), normal.y());

    normal /= normal.x();
    double ny = normal.y(), nz = normal.z();
    double a = ny * ny + 1;
    double b = 2 * ny * nz;
    double c = nz * nz - r2;

    double d = b * b - 4 * a * c;
    if (d < 0) {
        VLOG(5) << "P2P: no solution for 2nd order equation, d = " << d;
        return false;
    }

    // Note a > 1.
    double dy1 = b < 0 ? (-b + std::sqrt(d)) / (2 * a)
                       : (-b - std::sqrt(d)) / (2 * a),
           dy2 = -b / a - dy1;
    double dx1 = -(ny * dy1 + nz), dx2 = -(ny * dy2 + nz);

    if (swap_xy) {
        std::swap(dx1, dy1);
        std::swap(dx2, dy2);
    }

    // Ill condition check.
    if (!std::isfinite(dx1) || !std::isfinite(dx2) || !std::isfinite(dy1) ||
        !std::isfinite(dy2)) {
        VLOG(5) << "Out of range value detected, problem may be ill "
                   "conditiond, (dx1, dy1, dx2, dy2) = ("
                << dx1 << ", " << dy1 << ", " << dx2 << ", " << dy2 << ")";
    }

    double angle1 = std::atan2(dy1, dx1) - std::atan2(W_dir.y(), W_dir.x()),
           angle2 = std::atan2(dy2, dx2) - std::atan2(W_dir.y(), W_dir.x());

    C_R_W->push_back(
        Eigen::AngleAxisd(angle1, Vector3d::UnitZ()).toRotationMatrix());

    constexpr double kTinyAngleDeg = 0.001,
                     kTinyAngleRad = kTinyAngleDeg * M_PI / 180;
    if (std::abs(angle1 - angle2) > kTinyAngleRad) {
        C_R_W->push_back(
            Eigen::AngleAxisd(angle2, Vector3d::UnitZ()).toRotationMatrix());
    } else {
        VLOG(10) << "Angle too close, only one solution found, angle diff = "
                 << std::abs(angle1 - angle2) * 180 / M_PI;
    }

    // Note the real and solved X--Y direciton may be opposite to each other. In
    // this case, the 3D point may behind camera. TODO(chen): Add check.

#ifdef _PS_DEBUG_P2P_
    {
        CHECK(!C_R_W->empty());
        constexpr double kMaxVol = 1E-9;

        // Check rotated direction lies on the plane defined by x, y
        double err =
            (C_R_W->front() * (X - Y))
                .normalized()
                .dot(x.homogeneous().cross(y.homogeneous()).normalized());
        CHECK_LT(std::abs(err), kMaxVol);

        err = (C_R_W->back() * (X - Y))
                  .normalized()
                  .dot(x.homogeneous().cross(y.homogeneous()).normalized());
        CHECK_LT(std::abs(err), kMaxVol);
    }
#endif

    return true;
}

bool SolveRotation(
    const Vector3d &X, const Vector2d &x, const Vector3d &Y, const Vector2d &y,
    const Eigen::Vector3d &gravity_dir, std::vector<Matrix3d> *C_R_W) {
    const Matrix3d unit_z_R_gravity_dir =
                       Eigen::Quaterniond::FromTwoVectors(
                           gravity_dir, Eigen::Vector3d::UnitZ())
                           .toRotationMatrix(),
                   gravity_dir_R_unit_z = unit_z_R_gravity_dir.transpose();

    // TODO(zhangye): check eigen problem
    if (!SolveRotationC(
            X,
            (unit_z_R_gravity_dir * Vector3d(x(0), x(1), 1.0)).head<2>() /
                (unit_z_R_gravity_dir * Vector3d(x(0), x(1), 1.0))(2),
            Y,
            (unit_z_R_gravity_dir * Vector3d(y(0), y(1), 1.0)).head<2>() /
                (unit_z_R_gravity_dir * Vector3d(y(0), y(1), 1.0))(2),
            C_R_W))
        return false;

    for (Matrix3d &R : *C_R_W)
        R = gravity_dir_R_unit_z * R;

    return true;
}

bool SolveTranslation(
    const Vector3d &X, const Vector2d &x, const Vector3d &Y, const Vector2d &y,
    const Matrix3d &C_R_W, Vector3d *C_t_W) {
    // [x 0 -I       [dx      [R * X
    //           *    dy   =
    //  0 y -I]        t]      R * Y]
    // 6x5 lienar system

    // TODO(chen): Use the structure of A to build a more efficient and accurate
    // solver.
    Eigen::Matrix<double, 6, 5> A;
    A << x.homogeneous(), Eigen::Vector3d::Zero(),
        -Eigen::Matrix3d::Identity(), //
        Eigen::Vector3d::Zero(), y.homogeneous(), -Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 1> b;
    b << C_R_W * X, C_R_W * Y;

    Eigen::Matrix<double, 5, 1> solution =
        (A.transpose() * A).ldlt().solve(A.transpose() * b);

    *DCHECK_NOTNULL(C_t_W) = solution.tail<3>();

    if (kEnableDepthCheck && (solution.x() <= 0 || solution.y() <= 0)) {
        VLOG(5) << "Point lies behind camera, dx = " << solution.x()
                << ", dy = " << solution.y();
        return false;
    }

    return true;
}
} // namespace detail

bool P2P(
    const Vector3d &obj_pt_1, const Vector2d &img_pt_1,
    const Vector3d &obj_pt_2, const Vector2d &img_pt_2,
    const Vector3d &gravity_dir, std::vector<Matrix3d> *C_R_W,
    std::vector<Vector3d> *C_t_W) {
    DCHECK_NOTNULL(C_R_W)->clear();
    DCHECK_NOTNULL(C_t_W)->clear();

    if (!detail::SolveRotation(
            obj_pt_1, img_pt_1, obj_pt_2, img_pt_2, gravity_dir, C_R_W))
        return false;

    C_t_W->reserve(C_R_W->size());
    int valid_cnt = 0;
    for (size_t i = 0; i < C_R_W->size(); i++) {
        const Matrix3d &R = (*C_R_W)[i];
        Vector3d t;
        if (!detail::SolveTranslation(
                obj_pt_1, img_pt_1, obj_pt_2, img_pt_2, R, &t))
            continue;

        (*C_R_W)[valid_cnt++] = R;
        C_t_W->push_back(t);
    }

    C_R_W->resize(valid_cnt);
    CHECK_EQ(C_R_W->size(), C_t_W->size());

    return !C_R_W->empty();
}
} // namespace PS
