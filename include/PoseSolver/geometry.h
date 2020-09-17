#ifndef POSE_SOLVER_GEOMETRY_H_
#define POSE_SOLVER_GEOMETRY_H_

#include <cmath>

#include <Eigen/Dense>
namespace PS {

constexpr double kPi = 3.1415927;

constexpr float Rad2Deg(float rad) {
    return 180 / static_cast<float>(kPi) * rad;
}

constexpr float Deg2Rad(float deg) {
    return static_cast<float>(kPi) / 180 * deg;
}

constexpr double Rad2Deg(double rad) {
    return 180 / kPi * rad;
}

constexpr double Deg2Rad(double deg) {
    return kPi / 180 * deg;
}

// Find cloest point on the line to pt.
template <typename D1, typename D2>
inline Eigen::Matrix<typename D1::Scalar, 2, 1> CloestPointOnLine(
    const Eigen::MatrixBase<D1> &pt, const Eigen::MatrixBase<D2> &line) {
    using Scalar = typename D1::Scalar;
    // line = ax + by + z = 0
    // err = -(ax + by + z) / sqrt(a^2 + b^2) * (a, b) / sqrt(a^2 + b^2)
    //     = -(ax + by + z) * (a, b) / (a^2 + b^2)
    Eigen::Matrix<Scalar, 2, 1> normal = line.template head<2>();
    normal /= normal.squaredNorm();
    if (!normal.allFinite()) {
        // ideal line
        const Eigen::Matrix<Scalar, 2, 1> kReallyBigErr(1E6, 1E6);
        return pt + kReallyBigErr;
    }
    auto err = -line.dot(pt.homogeneous()) * normal;
    return pt + err;
}

// Find cloest point on the line to pt.
template <typename D1, typename D2>
inline typename D1::Scalar DistanceToLine(
    const Eigen::MatrixBase<D2> &pt, const Eigen::MatrixBase<D1> &line) {
    return (CloestPointOnLine(pt, line) - pt).norm();
}

template <typename D>
inline void ProperRotationMatrixCheck(const Eigen::MatrixBase<D> &R) {
    using Scalar = typename D::Scalar;
    CHECK_EQ(R.rows(), 3);
    CHECK_EQ(R.cols(), 3);
    using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
    Mat3 diff_to_identity = R.transpose() * R - Mat3::Identity();
    CHECK_LT(diff_to_identity.template lpNorm<Eigen::Infinity>(), 1E-6);
    CHECK_LT(std::abs(R.determinant() - 1), 1E-6);
}

template <typename D1, typename D2>
inline typename D1::Scalar AngleBetweenTwoDirInRad(
    const Eigen::MatrixBase<D1> &dir1, const Eigen::MatrixBase<D2> &dir2) {
    using Scalar = typename D1::Scalar;
    Scalar dot = dir1.dot(dir2) / (dir1.norm() * dir2.norm());
    CHECK(std::isfinite(dot));
    // Clamping to avoid numeric issues.
    return std::acos(std::min(Scalar(1), std::max(dot, Scalar(-1))));
}

template <typename D1, typename D2>
inline typename D1::Scalar AngleBetweenTwoDirInDeg(
    const Eigen::MatrixBase<D1> &dir1, const Eigen::MatrixBase<D2> &dir2) {
    return Rad2Deg(AngleBetweenTwoDirInRad(dir1, dir2));
}
} // namespace PS
#endif // POSE_SOLVER_GEOMETRY_H_
