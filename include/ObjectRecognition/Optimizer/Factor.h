//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_FACTOR_H
#define ORB_SLAM3_FACTOR_H
#include "ceres-solver/include/ceres/local_parameterization.h"
#include "Utility/Utility.h"
#include "ceres/sized_cost_function.h"
#include "Utility/Camera.h"

namespace ObjRecognition {

class ProjectionFactorXYZLeftJacobians
    : public ceres::SizedCostFunction<2, 7, 3> {
public:
    ProjectionFactorXYZLeftJacobians(const Eigen::Vector3d &_pt_norm)
        : pt_norm(_pt_norm){};
    virtual bool Evaluate(
        double const *const *parameters, double *residuals,
        double **jacobians) const {

        /// get pararameter of IMU pose
        Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Q(
            parameters[0][6], parameters[0][3], parameters[0][4],
            parameters[0][5]);

        /// get parameter of 3d point
        Eigen::Vector3d P_w(
            parameters[1][0], parameters[1][1], parameters[1][2]);

        /// project point
        Eigen::Matrix3d RIC = Eigen::Matrix3d::Identity();
        Eigen::Vector3d TIC = Eigen::Vector3d::Zero();
        Eigen::Vector3d pts_imu = Q.inverse() * (P_w - P);
        Eigen::Vector3d pts_camera = RIC.inverse() * (pts_imu - TIC);

        /// compute residual
        double dep = pts_camera.z();
        double dep_inv = 1 / dep;
        double dep_inv2 = dep_inv * dep_inv;
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = (pts_camera / dep).head<2>() -
                   pt_norm.head<2>(); /// residual in normal plane

        Eigen::Vector2d PROJECTION_SQRT_INFO;
        const Eigen::Matrix3d K =
            ObjRecognition::CameraIntrinsic::GetInstance().GetEigenK();
        double FX = K(0, 0);
        PROJECTION_SQRT_INFO << FX / 1.5, FX / 1.5;

        residual =
            PROJECTION_SQRT_INFO.asDiagonal() * residual; /// residual in pixels
        /// TODO(wangnan) compute jacobians
        if (jacobians) {
            Eigen::Matrix3d R = Q.toRotationMatrix();
            Eigen::Matrix3d ric = RIC;
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1. / dep, 0.0, -pts_camera(0) / (dep * dep), 0.0,
                1. / dep, -pts_camera(1) / (dep * dep);

            reduce = PROJECTION_SQRT_INFO.asDiagonal() * reduce;
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
                    jacobian_imu(jacobians[0]);

                Eigen::Matrix<double, 3, 6> jaco_i;
                jaco_i.leftCols<3>() = -ric.transpose() * R.transpose();
                jaco_i.rightCols<3>() =
                    ric.transpose() * R.transpose() * SkewSymmetric(P_w - P);

                jacobian_imu.leftCols<6>() = reduce * jaco_i;
                jacobian_imu.rightCols<1>().setZero();
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
                    jacobian_feature(jacobians[1]);

                jacobian_feature = reduce * ric.transpose() * R.transpose();
            }
        }
        return true;
    }

    Eigen::Vector3d pt_norm;
};

class PoseLocalParameterizationLeftJacobians
    : public ceres::LocalParameterization {
    virtual bool
    Plus(const double *x, const double *delta, double *x_plus_delta) const {
        Eigen::Map<const Eigen::Vector3d> _p(x);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

        Eigen::Map<const Eigen::Vector3d> dp(delta);
        Eigen::Quaterniond dq =
            deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (dq * _q).normalized();
        q = q.w() >= 0.0 ? q
                         : Eigen::Quaterniond(-q.w(), -q.x(), -q.y(), -q.z());

        return true;
    }

    virtual bool ComputeJacobian(const double *x, double *jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();
        return true;
    }

    virtual int GlobalSize() const {
        return 7;
    };

    virtual int LocalSize() const {
        return 6;
    };
};
} // namespace ObjRecognition
#endif // OBJECTRECOGNITION_FACTOR_H