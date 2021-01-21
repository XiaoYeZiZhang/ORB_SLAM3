#include "PoseSolver.h"
#include <float.h>

namespace Eigen {

class Matrix2x3f : public Matrix<float, 2, 3> {
public:
    inline Matrix2x3f() : Matrix<float, 2, 3>() {
    }
    inline Matrix2x3f(const Matrix<float, 2, 3> &M) : Matrix<float, 2, 3>(M) {
    }
    inline Matrix2x3f(
        const float m00, const float m01, const float m02, const float m10,
        const float m11, const float m12) {
        Matrix2x3f &M = *this;
        M(0, 0) = m00;
        M(0, 1) = m01;
        M(0, 2) = m02;
        M(1, 0) = m10;
        M(1, 1) = m11;
        M(1, 2) = m12;
    }
};

class Matrix2x6f : public Matrix<float, 2, 6> {

public:
    inline Matrix2x6f() : Matrix<float, 2, 6>() {
    }
    inline Matrix2x6f(const Matrix2x3f &M0, const Matrix2x3f &M1) {
        block<2, 3>(0, 0) = M0;
        block<2, 3>(0, 3) = M1;
    }
};

class Matrix3x6f : public Matrix<float, 3, 6> {

public:
    inline Matrix3x6f() : Matrix<float, 3, 6>() {
    }
    inline Matrix3x6f(const Matrix3f &M0, const Matrix3f &M1) {
        block<3, 3>(0, 0) = M0;
        block<3, 3>(0, 3) = M1;
    }
};

class Matrix6f : public Matrix<float, 6, 6> {};

class Vector6f : public Matrix<float, 6, 1> {

public:
    inline Vector6f() : Matrix<float, 6, 1>() {
    }
    inline Vector6f(const Matrix<float, 6, 1> &v) : Matrix<float, 6, 1>(v) {
    }
};

static Matrix3f Exp(const Vector3f &w, const float eps = 1.745329252e-7f) {
    const Vector3f w2 = w.cwiseProduct(w);
    const float th2 = w2.sum(), th = sqrtf(th2);
    if (th < eps) {
        Quaternionf q;
        const float s = 1.0f / sqrtf(th2 + 4.0f);
        q.x() = w.x() * s;
        q.y() = w.y() * s;
        q.z() = w.z() * s;
        q.w() = -(s + s);
        return q.toRotationMatrix();
    }
    const float t1 = sinf(th) / th;
    const float t2 = (1.0f - cosf(th)) / th2;
    const float t3 = 1.0f - t2 * th2;
    const Vector3f t1w = w * t1;
    const Vector3f t2w2 = w2 * t2;
    const float t2wx = t2 * w.x();
    const float t2wxy = t2wx * w.y();
    const float t2wxz = t2wx * w.z();
    const float t2wyz = t2 * w.y() * w.z();
    Matrix3f R;
    R(0, 0) = t3 + t2w2.x();
    R(0, 1) = t2wxy + t1w.z();
    R(0, 2) = t2wxz - t1w.y();
    R(1, 0) = t2wxy - t1w.z();
    R(1, 1) = t3 + t2w2.y();
    R(1, 2) = t2wyz + t1w.x();
    R(2, 0) = t2wxz + t1w.y();
    R(2, 1) = t2wyz - t1w.x();
    R(2, 2) = t3 + t2w2.z();
    return R;
}
} // namespace Eigen

namespace ME {
#define ME_VARIANCE_HUBER 1.809025f // 1.345^2
inline float Weight(const float r2) {
    if (r2 > ME_VARIANCE_HUBER) {
        return sqrtf(ME_VARIANCE_HUBER / r2);
    } else {
        return 1.0f;
    }
}
} // namespace ME

namespace PS {

int REFINE_MAX_ITERATIONS = 100;
float REFINE_MIN_DEPTH = 1.0e-3f;
float REFINE_STD_FEATURE = 1.0f;
float REFINE_STD_FEATURE_ROTATION = 10.0f;
float REFINE_STD_GRAVITY = 1.0f;
float REFINE_CONVERGE_ROTATION = 1.0e-2f;
float REFINE_CONVERGE_TRANSLATION = 1.0e-3f;
float REFINE_MIN_INLIER_RATIO = 0.8f;
int REFINE_DL_MAX_ITERATIONS = 10;
float REFINE_DL_RADIUS_INITIAL = 1.0f; // 1.0^2
float REFINE_DL_RADIUS_MIN = 1.0e-10f; // 0.00001^2
float REFINE_DL_RADIUS_MAX = 1.0e4f;   // 100.0^2
float REFINE_DL_RADIUS_FACTOR_INCREASE = 3.0f;
float REFINE_DL_RADIUS_FACTOR_DECREASE = 0.5f;
float REFINE_DL_GAIN_RATIO_MIN = 0.25f;
float REFINE_DL_GAIN_RATIO_MAX = 0.75f;

bool Refine(
    const float f, const MatchSet3D &M3D, const std::vector<MatchSet2D> &Ms2D,
    Pose *T, const Eigen::Vector3f *g, const float min_translation) {
    Eigen::Vector6f xGN, xGD, xDL;
    float x2GN, x2GD, x2DL, dFa, dFp, delta2, beta, rho;
    bool update, converge;
    std::vector<float> ws3D, Fs3D;
    std::vector<std::vector<float>> ws2D, Fs2D;
    std::vector<bool> rs2D;
    float Fg = 0;
    x2GD = 0;
    x2GN = 0;

    delta2 = REFINE_DL_RADIUS_INITIAL;
    const float s2x = (REFINE_STD_FEATURE * REFINE_STD_FEATURE) / (f * f);
    const float s2xI = 1 / s2x;
    const float s2r =
        (REFINE_STD_FEATURE_ROTATION * REFINE_STD_FEATURE_ROTATION) / (f * f);
    const float s2rI = 1 / s2r;
    const float wr = s2x / s2r;
    const float wg = s2x / (REFINE_STD_GRAVITY * REFINE_STD_GRAVITY);
    const float xrMax =
        REFINE_CONVERGE_ROTATION * 3.14159265358979323846f / 180.0f;
    for (int iIter = 0; iIter < REFINE_MAX_ITERATIONS; ++iIter) {
        Eigen::Matrix6f A;
        Eigen::Vector6f b;
        A.setZero();
        b.setZero();
#ifdef CFG_VERBOSE
        float Se2 = 0.0f;
        int SN = 0;
#endif
        const int N3D = static_cast<int>(M3D.size());
        if (REFINE_DL_MAX_ITERATIONS > 0) {
            ws3D.resize(N3D);
            Fs3D.resize(N3D);
        }
        for (int i = 0; i < N3D; ++i) {
            const Match3D &M = M3D[i];
            const Eigen::Vector3f RX = T->m_R * M.m_X;
            const Eigen::Vector3f TX = RX + T->m_t;
            if (TX.z() < REFINE_MIN_DEPTH) {
                ws3D[i] = 0.0f;
                continue;
            }
            const float zI = 1 / TX.z();
            const Eigen::Vector2f x = Eigen::Vector2f(TX.x(), TX.y()) * zI;
            const Eigen::Vector2f e = x - M.m_x;
            const Eigen::Matrix2x3f Jt(
                zI, 0.0f, -x.x() * zI, 0.0f, zI, -x.y() * zI);
            const Eigen::Matrix2x3f Jr =
                Eigen::Matrix2x3f(Jt * Eigen::SkewSymmetric(RX));
            const Eigen::Matrix2x6f J(Jr, Jt);
            const float e2 = e.squaredNorm();
            const float r2 = e2 * s2xI, w = ME::Weight(r2);
            const auto JTw = J.transpose() * w;
            for (int k = 0; k < 6; ++k) {
                A.block(k, k, 6 - k, 1) +=
                    JTw.block(k, 0, 6 - k, 2) * J.block(0, k, 2, 1);
            }
            b += JTw * e;
            if (REFINE_DL_MAX_ITERATIONS > 0) {
                ws3D[i] = w;
                Fs3D[i] = w * e2;
            }
#ifdef CFG_VERBOSE
            Se2 += e2;
            ++SN;
#endif
        }
#ifdef CFG_VERBOSE
        printf("%d: %f", iIter, SN == 0 ? 0 : sqrtf(Se2 / SN) * f);
#endif
        const int Ns2D = static_cast<int>(Ms2D.size());
        if (REFINE_DL_MAX_ITERATIONS > 0) {
            ws2D.resize(Ns2D);
            Fs2D.resize(Ns2D);
            rs2D.resize(Ns2D);
        }
        for (int i = 0; i < Ns2D; ++i) {
            const MatchSet2D &M2D = Ms2D[i];
            const Pose T21 = *T / M2D.m_T1;
#ifdef CFG_VERBOSE
            float Se2i = 0.0f;
            int SNi = 0;
#endif
            const int N2D = static_cast<int>(M2D.size());
            if (REFINE_DL_MAX_ITERATIONS > 0) {
                ws2D[i].resize(N2D);
                Fs2D[i].resize(N2D);
            }
            if (T21.m_t.norm() <= min_translation) {
                if (REFINE_DL_MAX_ITERATIONS > 0) {
                    rs2D[i] = true;
                }
                for (int j = 0; j < N2D; ++j) {
                    const Match2D &M = M2D[j];
                    const Eigen::Vector3f x1(M.m_x1.x(), M.m_x1.y(), 1.0f);
                    const Eigen::Vector3f Rx1 = T21.m_R * x1;
                    const float zI = 1 / Rx1.z();
                    const Eigen::Vector2f x2 =
                        Eigen::Vector2f(Rx1.x(), Rx1.y()) * zI;
                    const Eigen::Vector2f e = x2 - M.m_x2;
                    const Eigen::Matrix2x3f Jp(
                        zI, 0.0f, -x2.x() * zI, 0.0f, zI, -x2.y() * zI);
                    const Eigen::Matrix2x3f J =
                        Eigen::Matrix2x3f(Jp * Eigen::SkewSymmetric(Rx1));
                    const float e2 = e.squaredNorm();
                    const float r2 = e2 * s2rI, w = wr * ME::Weight(r2);
                    const auto JTw = J.transpose() * w;
                    for (int k = 0; k < 3; ++k) {
                        A.block(k, k, 3 - k, 1) +=
                            JTw.block(k, 0, 3 - k, 3) * J.block(0, k, 3, 1);
                    }
                    b.block<3, 1>(0, 0) += JTw * e;
                    if (REFINE_DL_MAX_ITERATIONS > 0) {
                        ws2D[i][j] = w;
                        Fs2D[i][j] = w * e2;
                    }
#ifdef CFG_VERBOSE
                    Se2 += e2;
                    ++SN;
#endif
                }
            } else {
                if (REFINE_DL_MAX_ITERATIONS > 0) {
                    rs2D[i] = false;
                }
                const Eigen::Matrix3f S1 = Eigen::SkewSymmetric(T21.m_t);
                const Eigen::Matrix3f S2 =
                    Eigen::SkewSymmetric(T21.m_R * M2D.m_T1.m_t);
                const Eigen::Matrix3f E = S1 * T21.m_R;
                for (int j = 0; j < N2D; ++j) {
                    const Match2D &M = M2D[j];
                    const Eigen::Vector3f x1(M.m_x1.x(), M.m_x1.y(), 1.0f);
                    const Eigen::Vector3f l = E * x1;
                    const Eigen::Matrix3f Sj =
                        Eigen::SkewSymmetric(T21.m_R * x1);
                    const Eigen::Matrix3f Jlr = S1 * Sj + Sj * S2;
                    const Eigen::Matrix3f Jlt = -Sj;
                    const float s2I = l.x() * l.x() + l.y() * l.y();
                    if (s2I < FLT_EPSILON) {
                        if (REFINE_DL_MAX_ITERATIONS > 0) {
                            ws2D[i][j] = 0.0f;
                        }
                        continue;
                    }
                    const Eigen::Vector3f x2(M.m_x2.x(), M.m_x2.y(), 1.0f);
                    const float s2 = 1 / s2I, s = sqrtf(s2), s3 = s2 * s;
                    const float Jslx = -s3 * l.x(), Jsly = -s3 * l.y();
                    const float d = x2.dot(l);
                    const float e = s * d;
                    const float Jelx = Jslx * d + s * x2.x();
                    const float Jely = Jsly * d + s * x2.y();
                    const float Jelz = s;
                    const Eigen::Vector6f JT = Eigen::Vector6f(
                        Eigen::Matrix3x6f(Jlr, Jlt).transpose() *
                        Eigen::Vector3f(Jelx, Jely, Jelz));
                    const auto J = JT.transpose();
                    const float e2 = e * e;
                    const float r2 = e2 * s2I, w = ME::Weight(r2);
                    const Eigen::Vector6f JTw = Eigen::Vector6f(JT * w);
                    for (int k = 0; k < 6; ++k) {
                        A.block(k, k, 6 - k, 1) +=
                            JTw.block(k, 0, 6 - k, 1) * J(0, k);
                    }
                    b += JT * e;
                    if (REFINE_DL_MAX_ITERATIONS > 0) {
                        ws2D[i][j] = w;
                        Fs2D[i][j] = w * e2;
                    }
#ifdef CFG_VERBOSE
                    Se2i += e2;
                    ++SNi;
#endif
                }
            }
#ifdef CFG_VERBOSE
            printf(" + %f", SNi == 0 ? 0 : sqrtf(Se2i / SNi) * f);
            Se2 += Se2i;
            SN += SNi;
#endif
        }
#ifdef CFG_VERBOSE
        printf(" = %f\n", sqrtf(Se2 / SN) * f);
#endif
        if (g) {
            const Eigen::Vector3f gp = T->m_R.block<3, 1>(0, 2);
            const Eigen::Vector3f e = gp - *g;
            const Eigen::Matrix3f J = Eigen::SkewSymmetric(gp);
            const auto JTw = J.transpose() * wg;
            for (int k = 0; k < 3; ++k) {
                A.block(k, k, 3 - k, 1) +=
                    JTw.block(k, 0, 3 - k, 3) * J.block(0, k, 3, 1);
            }
            b.block<3, 1>(0, 0) += JTw * e;
            if (REFINE_DL_MAX_ITERATIONS > 0) {
                Fg = wg * e.squaredNorm();
            }
        }

        for (int i = 0; i < 6; ++i) {
            for (int j = i + 1; j < 6; ++j) {
                A(i, j) = A(j, i);
            }
        }
        xGN = Eigen::Vector6f(A.ldlt().solve(-b));
        if (REFINE_DL_MAX_ITERATIONS > 0) {
            x2GN = xGN.squaredNorm();
            x2GD = -1.0f;
        }
        const int nItersDL = std::max(REFINE_DL_MAX_ITERATIONS, 1);
        for (int iIterDL = 0; iIterDL < nItersDL; ++iIterDL) {
            if (REFINE_DL_MAX_ITERATIONS > 0 && x2GN > delta2 &&
                x2GD == -1.0f) {
                // SolveGradientDescent
                const float bl = b.norm();
                if (bl != 0.0f) {
                    xGD = Eigen::Vector6f(b / bl);
                } else {
                    xGD.setZero();
                }
                const float gTAg = xGD.dot(A * xGD);
                const float xl = bl / gTAg;
                xGD *= -xl;
                x2GD = xl * xl;
            }
            // SolveDogLeg
            if (REFINE_DL_MAX_ITERATIONS == 0 || x2GN <= delta2) {
                xDL = xGN;
                x2DL = x2GN;
                beta = 1.0f;
            } else if (x2GD >= delta2) {
                if (delta2 == 0.0f) {
                    xDL = xGD;
                    x2DL = x2GD;
                } else {
                    xDL = Eigen::Vector6f(xGD * sqrtf(delta2 / x2GD));
                    x2DL = delta2;
                }
                beta = 0.0f;
            } else {
                const Eigen::Vector6f dx = Eigen::Vector6f(xGN - xGD);
                const float d = xGD.dot(dx), dx2 = dx.squaredNorm();
                beta = (-d + sqrtf(d * d + (delta2 - x2GD) * dx2)) / dx2;
                xDL = Eigen::Vector6f(xGD + dx * beta);
                x2DL = delta2;
            }
            // UpdateStatesPropose
            const Pose TBkp = *T;
            const Eigen::Vector3f xr = xDL.block<3, 1>(0, 0);
            const Eigen::Vector3f xt = xDL.block<3, 1>(3, 0);
            T->m_R = Eigen::Exp(xr) * T->m_R;
            T->m_t += xt;
            if (REFINE_DL_MAX_ITERATIONS > 0) {
                // ComputeReduction
                dFa = 0.0f;
                for (int i = 0; i < N3D; ++i) {
                    const float w = ws3D[i];
                    if (w == 0.0f) {
                        continue;
                    }
                    const Match3D &M = M3D[i];
                    const Eigen::Vector3f TX = T->m_R * M.m_X + T->m_t;
                    if (fabsf(TX.z()) < FLT_EPSILON) {
                        continue;
                    }
                    const float zI = 1 / TX.z();
                    const Eigen::Vector2f x =
                        Eigen::Vector2f(TX.x(), TX.y()) * zI;
                    const Eigen::Vector2f e = x - M.m_x;
                    const float e2 = e.squaredNorm();
                    const float F = w * e2;
                    dFa += Fs3D[i] - F;
                }
                for (int i = 0; i < Ns2D; ++i) {
                    const MatchSet2D &M2D = Ms2D[i];
                    const Pose T21 = *T / M2D.m_T1;
                    const int N2D = static_cast<int>(M2D.size());
                    const std::vector<float> &_ws2D = ws2D[i], &_Fs2D = Fs2D[i];
                    if (rs2D[i]) {
                        for (int j = 0; j < N2D; ++j) {
                            const Match2D &M = M2D[j];
                            const Eigen::Vector3f x1(
                                M.m_x1.x(), M.m_x1.y(), 1.0f);
                            const Eigen::Vector3f Rx1 = T21.m_R * x1;
                            const float zI = 1 / Rx1.z();
                            const Eigen::Vector2f x2 =
                                Eigen::Vector2f(Rx1.x(), Rx1.y()) * zI;
                            const Eigen::Vector2f e = x2 - M.m_x2;
                            const float e2 = e.squaredNorm();
                            const float F = _ws2D[j] * e2;
                            dFa += _Fs2D[j] - F;
                        }
                    } else {
                        const Eigen::Matrix3f E =
                            Eigen::SkewSymmetric(T21.m_t) * T21.m_R;
                        for (int j = 0; j < N2D; ++j) {
                            const float w = _ws2D[j];
                            if (w == 0.0f) {
                                continue;
                            }
                            const Match2D &M = M2D[j];
                            const Eigen::Vector3f x1(
                                M.m_x1.x(), M.m_x1.y(), 1.0f);
                            const Eigen::Vector3f l = E * x1;
                            const float s2I = l.x() * l.x() + l.y() * l.y();
                            const Eigen::Vector3f x2(
                                M.m_x2.x(), M.m_x2.y(), 1.0f);
                            const float s2 = 1 / s2I, s = sqrtf(s2);
                            const float d = x2.dot(l);
                            const float e = s * d;
                            const float e2 = e * e;
                            const float F = w * e2;
                            dFa += _Fs2D[j] - F;
                        }
                    }
                }
                if (g) {
                    const Eigen::Vector3f gp = T->m_R.block<3, 1>(0, 2);
                    const Eigen::Vector3f e = gp - *g;
                    const float F = wg * e.squaredNorm();
                    dFa += Fg - F;
                }
                dFp = -(A * xDL + b + b).dot(xDL);
                rho = dFa > 0.0f && dFp > 0.0f ? dFa / dFp : -1.0f;
                // UpdateStatesDecide
                if (rho < REFINE_DL_GAIN_RATIO_MIN) {
                    // printf("dFa = %f, dFp = %f\n", dFa, dFp);
                    *T = TBkp;
                    update = false;
                    converge = false;
                    // delta2 *= REFINE_DL_RADIUS_FACTOR_DECREASE;
                    if (delta2 == REFINE_DL_RADIUS_MIN) {
                        break;
                    }
                    delta2 *= REFINE_DL_RADIUS_FACTOR_DECREASE;
                    while (x2GN < delta2) {
                        delta2 *= REFINE_DL_RADIUS_FACTOR_DECREASE;
                    }
                    if (delta2 < REFINE_DL_RADIUS_MIN) {
                        delta2 = REFINE_DL_RADIUS_MIN;
                    }
                    continue;
                } else if (rho > REFINE_DL_GAIN_RATIO_MAX) {
                    delta2 = std::max(
                        delta2, REFINE_DL_RADIUS_FACTOR_INCREASE * x2DL);
                    if (delta2 > REFINE_DL_RADIUS_MAX) {
                        delta2 = REFINE_DL_RADIUS_MAX;
                    }
                }
            }
            update = true;
            converge =
                xr.norm() < xrMax && xt.norm() < REFINE_CONVERGE_TRANSLATION;
            break;
        }
        if (!update || converge) {
            break;
        }
    }

    int SN = 0, SNi = 0;
    const int N3D = static_cast<int>(M3D.size());
    for (int i = 0; i < N3D; ++i) {
        const Match3D &M = M3D[i];
        const Eigen::Vector3f TX = T->m_R * M.m_X + T->m_t;
        if (TX.z() <= 0.0f) {
            continue;
        }
        const Eigen::Vector2f x = Eigen::Vector2f(TX.x(), TX.y()) / TX.z();
        const Eigen::Vector2f e = x - M.m_x;
        const float e2 = e.squaredNorm();
        const float r2 = e2 * s2xI;
        if (r2 < ME_VARIANCE_HUBER) {
            ++SNi;
        }
    }
    SN += N3D;
    const int Ns2D = static_cast<int>(Ms2D.size());
    for (int i = 0; i < Ns2D; ++i) {
        const MatchSet2D &M2D = Ms2D[i];
        const Pose T21 = *T / M2D.m_T1;
        const int N2D = static_cast<int>(M2D.size());
        if (T21.m_t.norm() < min_translation) {
            for (int j = 0; j < N2D; ++j) {
                const Match2D &M = M2D[j];
                const Eigen::Vector3f x1(M.m_x1.x(), M.m_x1.y(), 1.0f);
                const Eigen::Vector3f Rx1 = T21.m_R * x1;
                const float zI = 1 / Rx1.z();
                const Eigen::Vector2f x2 =
                    Eigen::Vector2f(Rx1.x(), Rx1.y()) * zI;
                const Eigen::Vector2f e = x2 - M.m_x2;
                const float e2 = e.squaredNorm();
                const float r2 = e2 * s2rI;
                if (r2 < ME_VARIANCE_HUBER) {
                    ++SNi;
                }
            }
        } else {
            const Eigen::Matrix3f E = Eigen::SkewSymmetric(T21.m_t) * T21.m_R;
            for (int j = 0; j < N2D; ++j) {
                const Match2D &M = M2D[j];
                const Eigen::Vector3f x1(M.m_x1.x(), M.m_x1.y(), 1.0f);
                const Eigen::Vector3f l = E * x1;
                const float s2I = l.x() * l.x() + l.y() * l.y();
                if (s2I < FLT_EPSILON) {
                    continue;
                }
                const Eigen::Vector3f x2(M.m_x2.x(), M.m_x2.y(), 1.0f);
                const float s2 = 1 / s2I, s = sqrtf(s2);
                const float d = x2.dot(l);
                const float e = s * d;
                const float e2 = e * e;
                const float r2 = e2 * s2I;
                if (r2 < ME_VARIANCE_HUBER) {
                    ++SNi;
                }
            }
        }
        SN += N2D;
    }
    const float r = static_cast<float>(SNi) / SN;
#ifdef CFG_VERBOSE
    printf("r = %f%%\n", r * 100);
#endif
    return r >= REFINE_MIN_INLIER_RATIO;
}
} // namespace PS
