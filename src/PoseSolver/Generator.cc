#include "Generator.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <random>
#include <utility>

#include <glog/logging.h>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "essential3.h"
#include "geometry.h"
#include "p2p.h"

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

using Eigen::Matrix3f;
using Eigen::Vector2f;
using Eigen::Vector3f;

static constexpr double kMaxReprojErrDeg = 10;

namespace PS {

template <typename It, typename T>
void ReplaceWith(It beg, It end, T to_be_replaced, T replaced_with) {
    for (; beg != end; ++beg)
        if (*beg == to_be_replaced)
            *beg = replaced_with;
}

// Randomly sampling a numbers in [0, limit)
template <typename RNG> int RandomSampling(int limit, RNG &gen) {
    DCHECK_GT(limit, 0);
    return std::uniform_int_distribution<int>(0, limit - 1)(gen);
}

// Unrepeated randomly sampling N numbers in [0, limit)
template <int N, typename RNG>
std::array<int, N> UnrepeatedRandomSampling(int limit, RNG &gen) {
    DCHECK_GE(limit, N);

    std::array<int, N> result;

    if (N == limit) {
        // Fast path.
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    for (int i = 0; i < N; i++)
        result[i] = std::uniform_int_distribution<int>(0, limit - 1 - i)(gen);

    for (int i = 1; i < N; i++)
        ReplaceWith(
            std::next(result.begin(), i), result.end(), result[i - 1],
            limit - i);

    return result;
}

double HybridGenerator::SuccProb(
    double inlier_ratio_3d, double inlier_ratio_2d) const {
    return scale_free_gen_->SuccProb(inlier_ratio_3d, inlier_ratio_2d) *
           scale_only_gen_->SuccProb(inlier_ratio_3d, inlier_ratio_2d);
}

bool HybridGenerator::RunOnce(Pose *C_T_W) {
    if (scale_free_gen_support_matches_.empty()) {
        LOG(WARNING) << "Empty support matches.";
        return false;
    }

    const int ref_frame = scale_free_gen_support_matches_[next_ref_index_];
    next_ref_index_ =
        (next_ref_index_ + 1) % scale_free_gen_support_matches_.size();

    Pose cur_T_ref;
    scale_free_gen_->set_ref_frame_index(ref_frame);
    if (!scale_free_gen_->RunOnce(&cur_T_ref))
        return false;

    scale_only_gen_->set_relative_pose(ref_frame, cur_T_ref);
    return scale_only_gen_->RunOnce(C_T_W);
}

bool HybridGenerator::HasEnoughSupport() const {
    return scale_free_gen_->HasEnoughSupport() &&
           scale_only_gen_->HasEnoughSupport();
}

double PnPGenerator::SuccProb(
    double inlier_ratio_3d, double /* inlier_ratio_2d */) const {
    return std::pow(inlier_ratio_3d, 4);
}

bool PnPGenerator::RunOnce(Pose *C_T_W) {
    CHECK_NOTNULL(C_T_W);

    constexpr int kMinSamples = 4;

    if (matches_.size() < static_cast<std::size_t>(kMinSamples)) {
        VLOG(5) << "No enough matches to solve PnP.";
        return false;
    }

    std::array<int, kMinSamples> sample_indices =
        UnrepeatedRandomSampling<kMinSamples>(
            static_cast<int>(matches_.size()), rand_eng_);

    std::vector<cv::Point3d> obj_pts;
    std::vector<cv::Point2d> img_pts;
    obj_pts.reserve(kMinSamples);
    img_pts.reserve(kMinSamples);
    for (int i : sample_indices) {
        const auto m = matches_[i];
        obj_pts.emplace_back(m.m_X.x(), m.m_X.y(), m.m_X.z());
        img_pts.emplace_back(m.m_x.x(), m.m_x.y());
    }

    cv::Mat_<double> rvec, tvec;

    constexpr bool kUseExtrinsicGuess = false;
    constexpr int kFlags = cv::SOLVEPNP_P3P;

    // NOTE: OpenCV solvePnP has problem with float input type, so use double
    // here. (MacOS, 4.1.0)
    if (!cv::solvePnP(
            obj_pts, img_pts, cv::Mat_<double>::eye(3, 3), cv::noArray(), rvec,
            tvec, kUseExtrinsicGuess, kFlags)) {
        VLOG(10) << "solvePnP failed.";
        return false;
    }
    cv::Mat C_R_W_cv;
    cv::Rodrigues(rvec, C_R_W_cv);

    CHECK_EQ(rvec.type(), CV_64FC1);
    CHECK_EQ(tvec.type(), CV_64FC1);
    CHECK_EQ(C_R_W_cv.type(), CV_64FC1);

    Eigen::Matrix3d tmp_R;
    Eigen::Vector3d tmp_t;
    cv::cv2eigen(C_R_W_cv, tmp_R);
    cv::cv2eigen(tvec, tmp_t);

    C_T_W->m_R = tmp_R.cast<float>();
    C_T_W->m_t = tmp_t.cast<float>();

    return true;
}

bool PnPGenerator::HasEnoughSupport() const {
    return matches_.size() >= 4u;
}

double P2PGenerator::SuccProb(
    double inlier_ratio_3d, double /* inlier_ratio_2d */) const {
    return std::pow(inlier_ratio_3d, 3);
}

bool P2PGenerator::RunOnce(Pose *C_T_W) {
    CHECK_NOTNULL(C_T_W);

    constexpr int kMinSamples = 3;

    if (matches_.size() < static_cast<std::size_t>(kMinSamples)) {
        VLOG(5) << "No enough matches to solve P2P.";
        return false;
    }

    std::array<int, kMinSamples> sample_indices =
        UnrepeatedRandomSampling<kMinSamples>(
            static_cast<int>(matches_.size()), rand_eng_);

    std::vector<Vector3d> obj_pts;
    std::vector<Vector2d> img_pts;
    obj_pts.reserve(kMinSamples);
    img_pts.reserve(kMinSamples);
    for (int i : sample_indices) {
        const auto m = matches_[i];
        obj_pts.emplace_back(m.m_X.x(), m.m_X.y(), m.m_X.z());
        img_pts.emplace_back(m.m_x.x(), m.m_x.y());
    }

    std::vector<Matrix3d> C_R_W;
    std::vector<Vector3d> C_t_W;
    if (!P2P(
            obj_pts[0], img_pts[0], obj_pts[1], img_pts[1],
            get_gravity_dir().cast<double>(), &C_R_W, &C_t_W)) {
        VLOG(5) << "P2P solver failed";
        return false;
    }

    CHECK(!C_R_W.empty());
    // P2P gives at most 2 solutions.
    CHECK_LE(C_R_W.size(), 2u);

    if (C_R_W.size() == 1u) {
        // Only one solution got.
        VLOG(10) << "P2P: only find one soultion";

        C_T_W->m_R = C_R_W.front().cast<float>();
        C_T_W->m_t = C_t_W.front().cast<float>();
        return true;
    }

    // Use the 3rd point as validation data point to do selection.
    const Vector3d &obj_pt_val = obj_pts[2];
    const Vector2d &img_pt_val = img_pts[2];
    double reproj_errs[2] = {std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max()};
    for (size_t i = 0; i < 2; i++) {
        Vector3d obj_pt_local = C_R_W[i] * obj_pt_val + C_t_W[i];
        if (obj_pt_local.z() < 0)
            continue;

        reproj_errs[i] =
            AngleBetweenTwoDirInDeg(obj_pt_local, img_pt_val.homogeneous());
    }

    // Make sure the first is the best.
    if (reproj_errs[0] > reproj_errs[1]) {
        std::swap(C_R_W.front(), C_R_W.back());
        std::swap(C_t_W.front(), C_t_W.back());
        std::swap(reproj_errs[0], reproj_errs[1]);
    }

    if (reproj_errs[0] == std::numeric_limits<double>::max()) {
        VLOG(10)
            << "3rd point projected behind camera for both soluitions, reject.";
        return false;
    }

    if (reproj_errs[0] > kMaxReprojErrDeg) {
        VLOG(10) << "3rd point reprojection error: " << reproj_errs[0]
                 << ", reject.";
        return false;
    }

    // TODO(chen): Ratio test?

    C_T_W->m_R = C_R_W.front().cast<float>();
    C_T_W->m_t = C_t_W.front().cast<float>();

    return true;
}

bool P2PGenerator::HasEnoughSupport() const {
    return matches_.size() >= 3u;
}

Vector2f EssentialGenerator::SolveDepth(
    const Vector2f &pt_ref, const Vector2f &pt_cur, const Matrix3f &cur_R_ref,
    const Vector3f &cur_t_ref) {
    // d_cur * x_cur = R * d_ref * x_ref + t
    // => [x_cur, -R * x_ref] * [d_cur; d_ref] = t  -- a 3x2 linear system
    Eigen::Matrix<float, 3, 2> A;
    A.col(0) = pt_cur.homogeneous();
    A.col(1) = -cur_R_ref * pt_ref.homogeneous();

    return (A.transpose() * A).ldlt().solve(A.transpose() * cur_t_ref);
}

double Essential3Generator::SuccProb(
    double /* inlier_ratio_3d */, double inlier_ratio_2d) const {
    return std::pow(inlier_ratio_2d, kResolveAmbiguityBeforeSolveScale ? 4 : 3);
}

bool Essential3Generator::RunOnce(Pose *cur_T_ref) {
    CHECK_NOTNULL(cur_T_ref);

    const int ref_idx = get_ref_frame_index();
    CHECK_GE(ref_idx, 0);
    CHECK_LT(ref_idx, static_cast<int>(matches_.size()));

    const MatchSet2D &matches = matches_[ref_idx];

    constexpr int kMinSamples =
        4 /* one more sample to resolve potential ambiguity */;

    if (static_cast<int>(matches.size()) < kMinSamples) {
        VLOG(10) << "No enough points to solve essential matrix.";
        return false;
    }

    // Solve Essential matrix
    std::vector<Vector2d> pts_ref, pts_cur;
    pts_ref.reserve(kMinSamples);
    pts_cur.reserve(kMinSamples);
    // TODO(chen): We have a diamond problem here.
    std::array<int, kMinSamples> samples_indices =
        UnrepeatedRandomSampling<kMinSamples>(
            static_cast<int>(matches.size()), GravityGenerator::rand_eng_);
    for (int i : samples_indices) {
        const auto &m = matches[i];
        pts_ref.emplace_back(m.m_x1.x(), m.m_x1.y());
        pts_cur.emplace_back(m.m_x2.x(), m.m_x2.y());
    }

    // Validation match.
    const Vector2d pt_ref_val = pts_ref.back(), pt_cur_val = pts_cur.back();
    pts_ref.pop_back();
    pts_cur.pop_back();

    std::vector<Matrix3d> cur_R_ref;
    std::vector<Vector3d> cur_t_ref;
    if (!Essential3(
            pts_ref, pts_cur, matches.m_T1.m_R.cast<double>(),
            get_gravity_dir().cast<double>(), &cur_R_ref, &cur_t_ref)) {
        VLOG(5) << "Essential3 failed";
        return false;
    }

    CHECK(!cur_R_ref.empty());

    if (kResolveAmbiguityBeforeSolveScale) {
        ResolveAmbiguity(pt_ref_val, pt_cur_val, &cur_R_ref, &cur_t_ref);
        if (cur_R_ref.empty()) {
            VLOG(5) << "No solution passed the validation.";
            return false;
        }
    }

    std::vector<std::pair<Matrix3f, Vector3f>> proposals;
    for (size_t i = 0; i < cur_R_ref.size(); i++)
        proposals.emplace_back(
            cur_R_ref[i].cast<float>(), cur_t_ref[i].cast<float>());

    constexpr float kMinDepth = 1E-6;
    const Vector2f pt0_ref = pts_ref.front().cast<float>(),
                   pt0_cur = pts_cur.front().cast<float>();

    auto it = std::find_if(
        proposals.begin(), proposals.end(),
        [&](const std::pair<Matrix3f, Vector3f> &p) {
            return SolveDepth(pt0_ref, pt0_cur, p.first, p.second).minCoeff() >
                   kMinDepth;
        });

    if (it == proposals.end()) {
        VLOG(5) << "No valid proposals.";
        return false;
    }

    if (std::find_if(
            std::next(it), proposals.end(),
            [&](const std::pair<Matrix3f, Vector3f> &p) {
                return SolveDepth(pt0_ref, pt0_cur, p.first, p.second)
                           .minCoeff() > kMinDepth;
            }) != proposals.end()) {
        VLOG(5) << "More than one valid proposals";
        return false;
    }

    std::tie(cur_T_ref->m_R, cur_T_ref->m_t) = *it;

    return true;
}

std::vector<int> Essential3Generator::GetSupport2DMatches() const {
    std::vector<int> indices;
    for (size_t i = 0; i < matches_.size(); i++) {
        if (matches_[i].size() >= 4u)
            indices.push_back(i);
    }
    return indices;
}

bool Essential3Generator::HasEnoughSupport() const {
    return std::any_of(
        matches_.begin(), matches_.end(), [](const MatchSet2D &ms) {
            return ms.size() >= (kResolveAmbiguityBeforeSolveScale ? 4u : 3u);
        });
}

void Essential3Generator::ResolveAmbiguity(
    const Eigen::Vector2d &ref, const Eigen::Vector2d &cur,
    std::vector<Matrix3d> *cur_R_ref, std::vector<Vector3d> *cur_t_ref) const {
    if (cur_R_ref->size() <= 1u) {
        VLOG(10) << "Essential3 solved " << cur_R_ref->size()
                 << " solutions, no ambiguity";
        return;
    }

    VLOG(10) << "Essential3 solved " << cur_R_ref->size() << " solutions.";
    int valid_cnt = 0;
    for (size_t i = 0; i < cur_R_ref->size(); i++) {
        Vector3d epipolar_line =
            cur_t_ref->at(i).cross(cur_R_ref->at(i) * ref.homogeneous());
        // Pure rotation case is auto handled unless cur_t_ref.norm() is
        // numerically tiny, which will not be our case.
        double reproj_err = AngleBetweenTwoDirInDeg(
            CloestPointOnLine(cur, epipolar_line).homogeneous(),
            cur.homogeneous());
        if (reproj_err > kMaxReprojErrDeg)
            continue;

        cur_R_ref->at(valid_cnt) = cur_R_ref->at(i);
        cur_t_ref->at(valid_cnt) = cur_t_ref->at(i);
        valid_cnt++;
    }

    cur_R_ref->resize(valid_cnt);
    cur_t_ref->resize(valid_cnt);

    VLOG(5) << "Essential3 remains " << cur_R_ref->size()
            << " solutions after ambiguity resolution.";
}

double Essential5Generator::SuccProb(
    double /* inlier_ratio_3d */, double inlier_ratio_2d) const {
    return std::pow(inlier_ratio_2d, 6);
}

bool Essential5Generator::RunOnce(Pose *cur_T_ref) {
    CHECK_NOTNULL(cur_T_ref);

    const int ref_idx = get_ref_frame_index();
    CHECK_GE(ref_idx, 0);
    CHECK_LT(ref_idx, static_cast<int>(matches_.size()));

    const MatchSet2D &matches = matches_[ref_idx];

    constexpr int kMinSamples =
        6 /* one more sample to resolve potential ambiguity */;

    if (static_cast<int>(matches.size()) < kMinSamples) {
        VLOG(5) << "No enough matches to solve essential matrix.";
        return false;
    }

    // Solve Essential matrix
    std::vector<cv::Point2f> pts_ref, pts_cur;
    pts_ref.reserve(kMinSamples);
    pts_cur.reserve(kMinSamples);
    std::array<int, kMinSamples> samples_indices =
        UnrepeatedRandomSampling<kMinSamples>(
            static_cast<int>(matches.size()), rand_eng_);
    for (int i : samples_indices) {
        const auto &m = matches[i];
        pts_ref.emplace_back(m.m_x1.x(), m.m_x1.y());
        pts_cur.emplace_back(m.m_x2.x(), m.m_x2.y());
    }

    cv::Mat_<float> cur_E_ref_cv = cv::findEssentialMat(
        pts_ref, pts_cur, 1, /* focal length */
        cv::Point2d(0, 0),   /* principal point */
        cv::LMEDS /* method */);
    CHECK_EQ(cur_E_ref_cv.type(), CV_32FC1);
    if (cur_E_ref_cv.rows != 3) {
        LOG(INFO) << "More than on essentail matrix solution";
        return false;
    }

    cv::Mat_<float> cur_R_ref_1_cv, cur_R_ref_2_cv, cur_t_ref_cv;
    cv::decomposeEssentialMat(
        cur_E_ref_cv, cur_R_ref_1_cv, cur_R_ref_2_cv, cur_t_ref_cv);

    CHECK_EQ(cur_R_ref_1_cv.type(), CV_32FC1);
    CHECK_EQ(cur_R_ref_2_cv.type(), CV_32FC1);
    CHECK_EQ(cur_t_ref_cv.type(), CV_32FC1);

    Matrix3f cur_R_ref_1, cur_R_ref_2;
    Vector3f cur_t_ref;

    cv::cv2eigen(cur_R_ref_1_cv, cur_R_ref_1);
    cv::cv2eigen(cur_R_ref_2_cv, cur_R_ref_2);
    cv::cv2eigen(cur_t_ref_cv, cur_t_ref);

    // Up to now, we have 4 possible solutions
    std::array<std::pair<Matrix3f, Vector3f>, 4> proposals{
        std::pair<Matrix3f, Vector3f>{cur_R_ref_1, cur_t_ref},
        std::pair<Matrix3f, Vector3f>{cur_R_ref_1, -cur_t_ref},
        std::pair<Matrix3f, Vector3f>{cur_R_ref_2, cur_t_ref},
        std::pair<Matrix3f, Vector3f>{cur_R_ref_2, -cur_t_ref}};
    constexpr float kMinDepth = 1E-6;
    const Vector2f pt0_ref(pts_ref[0].x, pts_ref[0].y),
        pt0_cur(pts_cur[0].x, pts_cur[0].y);
    auto it = std::find_if(
        proposals.begin(), proposals.end(),
        [&](const std::pair<Matrix3f, Vector3f> &p) {
            return SolveDepth(pt0_ref, pt0_cur, p.first, p.second).minCoeff() >
                   kMinDepth;
        });

    if (it == proposals.end()) {
        LOG(ERROR) << "No valid proposals.";
        return false;
    }

    if (std::find_if(
            std::next(it), proposals.end(),
            [&](const std::pair<Matrix3f, Vector3f> &p) {
                return SolveDepth(pt0_ref, pt0_cur, p.first, p.second)
                           .minCoeff() > kMinDepth;
            }) != proposals.end()) {
        VLOG(5) << "More than one valid proposals";
        return false;
    }

    std::tie(cur_T_ref->m_R, cur_T_ref->m_t) = *it;

    return true;
}

std::vector<int> Essential5Generator::GetSupport2DMatches() const {
    std::vector<int> indices;
    for (size_t i = 0; i < matches_.size(); i++) {
        if (matches_[i].size() >= 6u)
            indices.push_back(i);
    }
    return indices;
}

bool Essential5Generator::HasEnoughSupport() const {
    return std::any_of(
        matches_.begin(), matches_.end(),
        [](const MatchSet2D &ms) { return ms.size() >= 6u; });
}

double ScaleSolver2D::SuccProb(
    double /* inlier_ratio_3d */, double inlier_ratio_2d) const {
    return inlier_ratio_2d;
}

bool ScaleSolver2D::RunOnce(Pose *C_T_W) {
    CHECK_NOTNULL(C_T_W);

    int ref_idx;
    Pose cur_T_ref;
    get_relative_pose(&ref_idx, &cur_T_ref);

    CHECK_GE(ref_idx, 0);
    CHECK_LT(ref_idx, matches_.size());

    if (matches_.size() < 2u) {
        VLOG(10) << "No enough match set to solve pose.";
        return false;
    }

    int another_ref_idx =
        RandomSampling(static_cast<int>(matches_.size()) - 1, rand_eng_);
    if (another_ref_idx == ref_idx)
        another_ref_idx = static_cast<int>(matches_.size()) - 1;

    const MatchSet2D &matches = matches_[ref_idx],
                     &another_matches = matches_[another_ref_idx];

    constexpr int kMinSamplesAnother = 1;

    if (static_cast<int>(another_matches.size()) < kMinSamplesAnother) {
        VLOG(10) << "No enough points to solve scale.";
        return false;
    }

    int another_sample_idx =
        RandomSampling(static_cast<int>(another_matches.size()), rand_eng_);

    const Pose &ref0_T_W = matches.m_T1, &ref1_T_W = another_matches.m_T1;
    const Pose ref0_T_ref1(
        ref0_T_W.m_R * ref1_T_W.m_R.transpose(),
        ref0_T_W.m_t - ref0_T_W.m_R * ref1_T_W.m_R.transpose() * ref1_T_W.m_t);

    float t_scale = 0;
    if (!SolveTranslationalScale(
            another_matches[another_sample_idx].m_x1,
            another_matches[another_sample_idx].m_x2, cur_T_ref.m_R,
            cur_T_ref.m_t, ref0_T_ref1.m_R, ref0_T_ref1.m_t, &t_scale))
        return false;

    cur_T_ref.m_t *= t_scale;

    C_T_W->m_R = cur_T_ref.m_R * matches.m_T1.m_R;
    C_T_W->m_t = cur_T_ref.m_R * matches.m_T1.m_t + cur_T_ref.m_t;

    return true;
}

bool ScaleSolver2D::HasEnoughSupport() const {
    // TODO(chen)
    return std::count_if(
               matches_.begin(), matches_.end(),
               [](const MatchSet2D &ms) { return !ms.empty(); }) >= 2;
}

bool ScaleSolver2D::SolveTranslationalScale(
    const Vector2f &pt_ref1 /* 2D point in ref1 frame */,
    const Vector2f &pt_cur /* 2D point in cur frame */,
    const Matrix3f &cur_R_ref0, const Vector3f &cur_t_ref0 /* unknown scale */,
    const Matrix3f &ref0_R_ref1, const Vector3f &ref0_t_ref1, float *scale) {
    // If the essential matrix between cur_ref0 and ref1_ref0 are almost equal,
    // abort the solving.
    // TODO(chen): Maybe we can choose another ref1
    constexpr float kPi = 3.141592654, kMinRotDeg = 1,
                    kMinRotRad = kMinRotDeg * kPi / 180,
                    kMinTranslationalDirAngleDeg = 5,
                    kMinTranslationalDirAngleRad =
                        kMinTranslationalDirAngleDeg * kPi / 180,
                    kMinTranslatinMeter = 0.02 /* 2cm */;

    if (ref0_t_ref1.norm() < kMinTranslatinMeter) {
        VLOG(5) << "Pure rotation between ref0 and ref1 detected, "
                   "translational size = "
                << ref0_t_ref1.norm() << " meter, abort scale solving.";
        return false;
    }

    const float cur_r_ref1 = std::abs(
                    Eigen::AngleAxisf(cur_R_ref0 * ref0_R_ref1).angle()),
                cur_t_ref0_angle_ref1_t_ref0 = std::acos(std::min(
                    std::abs(cur_t_ref0.normalized().dot(
                        (ref0_R_ref1.transpose() * ref0_t_ref1).normalized())),
                    1.f));
    if (cur_r_ref1 < kMinRotRad &&
        cur_t_ref0_angle_ref1_t_ref0 < kMinTranslationalDirAngleRad) {
        VLOG(5) << "Degenerated case detected, scale free relative pose "
                   "cur_T_ref0, ref1_T_ref0 are "
                   "almost identical, cur_r_ref1 = "
                << cur_r_ref1 << " rad, cur_t_ref0_angle_ref1_t_ref0 = "
                << cur_t_ref0_angle_ref1_t_ref0 << " rad.";
        return false;
    }

    // d_ref1 -- pt_ref1 depth
    // d_cur -- pt_cur depth
    // s -- cur_t_ref0 scale
    //
    // Equaling pt_ref1's 3D point and pt_cur's 3D in ref0
    //
    // clang-format off
  // ref0_R_ref1 * (d_ref1 * pt_ref1) + ref0_t_ref1 = inv(cur_R_ref0) * (d_cur * pt_cur) - inv(cur_R_ref0) * s * cur_t_ref0
  // => [ref0_R_ref1 * pt_ref1, inv(cur_R_ref0) * pt_cur, inv(cur_R_ref0) * cur_t_ref0] * [d_ref1; -d_cur; s] = -ref0_t_ref1  -- a 3x3 linear system
    // clang-format on

    Eigen::Matrix3f A;
    A.col(0) = ref0_R_ref1 * pt_ref1.homogeneous();
    A.col(1) = cur_R_ref0.transpose() * pt_cur.homogeneous();
    A.col(2) = cur_R_ref0.transpose() * cur_t_ref0;

    Eigen::Vector3f sol = A.lu().solve(-ref0_t_ref1);
    if (sol.x() < 0 || sol.y() > 0 || sol.z() < 0) {
        VLOG(5) << "Negative depth/scale, solution = " << sol.transpose();
        return false;
    }

    *DCHECK_NOTNULL(scale) = sol.z();
    return true;
}

double ScaleSolver3D::SuccProb(
    double inlier_ratio_3d, double /* inlier_ratio_2d */) const {
    return inlier_ratio_3d;
}

bool ScaleSolver3D::RunOnce(Pose *C_T_W) {
    CHECK_NOTNULL(C_T_W);

    int ref_idx;
    Pose cur_T_ref;
    get_relative_pose(&ref_idx, &cur_T_ref);

    CHECK_GE(ref_idx, 0);
    CHECK_LT(ref_idx, data_.matches_2d.size());

    if (data_.matches_3d.empty()) {
        VLOG(10) << "No enough match set.";
        return false;
    }

    const MatchSet2D &matches_2d = data_.matches_2d[ref_idx];
    const MatchSet3D &matches_3d = data_.matches_3d;

    int sample_3d_idx =
        RandomSampling(static_cast<int>(matches_3d.size()), rand_eng_);

    float t_scale = 0;
    if (!SolveTranslationalScale(
            matches_2d.m_T1.m_R * matches_3d[sample_3d_idx].m_X +
                matches_2d.m_T1.m_t,
            matches_3d[sample_3d_idx].m_x, cur_T_ref.m_R, cur_T_ref.m_t,
            &t_scale))
        return false;

    cur_T_ref.m_t *= t_scale;

    C_T_W->m_R = cur_T_ref.m_R * matches_2d.m_T1.m_R;
    C_T_W->m_t = cur_T_ref.m_R * matches_2d.m_T1.m_t + cur_T_ref.m_t;

    return true;
}

bool ScaleSolver3D::HasEnoughSupport() const {
    return !data_.matches_3d.empty();
}

bool ScaleSolver3D::SolveTranslationalScale(
    const Vector3f &pt_ref /* 3D point in ref frame */,
    const Vector2f &pt_cur /* 2D point in cur frame */,
    const Matrix3f &cur_R_ref, const Vector3f &cur_t_ref, float *scale) {
    // d -- pt_cur depth
    // s -- translational scale
    // d * pt_cur = R * pt_ref + s * t
    // => [pt_cur, -t] * [d; s] = R * pt_ref  -- a 3x2 linear system

    Eigen::Matrix<float, 3, 2> A;
    A.col(0) = pt_cur.homogeneous();
    A.col(1) = -cur_t_ref;
    Eigen::Vector2f sol =
        (A.transpose() * A).ldlt().solve(A.transpose() * cur_R_ref * pt_ref);
    if (sol.x() < 0 || sol.y() < 0) {
        VLOG(5) << "Negative depth/scale, solution = " << sol.transpose();
        return false;
    }

    *DCHECK_NOTNULL(scale) = sol.y();
    return true;
}

void RoundRobinGenerator::Add(
    std::unique_ptr<HypoGenerator> &&generator, int weight) {
    CHECK_GT(weight, 0);
    generators_.push_back(std::move(generator));
    weights_.push_back(weight);
}

double RoundRobinGenerator::SuccProb(
    double inlier_ratio_3d, double inlier_ratio_2d) const {
    double acc = 0;
    int cnt = 0;
    for (int i = 0; i < static_cast<int>(generators_.size()); i++) {
        acc += weights_[i] *
               generators_[i]->SuccProb(inlier_ratio_3d, inlier_ratio_2d);
        cnt += weights_[i];
    }
    return cnt == 0 ? 0 : acc / cnt;
}

bool RoundRobinGenerator::RunOnce(Pose *C_T_W) {
    CHECK_NOTNULL(C_T_W);

    if (generators_.empty()) {
        VLOG(5) << "No generators.";
        return false;
    }

    const int num_gens = static_cast<int>(generators_.size());
    CHECK_LT(idx_, num_gens);

    if (count_ == weights_[idx_]) {
        count_ = 0;
        idx_ = (idx_ + 1) % num_gens;
    }

    ++count_;
    return generators_[idx_]->RunOnce(C_T_W);
}

bool RoundRobinGenerator::HasEnoughSupport() const {
    return std::any_of(
        generators_.begin(), generators_.end(),
        [](const std::unique_ptr<HypoGenerator> &gen) {
            return gen->HasEnoughSupport();
        });
}

double GravityCheckGenerator::SuccProb(
    double inlier_ratio_3d, double inlier_ratio_2d) const {
    return gen_->SuccProb(inlier_ratio_3d, inlier_ratio_2d);
}

bool GravityCheckGenerator::RunOnce(Pose *C_T_W) {
    return gen_->RunOnce(C_T_W) && CheckGravityDirectionError(C_T_W->m_R);
}

bool GravityCheckGenerator::CheckGravityDirectionError(
    const Eigen::Matrix3f &C_R_W) const {
    const Eigen::Vector3f kGravityDirW = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f C_gravity_dir_estimated = C_R_W * kGravityDirW;
    if (C_gravity_dir_estimated.dot(C_gravity_dir_expected_) <
        cos_gravity_dir_err_th_) {
        VLOG(10) << "Gravity check failed";
        return false;
    }
    return true;
}

} // namespace PS
