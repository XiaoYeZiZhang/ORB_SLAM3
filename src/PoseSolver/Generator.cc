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

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

using Eigen::Matrix3f;
using Eigen::Vector2f;
using Eigen::Vector3f;

namespace PS {

template <typename It, typename T>
void ReplaceWith(It beg, It end, T to_be_replaced, T replaced_with) {
    for (; beg != end; ++beg)
        if (*beg == to_be_replaced)
            *beg = replaced_with;
}

template <int N, typename RNG>
std::array<int, N> UnrepeatedRandomSampling(int limit, RNG &gen) {
    DCHECK_GE(limit, N);
    std::array<int, N> result;
    if (N == limit) {
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

    if (!cv::solvePnP(
            obj_pts, img_pts, cv::Mat_<double>::eye(3, 3), cv::noArray(), rvec,
            tvec, kUseExtrinsicGuess, kFlags)) {
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

Vector2f Essential5Generator::SolveDepth(
    const Vector2f &pt_ref, const Vector2f &pt_cur, const Matrix3f &cur_R_ref,
    const Vector3f &cur_t_ref) {
    Eigen::Matrix<float, 3, 2> A;
    A.col(0) = pt_cur.homogeneous();
    A.col(1) = -cur_R_ref * pt_ref.homogeneous();

    return (A.transpose() * A).ldlt().solve(A.transpose() * cur_t_ref);
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
        return false;
    }

    const MatchSet2D &matches_2d = data_.matches_2d[ref_idx];
    const MatchSet3D &matches_3d = data_.matches_3d;

    int sample_3d_idx =
        std::uniform_int_distribution<int>(0, matches_3d.size() - 1)(rand_eng_);

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
} // namespace PS
