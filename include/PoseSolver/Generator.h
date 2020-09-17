#ifndef POSE_SOLVER_GENERATOR_H_
#define POSE_SOLVER_GENERATOR_H_

#include <memory>
#include <random>

#include <glog/logging.h>

#include "PoseSolver.h"

namespace PS {

// Captures the user input
struct MatchData {
    MatchData(
        const std::vector<MatchSet2D> &matches_2d, const MatchSet3D &matches_3d)
        : matches_2d(matches_2d), matches_3d(matches_3d) {
    }

    std::vector<MatchSet2D> matches_2d;
    MatchSet3D matches_3d;
};

// Hypothesis Generator.
class HypoGenerator {
public:
    virtual ~HypoGenerator() = default;

    // Probability of inlier pose proposal.
    virtual double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const = 0;

    // Generate one hypothesis.
    virtual bool RunOnce(Pose *C_T_W) = 0;

    // Whether does the matches contain enough data to generate hypothesis.
    virtual bool HasEnoughSupport() const = 0;

protected:
    std::default_random_engine rand_eng_;
};

class GravityGenerator : public HypoGenerator {
public:
    explicit GravityGenerator(
        // Assume gravity direciton is [0, 0, 1] in world frame.
        const Eigen::Vector3f &gravity_dir)
        : gravity_dir_(gravity_dir.normalized()) {
        CHECK_LT(std::abs(gravity_dir.norm() - 1.f), 1E-3);
    }

    Eigen::Vector3f get_gravity_dir() const {
        return gravity_dir_;
    }

private:
    const Eigen::Vector3f gravity_dir_;
};

// Solve relative pose between ref frame and cur frame (via. 2D matches).
// TODO(chen): Need a better way to do this.
class ScaleFreeGenerator : public HypoGenerator {
public:
    int get_ref_frame_index() const {
        return frame_index_;
    }

    void set_ref_frame_index(int index) {
        frame_index_ = index;
    }

    // List of 2D matches, which contains enough data to generate hypothesis.
    virtual std::vector<int> GetSupport2DMatches() const = 0;

private:
    int frame_index_ = -1;
};

// This type takes scale less cur_T_ref as input, solve the scale, and compoute
// C_T_W.
class ScaleOnlyGenerator : public HypoGenerator {
public:
    void set_relative_pose(int ref_frame, const Pose &cur_T_ref) {
        ref_frame_ = ref_frame;
        cur_T_ref_ = cur_T_ref;
    }

    void get_relative_pose(int *ref_frame, Pose *cur_T_ref) const {
        *CHECK_NOTNULL(ref_frame) = ref_frame_;
        *CHECK_NOTNULL(cur_T_ref) = cur_T_ref_;
    }

private:
    int ref_frame_ = -1;
    Pose cur_T_ref_;
};

// Glues ScaleFreeGenerator and ScaleOnlyGenerator together to solve C_T_W.
class HybridGenerator : public HypoGenerator {
public:
    HybridGenerator(
        std::unique_ptr<ScaleFreeGenerator> &&scale_free_gen,
        std::unique_ptr<ScaleOnlyGenerator> &&scale_only_gen)
        : scale_free_gen_(std::move(scale_free_gen)),
          scale_only_gen_(std::move(scale_only_gen)) {
        scale_free_gen_support_matches_ =
            scale_free_gen_->GetSupport2DMatches();
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override;

private:
    std::unique_ptr<ScaleFreeGenerator> scale_free_gen_;
    std::vector<int> scale_free_gen_support_matches_;
    int next_ref_index_ = 0; // round robin

    std::unique_ptr<ScaleOnlyGenerator> scale_only_gen_;
};

class PnPGenerator : public HypoGenerator {
public:
    explicit PnPGenerator(const MatchData &data) : matches_(data.matches_3d) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override;

private:
    const MatchSet3D matches_;
};

class P2PGenerator : public GravityGenerator {
public:
    P2PGenerator(const MatchData &data, const Eigen::Vector3f &gravity_dir)
        : GravityGenerator(gravity_dir), matches_(data.matches_3d) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override;

private:
    const MatchSet3D matches_;
};

class EssentialGenerator : public ScaleFreeGenerator {
protected:
    static Eigen::Vector2f SolveDepth(
        const Eigen::Vector2f &pt_ref, const Eigen::Vector2f &pt_cur,
        const Eigen::Matrix3f &cur_R_ref,
        const Eigen::Vector3f &cur_t_ref /* up to a scale */
    );
};

class Essential3Generator : public EssentialGenerator, GravityGenerator {
public:
    Essential3Generator(
        const MatchData &data, const Eigen::Vector3f &gravity_dir)
        : GravityGenerator(gravity_dir), matches_(data.matches_2d) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *cur_T_ref) override;

    std::vector<int> GetSupport2DMatches() const override;

    bool HasEnoughSupport() const override;

protected:
    // Essential3 may give several solutions, we can try to resolve the
    // ambiguity by sampling another match. Or relis on the scale solving to
    // resolve the ambiguity (by remove impossible soultions).
    static constexpr bool kResolveAmbiguityBeforeSolveScale = false;

    void ResolveAmbiguity(
        const Eigen::Vector2d &ref, const Eigen::Vector2d &cur,
        std::vector<Eigen::Matrix3d> *cur_R_ref,
        std::vector<Eigen::Vector3d> *cur_t_ref) const;

    std::vector<MatchSet2D> matches_;
};

class Essential5Generator : public EssentialGenerator {
public:
    explicit Essential5Generator(const MatchData &data)
        : matches_(data.matches_2d) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *cur_T_ref) override;

    std::vector<int> GetSupport2DMatches() const override;

    bool HasEnoughSupport() const override;

protected:
    std::vector<MatchSet2D> matches_;
};

class ScaleSolver2D : public ScaleOnlyGenerator {
public:
    explicit ScaleSolver2D(const MatchData &match_data)
        : matches_(match_data.matches_2d) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override;

private:
    static bool SolveTranslationalScale(
        const Eigen::Vector2f &pt_ref1 /* 2D point in ref1 frame */,
        const Eigen::Vector2f &pt_cur /* 2D point in cur frame */,
        const Eigen::Matrix3f &cur_R_ref0,
        const Eigen::Vector3f &cur_t_ref0 /* unknown scale */,
        const Eigen::Matrix3f &ref0_R_ref1, const Eigen::Vector3f &ref0_t_ref1,
        float *scale);

    const std::vector<MatchSet2D> matches_;
};

class ScaleSolver3D : public ScaleOnlyGenerator {
public:
    explicit ScaleSolver3D(const MatchData &match_data) : data_(match_data) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override;

private:
    static bool SolveTranslationalScale(
        const Eigen::Vector3f &pt_ref /* 3D point in ref frame */,
        const Eigen::Vector2f &pt_cur /* 2D point in cur frame */,
        const Eigen::Matrix3f &cur_R_ref, const Eigen::Vector3f &cur_t_ref,
        float *scale);

    MatchData data_;
};

class RoundRobinGenerator : public HypoGenerator {
public:
    void Add(std::unique_ptr<HypoGenerator> &&generator, int weight);

    bool Empty() const {
        return generators_.empty();
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override;

private:
    std::vector<std::unique_ptr<HypoGenerator>> generators_;
    std::vector<int> weights_;
    // Next generator index, and many time it runs.
    int idx_ = 0, count_ = 0;
};

class GravityCheckGenerator : public HypoGenerator {
public:
    GravityCheckGenerator(
        std::unique_ptr<HypoGenerator> &&gen,
        const Eigen::Vector3f &gravity_dir, const float gravity_dir_max_err_deg)
        : gen_(std::move(gen)), C_gravity_dir_expected_(gravity_dir),
          cos_gravity_dir_err_th_(
              std::cos(gravity_dir_max_err_deg * 3.1415927F / 180)) {
    }

    double
    SuccProb(double inlier_ratio_3d, double inlier_ratio_2d) const override;

    bool RunOnce(Pose *C_T_W) override;

    bool HasEnoughSupport() const override {
        return gen_->HasEnoughSupport();
    }

private:
    bool CheckGravityDirectionError(const Eigen::Matrix3f &C_R_W) const;

    std::unique_ptr<HypoGenerator> gen_;
    const Eigen::Vector3f C_gravity_dir_expected_;
    const float cos_gravity_dir_err_th_;
};

template <typename T, typename... Args>
inline std::unique_ptr<T> MakeIfHasEnoughSupport(Args... args) {
    std::unique_ptr<T> ptr = std::make_unique<T>(std::forward<Args>(args)...);
    if (!ptr->HasEnoughSupport())
        return nullptr;
    return ptr;
}

} // namespace PS

#endif // POSE_SOLVER_GENERATOR_H_
