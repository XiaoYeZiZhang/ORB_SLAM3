//
// Created by zhangye on 2020/9/17.
//

#ifndef ORB_SLAM3_GLOBALSUMMARY_H
#define ORB_SLAM3_GLOBALSUMMARY_H
#include <map>
#include <vector>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace ObjRecognition {

class GlobalSummary {

private:
    static std::map<
        std::string,
        std::vector<
            std::pair<double, std::pair<Eigen::Quaterniond, Eigen::Vector3d>>>>
        all_poses_;
    static std::mutex pose_mutex_;

public:
    static void AddPose(
        const std::string &pose_name,
        const std::pair<double, std::pair<Eigen::Quaterniond, Eigen::Vector3d>>
            &pose);

    static void SaveAllPoses(const std::string &result_path);

    static void
    SaveTimer(const std::string &result_path, const std::string &timer_result);

    static void SaveStatics(
        const std::string &result_path, const std::string &statics_result,
        const std::string &file_name);

    static void SetDatasetPath(const std::string &dataset_path);

private:
    static std::string m_dataset_path;
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_GLOBALSUMMARY_H
