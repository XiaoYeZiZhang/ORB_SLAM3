//
// Created by zhangye on 2020/9/17.
//
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "Utility/GlobalSummary.h"
namespace ObjRecognition {

std::map<
    std::string, std::vector<std::pair<
                     double, std::pair<Eigen::Quaterniond, Eigen::Vector3d>>>>
    GlobalSummary::all_poses_;
std::mutex GlobalSummary::pose_mutex_;

std::string GlobalSummary::m_dataset_path;

void GlobalSummary::AddPose(
    const std::string &pose_name,
    const std::pair<double, std::pair<Eigen::Quaterniond, Eigen::Vector3d>>
        &pose) {
#ifdef MOBILE_PLATFORM
    return;
#endif

    std::lock_guard<std::mutex> lock_guard(pose_mutex_);

    all_poses_[pose_name].push_back(pose);

} // GlobalSummary::AddPose

void GlobalSummary::SaveAllPoses(const std::string &result_path) {

#ifdef MOBILE_PLATFORM
    return;
#endif

    pose_mutex_.lock();

    std::cout << all_poses_.size() << std::endl;
    for (auto const &trajectory : all_poses_) {

        std::ofstream f_stream;
        std::string result_file = result_path + "/" + trajectory.first + ".txt";
        VLOG(10) << "Global summary save poses: " << result_file;
        f_stream.open(result_file);

        for (auto const &pose : trajectory.second) {
            Eigen::Vector3d t = pose.second.second;
            Eigen::Quaterniond q = pose.second.first;
            f_stream << std::to_string(pose.first) << ","
                     << std::setprecision(7) << t(0) << "," << t(1) << ","
                     << t(2) << "," << q.w() << "," << q.x() << "," << q.y()
                     << "," << q.z() << std::endl;
        }
    }

    pose_mutex_.unlock();
} // GlobalSummary::SaveAllPoses

void GlobalSummary::SaveTimer(
    const std::string &result_path, const std::string &timer_result) {

    std::ofstream f_stream;
    std::string result_file = result_path + "/" + "timer_result.txt";
    VLOG(10) << "Global summary timer result: " << result_file;
    f_stream.open(result_file);

    f_stream << "Dataset path" << std::endl;
    f_stream << "-----------" << std::endl;
    f_stream << m_dataset_path.c_str() << std::endl;
    f_stream << std::endl;
    f_stream << timer_result << std::endl;

    f_stream.close();
}

void GlobalSummary::SaveStatics(
    const std::string &result_path, const std::string &statics_result) {
    std::ofstream f_stream;
    std::string result_file = result_path + "/" + "statics_result.txt";
    VLOG(10) << "Global summary statics result: " << result_file;
    f_stream.open(result_file);

    f_stream << "Dataset path" << std::endl;
    f_stream << "-----------" << std::endl;
    f_stream << m_dataset_path << std::endl;
    f_stream << std::endl;
    f_stream << statics_result << std::endl;

    f_stream.close();
}

void GlobalSummary::SetDatasetPath(const std::string &dataset_path) {
    m_dataset_path = dataset_path;
}

} // namespace ObjRecognition