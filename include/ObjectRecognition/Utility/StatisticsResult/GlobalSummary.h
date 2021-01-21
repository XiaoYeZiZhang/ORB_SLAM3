#ifndef ORB_SLAM3_GLOBALSUMMARY_H
#define ORB_SLAM3_GLOBALSUMMARY_H
#include <map>
#include <vector>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace ObjRecognition {

class GlobalSummary {
public:
    static void
    SaveTimer(const std::string &result_path, const std::string &timer_result);

    static void SaveStatics(
        const std::string &result_path, const std::string &statics_result,
        const std::string &file_name);
};

} // namespace ObjRecognition
#endif // ORB_SLAM3_GLOBALSUMMARY_H
