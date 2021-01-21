#include <iostream>
#include <fstream>
#include "StatisticsResult/GlobalSummary.h"
namespace ObjRecognition {
void GlobalSummary::SaveTimer(
    const std::string &result_path, const std::string &timer_result) {

    std::ofstream f_stream;
    std::string result_file = result_path + "/" + "timer_result.txt";
    f_stream.open(result_file);
    f_stream << timer_result << std::endl;
    f_stream.close();
}

void GlobalSummary::SaveStatics(
    const std::string &result_path, const std::string &statics_result,
    const std::string &file_name) {
    std::ofstream f_stream;
    std::string result_file = result_path + "/" + file_name;
    f_stream.open(result_file);
    f_stream << statics_result << std::endl;
    f_stream.close();
}
} // namespace ObjRecognition