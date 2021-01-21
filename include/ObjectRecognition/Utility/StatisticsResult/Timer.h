#ifndef ORB_SLAM3_TIMER_H
#define ORB_SLAM3_TIMER_H
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>
using namespace std::chrono;

namespace TIMER_UTILITY {
class Timer {
public:
    Timer();
    ~Timer();
    double Stop() {
        return (duration_cast<microseconds>(
                    high_resolution_clock::now() - m_start_time))
                   .count() /
               1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point m_start_time;
};
} // namespace TIMER_UTILITY
#endif // ORB_SLAM3_TIMER_H
