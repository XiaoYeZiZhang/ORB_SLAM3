#include "StatisticsResult/Timer.h"
namespace TIMER_UTILITY {
Timer::Timer() {
    m_start_time = high_resolution_clock::now();
}
Timer::~Timer() {
}
} // namespace TIMER_UTILITY