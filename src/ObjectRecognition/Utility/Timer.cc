//
// Created by root on 2020/11/15.
//

#include "ObjectRecognition/Utility/Timer.h"
namespace TIMER_UTILITY {
Timer::Timer() {
    start_time = high_resolution_clock::now();
}
Timer::~Timer() {
}
} // namespace TIMER_UTILITY