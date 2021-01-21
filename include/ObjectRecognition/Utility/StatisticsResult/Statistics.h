#ifndef ORB_SLAM3_STATISTICS_H
#define ORB_SLAM3_STATISTICS_H
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "StatisticsResult/Accumulator.h"

namespace STATISTICS_UTILITY {
constexpr double kNumSecondsPerNanosecond = 1.e-9;
struct StatisticsMapValue {
    static const int kWindowSize = 100;
    inline StatisticsMapValue() {
        m_time_last_called = std::chrono::system_clock::now();
    }

    inline void AddValue(double sample) {
        std::chrono::time_point<std::chrono::system_clock> now =
            std::chrono::system_clock::now();
        double dt = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            now - m_time_last_called)
                            .count()) *
                    kNumSecondsPerNanosecond;
        m_time_last_called = now;

        m_values.Add(sample);
        m_time_deltas.Add(dt);
    }
    inline double GetLastDeltaTime() const {
        if (m_time_deltas.total_samples()) {
            return m_time_deltas.GetMostRecent();
        } else {
            return 0;
        }
    }
    int TotalSamples() const {
        return m_values.total_samples();
    }
    double Mean() const {
        return m_values.Mean();
    }
    double Max() const {
        return m_values.max();
    }
    double Min() const {
        return m_values.min();
    }
    double LazyVariance() const {
        return m_values.LazyVariance();
    }
    double MeanCallsPerSec() const {
        double mean_dt = m_time_deltas.Mean();
        if (mean_dt != 0) {
            return 1.0 / mean_dt;
        } else {
            return -1.0;
        }
    }

    double MeanDeltaTime() const {
        return m_time_deltas.Mean();
    }

private:
    Accumulator<double, double, kWindowSize> m_values;
    Accumulator<double, double, kWindowSize> m_time_deltas;
    std::chrono::time_point<std::chrono::system_clock> m_time_last_called;
};

class StatsCollectorImpl {
public:
    explicit StatsCollectorImpl(size_t handle);
    explicit StatsCollectorImpl(std::string const &tag);

    ~StatsCollectorImpl() = default;
    void AddSample(double sample) const;
    void IncrementOne() const;
    size_t GetHandle() const;

private:
    size_t m_handle;
};

class Statistics {
public:
    typedef std::map<std::string, size_t> map_t;
    friend class StatsCollectorImpl;
    static size_t GetHandle(std::string const &tag);
    static double GetMean(size_t handle);
    static size_t GetNumSamples(size_t handle);
    static double GetVariance(size_t handle);
    static double GetVariance(std::string const &tag);
    static double GetMin(size_t handle);
    static double GetMax(size_t handle);
    static double GetHz(size_t handle);
    static double GetHz(std::string const &tag);

    static double GetMeanDeltaTime(std::string const &tag);
    static double GetMeanDeltaTime(size_t handle);
    static double GetLastDeltaTime(std::string const &tag);
    static double GetLastDeltaTime(size_t handle);

    static void Print(std::ostream &out);
    static std::string Print();
    static void Reset();
    static const map_t &GetStatsCollectors() {
        return Instance().m_tag_map;
    }

private:
    void AddSample(size_t handle, double sample);
    static Statistics &Instance();
    Statistics();
    ~Statistics();
    typedef std::vector<STATISTICS_UTILITY::StatisticsMapValue> list_t;
    list_t m_stats_collectors;
    map_t m_tag_map;
    size_t m_max_tag_length;
    std::mutex m_mutex;
};

typedef StatsCollectorImpl StatsCollector;
} // namespace STATISTICS_UTILITY
#endif // ORB_SLAM3_STATISTICS_H
