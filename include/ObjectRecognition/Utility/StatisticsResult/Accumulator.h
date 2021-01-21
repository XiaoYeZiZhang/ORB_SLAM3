#ifndef ORB_SLAM3_ACCUMULATOR_H
#define ORB_SLAM3_ACCUMULATOR_H
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <glog/logging.h>

namespace STATISTICS_UTILITY {
static constexpr int kInfiniteWindowSize = std::numeric_limits<int>::max();
template <typename SampleType, typename SumType, int WindowSize>
class Accumulator {
public:
    Accumulator()
        : m_sample_index(0), m_total_samples(0), m_sum(0), m_window_sum(0),
          m_min(std::numeric_limits<SampleType>::max()),
          m_max(std::numeric_limits<SampleType>::lowest()), m_most_recent(0) {
        CHECK_GT(WindowSize, 0);
        if (WindowSize < kInfiniteWindowSize) {
            m_samples.reserve(WindowSize);
        }
    }

    void Add(SampleType sample) {
        m_most_recent = sample;
        if (m_sample_index < WindowSize) {
            m_samples.push_back(sample);
            m_window_sum += sample;
            ++m_sample_index;
        } else {
            SampleType &oldest = m_samples.at(m_sample_index++ % WindowSize);
            m_window_sum += sample - oldest;
            oldest = sample;
        }
        m_sum += sample;
        ++m_total_samples;
        if (sample > m_max) {
            m_max = sample;
        }
        if (sample < m_min) {
            m_min = sample;
        }
    }

    int total_samples() const {
        return m_total_samples;
    }

    SumType sum() const {
        return m_sum;
    }

    SumType Mean() const {
        return (m_total_samples < 1) ? 0.0 : m_sum / m_total_samples;
    }

    // Rolling mean is only used for fixed sized data for now. We don't need
    // this function for our infinite accumulator at this point.
    SumType RollingMean() const {
        if (WindowSize < kInfiniteWindowSize) {
            return m_window_sum / std::min(m_sample_index, WindowSize);
        } else {
            return Mean();
        }
    }

    SampleType GetMostRecent() const {
        return m_most_recent;
    }

    SumType max() const {
        return m_max;
    }

    SumType min() const {
        return m_min;
    }

    SumType LazyVariance() const {
        if (m_samples.size() < 2) {
            return 0.0;
        }

        SumType var = static_cast<SumType>(0.0);
        SumType mean = RollingMean();

        for (unsigned int i = 0; i < m_samples.size(); ++i) {
            var += (m_samples[i] - mean) * (m_samples[i] - mean);
        }

        var /= m_samples.size() - 1;
        return var;
    }

    const std::vector<SampleType> &GetSamples() const {
        return m_samples;
    }

private:
    std::vector<SampleType> m_samples;
    int m_sample_index;
    int m_total_samples;
    SumType m_sum;
    SumType m_window_sum;
    SampleType m_min;
    SampleType m_max;
    SampleType m_most_recent;
};
} // namespace STATISTICS_UTILITY
#endif // ORB_SLAM3_ACCUMULATOR_H
