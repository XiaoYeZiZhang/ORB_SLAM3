#include <cmath>
#include <ostream>
#include "StatisticsResult/Statistics.h"

namespace STATISTICS_UTILITY {

Statistics &Statistics::Instance() {
    static Statistics instance;
    return instance;
}

Statistics::Statistics() : m_max_tag_length(0) {
}

Statistics::~Statistics() {
}

StatsCollectorImpl::StatsCollectorImpl(size_t handle) : m_handle(handle) {
}

StatsCollectorImpl::StatsCollectorImpl(std::string const &tag)
    : m_handle(Statistics::GetHandle(tag)) {
}

size_t Statistics::GetHandle(std::string const &tag) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    map_t::iterator i = Instance().m_tag_map.find(tag);
    if (i == Instance().m_tag_map.end()) {
        size_t handle = Instance().m_stats_collectors.size();
        Instance().m_tag_map[tag] = handle;
        Instance().m_stats_collectors.push_back(StatisticsMapValue());
        Instance().m_max_tag_length =
            std::max(Instance().m_max_tag_length, tag.size());
        return handle;
    } else {
        return i->second;
    }
}

size_t StatsCollectorImpl::GetHandle() const {
    return m_handle;
}
void StatsCollectorImpl::AddSample(double sample) const {
    Statistics::Instance().AddSample(m_handle, sample);
}
void StatsCollectorImpl::IncrementOne() const {
    Statistics::Instance().AddSample(m_handle, 1.0);
}
void Statistics::AddSample(size_t handle, double seconds) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    m_stats_collectors[handle].AddValue(seconds);
}
double Statistics::GetMean(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].Mean();
}
size_t Statistics::GetNumSamples(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].TotalSamples();
}
double Statistics::GetVariance(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].LazyVariance();
}
double Statistics::GetVariance(std::string const &tag) {
    return GetVariance(GetHandle(tag));
}
double Statistics::GetMin(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].Min();
}
double Statistics::GetMax(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].Max();
}
double Statistics::GetHz(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].MeanCallsPerSec();
}
double Statistics::GetHz(std::string const &tag) {
    return GetHz(GetHandle(tag));
}

double Statistics::GetMeanDeltaTime(std::string const &tag) {
    return GetMeanDeltaTime(GetHandle(tag));
}
double Statistics::GetMeanDeltaTime(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].MeanDeltaTime();
}
double Statistics::GetLastDeltaTime(std::string const &tag) {
    return GetLastDeltaTime(GetHandle(tag));
}
double Statistics::GetLastDeltaTime(size_t handle) {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    return Instance().m_stats_collectors[handle].GetLastDeltaTime();
}
void Statistics::Print(std::ostream &out) { // NOLINT
    const map_t &tag_map = Instance().m_tag_map;

    if (tag_map.empty()) {
        return;
    }

    out << "Statistics\n";

    out.width((std::streamsize)Instance().m_max_tag_length);
    out.setf(std::ios::left, std::ios::adjustfield);
    out << "-----------";
    out.width(7);
    out.setf(std::ios::right, std::ios::adjustfield);
    out << "#\t";
    out << "Hz\t";
    out << "(avg     +- std    )\t";
    out << "[min,max]\n";

    for (const typename map_t::value_type &t : tag_map) {
        size_t i = t.second;
        out.width((std::streamsize)Instance().m_max_tag_length);
        out.setf(std::ios::left, std::ios::adjustfield);
        out << t.first << "\t";
        out.width(7);

        out.setf(std::ios::right, std::ios::adjustfield);
        out << GetNumSamples(i) << "\t";
        if (GetNumSamples(i) > 0) {
            out << GetHz(i) << "\t";
            double mean = GetMean(i);
            double stddev = sqrt(GetVariance(i));
            out << "(" << mean << " +- ";
            out << stddev << ")\t";

            double min_value = GetMin(i);
            double max_value = GetMax(i);

            out << "[" << min_value << "," << max_value << "]";
        }
        out << std::endl;
    }
}

std::string Statistics::Print() {
    std::stringstream ss;
    Print(ss);
    return ss.str();
}

void Statistics::Reset() {
    std::lock_guard<std::mutex> lock(Instance().m_mutex);
    Instance().m_tag_map.clear();
}

} // namespace STATISTICS_UTILITY
