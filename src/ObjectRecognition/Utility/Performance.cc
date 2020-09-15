#if defined(ANDROID) || defined(__ANDOIRD__)
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <errno.h>
#include <bitset>
#include <fstream>
#define MAX_LOG_LEVEL 0
#include "glog/logging.h"
#include "Helper/Utility.h"
#endif

#include "include/ObjectRecognition/Utility/Performance.h"

namespace Common {

#if defined(ANDROID) || defined(__ANDOIRD__)
///< 根据cpu索引 获取最大频率
static int GetCPUMaxFreqByIndex(int cpu_index) {
    int result = -1;
    const std::string cpuinfo_max_freq_filepath =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) +
        "/cpufreq/cpuinfo_max_freq";
    FILE *fd = fopen(cpuinfo_max_freq_filepath.c_str(), "r");
    if (nullptr == fd) {
        LOG(WARNING) << "open file " << cpuinfo_max_freq_filepath
                     << " failed, errno = " << errno;
        return result;
    }

    int cpu_max_freq = -1;
    if (1 == fscanf(fd, "%d", &cpu_max_freq)) {
        result = cpu_max_freq;
    }
    fclose(fd);
    return result;
}

///< 根据cpu频率区分大小核
static int DetectBigCoreByFreq(bool &out_is_the_first_core_big) {
    const int first_cpu_max_freq = GetCPUMaxFreqByIndex(0);
    const int last_cpu_max_freq = GetCPUMaxFreqByIndex(7);

    // 小于100MHz, 频率可能有问题, 就不靠频率区分了
    if ((first_cpu_max_freq <= 0) || (last_cpu_max_freq <= 0) ||
        (std::abs(first_cpu_max_freq - last_cpu_max_freq) < 100000)) {
        LOG(WARNING) << "DetectBigCoreByFreq failed, cpu0 freq = "
                     << first_cpu_max_freq
                     << ", cpu7 freq = " << last_cpu_max_freq;
        return -1;
    }

    // 如果0号核的频率大于7号核, 则返回true
    out_is_the_first_core_big = (first_cpu_max_freq > last_cpu_max_freq);
    LOG(INFO) << "DetectBigCoreByFreq ok, is cpu0 big core = "
              << out_is_the_first_core_big
              << ", cpu0 freq = " << first_cpu_max_freq
              << ", cpu7 freq = " << last_cpu_max_freq;
    return 0;
}

///< 根据SoC芯片类型区分大小核
static int DetectBigCoreBySoCType(bool &out_is_the_first_core_big) {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        LOG(WARNING) << "open file /proc/cpuinfo failed, errno = " << errno;
        return -1;
    }

    std::string lineString;
    while (std::getline(cpuinfo, lineString)) {
        if (0 == lineString.find("Hardware")) {
            // 这里认为MTK的为0号是大核, 其余情况7号是大核
            out_is_the_first_core_big =
                (std::string::npos != lineString.find("MT")) &&
                (std::string::npos == lineString.find("Qualcomm"));
            LOG(INFO) << "DetectBigCoreBySoCType ok, is cpu0 big core = "
                      << out_is_the_first_core_big << ", from " << lineString;
            return 0;
        }
    }

    LOG(WARNING) << "Not Found Hardware item in /proc/cpuinfo";
    return -1;
}

///< 第一个0号核是否是大核
static bool IsTheFirstCoreBig() {
    bool result = false;
    // 先尝试使用频率判断
    if (0 != DetectBigCoreByFreq(result)) {
        // 频率判断失败则使用SoC芯片类型判断
        if (0 != DetectBigCoreBySoCType(result)) {
            // 无法判断时, 认为cpu7为大核(多数高通cpu)
            result = false;
        }
    }
    LOG(INFO) << "is cpu0 big core = " << result;
    return result;
}

static long AutoChangeMask(long input_mask) {
    static const int k_is_the_first_core_big = IsTheFirstCoreBig();

    long result = input_mask;

    // 若序号为0的核为大核(如mtk), 在这里需要做一下交换
    if (k_is_the_first_core_big) {
        const int k_core_number = 8;
        const std::bitset<k_core_number> src_mask(input_mask);
        std::bitset<k_core_number> dst_mask;
        for (int i = 0; i < k_core_number; ++i) {
            dst_mask[i] = src_mask[k_core_number - i - 1];
        }
        result = dst_mask.to_ulong();
    }

    VLOG(3) << "is cpu0 big core = " << k_is_the_first_core_big
            << ", auto change mask 0x" << std::hex << input_mask << " --> 0x"
            << result;
    return result;
}

int BindCore(long input_mask, const char *thread_name, int tid) {
    int result = 0;
    long cur_mask = 0;
    long dst_mask = AutoChangeMask(input_mask);
    pid_t dst_pid = (tid >= 0) ? tid : gettid();

    long get_mask_result =
        syscall(__NR_sched_getaffinity, dst_pid, sizeof(cur_mask), &cur_mask);
    VLOG(3) << "BindCore " << thread_name << ": [" << dst_pid
            << "]: get current mask = 0x" << std::hex << cur_mask
            << ", result = " << std::dec << get_mask_result
            << ", sizeof(cur_mask) = " << sizeof(cur_mask)
            << ". will set to mask 0x" << std::hex << dst_mask;

    // 获取mask不成功或者目标mask不同于当前mask, 则需要重新设置mask
    if ((sizeof(cur_mask) != get_mask_result) || (cur_mask != dst_mask)) {
        long set_mask_result = syscall(
            __NR_sched_setaffinity, dst_pid, sizeof(dst_mask), &dst_mask);
        if (0 == set_mask_result) {
            LOG(INFO) << "BindCore " << thread_name << ": [" << dst_pid
                      << "]: get current result = " << get_mask_result
                      << ", sizeof(cur_mask) = " << sizeof(cur_mask)
                      << ", set mask = 0x" << std::hex << cur_mask << " --> 0x"
                      << dst_mask << " ok";
        } else {
            LOG(WARNING) << "BindCore " << thread_name << ": [" << dst_pid
                         << "]: get current result = " << get_mask_result
                         << ", sizeof(cur_mask) = " << sizeof(cur_mask)
                         << ", set mask = 0x" << std::hex << cur_mask
                         << " --> 0x" << dst_mask
                         << " failed, result = " << set_mask_result
                         << ", errno = " << errno;
            result = -1;
        }
    }

    if (!STSLAM_CSTR_IS_EMPTY(thread_name)) {
        prctl(PR_SET_NAME, thread_name);
    }
    return result;
}

#else
// Not Android
int BindCore(long input_mask, const char *thread_name, int tid) {
}
#endif
} // namespace Common