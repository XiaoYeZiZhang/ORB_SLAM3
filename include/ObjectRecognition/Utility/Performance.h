#ifndef STSLAM_PERFORMANCE_H_
#define STSLAM_PERFORMANCE_H_

#define USE_TRACE
#if (defined(ANDROID) || defined(__ANDOIRD__)) && defined(USE_TRACE)
#else
#endif

namespace Common {
///< 绑核, 仅Android平台有效. input_mask(0x01~0x80, 按位对应cpu0~cpu7,
///< cpu7为大核), tid可指定要绑核的线程(-1表示当前线程)
int BindCore(long input_mask, const char *thread_name = nullptr, int tid = -1);
} // namespace Common
#endif