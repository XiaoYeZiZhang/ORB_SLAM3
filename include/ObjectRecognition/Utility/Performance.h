#ifndef STSLAM_PERFORMANCE_H_
#define STSLAM_PERFORMANCE_H_

///< Trace
#define USE_TRACE
#if (defined(ANDROID) || defined(__ANDOIRD__)) && defined(USE_TRACE)

namespace Common {
class ScopedTrace {
public:
    inline ScopedTrace(const char *name) {
        ATrace_beginSection(name);
    }

    inline ~ScopedTrace() {
        ATrace_endSection();
    }
};
} // namespace Common

#else

#define ATRACE_NAME(name)
#define ATRACE_CALL()

#endif

namespace Common {

///< 绑核, 仅Android平台有效. input_mask(0x01~0x80, 按位对应cpu0~cpu7,
///< cpu7为大核), tid可指定要绑核的线程(-1表示当前线程)
int BindCore(long input_mask, const char *thread_name = nullptr, int tid = -1);
} // namespace Common
#endif