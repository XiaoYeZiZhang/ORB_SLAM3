#ifndef SLAM_SRC_UTILITY_THREAD_H_
#define SLAM_SRC_UTILITY_THREAD_H_

#include <thread>
#include <atomic>
#include <string>
#include <functional>

namespace Common {

namespace multi_thread {

namespace ThisThread {
inline void Yield() {
}
} // namespace ThisThread

class Thread : public std::thread {
public:
    Thread() = default;

    Thread(std::string name, int /* ignore */ = 0) : name_(std::move(name)) {
    }

    template <class Function, class... Args>
    Thread(std::string name, int /* ignore */, Function &&f, Args &&... args)
        : std::thread(std::forward<Function>(f), std::forward<Args>(args)...),
          name_(std::move(name)) {
    }

    const std::string &Name() {
        return name_;
    }

private:
    std::string name_ = "anonymous";
};
} // namespace multi_thread

namespace deterministic_multi_thread {
/*
 * main thread id is 0
 */
class Thread;

namespace ThisThread {
extern thread_local Thread *this_Thread;
Thread *Current();
inline void Yield() {
}
} // namespace ThisThread

class Thread {
public:
    static Thread *FakeThread(int id, std::string name) {
        Thread *t = new Thread();
        t->id_ = id;
        t->name_ = std::move(name);
        return t;
    }

    // without id!
    Thread() {
    }

    Thread(std::string name, int = 0 /* ignore */) : id_(g_id_++), name_(name) {
    }

    template <class Function, class... Args>
    Thread(std::string name, int /* ignore */, Function &&f, Args &&... args)
        : id_(g_id_++), name_(std::move(name)) {
        thread_ = std::thread(MakeWrapper(
            this,
            std::bind(std::forward<Function>(f), std::forward<Args>(args)...)));
    }

    // we will use thread address in latter scheduling, so disablbe duplication
    Thread(Thread &&) = delete;

    // we will use thread address in latter scheduling, so disablbe duplication
    Thread &operator=(Thread &&) = delete;

    int Id() const {
        return id_;
    }
    const std::string &Name() const {
        return name_;
    }

    bool joinable() {
        return thread_.joinable();
    }

    void join() {
        thread_.join();
    }

    void detach() {
        thread_.detach();
    }

private:
    template <class Callable> struct FuncWrapper {
        FuncWrapper(Thread *t, Callable &&f)
            : thread_(t), func_(std::forward<Callable>(f)) {
        }

        void operator()() {
            ThisThread::this_Thread = thread_;
            func_();
        }

        Thread *thread_ = nullptr;
        typename std::decay<Callable>::type func_;
    };

    template <typename Callable>
    FuncWrapper<Callable> MakeWrapper(Thread *t, Callable &&f) {
        return FuncWrapper<Callable>(t, std::forward<Callable>(f));
    }

    int id_ = -1;
    std::string name_ = "anonymous";
    std::thread thread_;

    static std::atomic_int g_id_;
};
} // namespace deterministic_multi_thread

#if defined DET_MULT_THREAD
using namespace deterministic_multi_thread;
#else
using namespace multi_thread;
#endif
} // namespace Common
#endif // SLAM_SRC_UTILITY_THREAD_H_