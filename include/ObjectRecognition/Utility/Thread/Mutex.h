#ifndef SENSESLAM_SRC_UTILITY_MUTEX_H_
#define SENSESLAM_SRC_UTILITY_MUTEX_H_

#include <mutex>
#include <condition_variable>
#include <memory>
#include <queue>
#include <fstream>
#include <boost/optional.hpp>
#include "LockAnnotaion.h"

namespace Common {

namespace multi_thread {
class CAPABILITY("mutex") LightMutex {
public:
    class SCOPED_CAPABILITY Locker {
    public:
        Locker(LightMutex &mutex) ACQUIRE(mutex) : lock_(mutex.mutex_) {
        }
        Locker(const Locker &) = delete;
        Locker &operator=(Locker &) = delete;
        ~Locker() RELEASE() = default;

    private:
        std::unique_lock<std::mutex> lock_;
    };

private:
    std::mutex mutex_;
};

class CAPABILITY("mutex") Mutex {
public:
    class SCOPED_CAPABILITY Locker {
    public:
        Locker(Mutex &mutex) ACQUIRE(mutex)
            : mutex_(mutex), lock_(mutex.mutex_) {
        }
        Locker(const Locker &) = delete;
        Locker &operator=(const Locker &) = delete;
        ~Locker() RELEASE() {
            mutex_.condition_.notify_all();
        }

        template <typename Predicate> void Wait(Predicate &&predicate) {
            mutex_.condition_.wait(lock_, std::forward<Predicate>(predicate));
        }

        template <typename Time, typename Predicate>
        bool WaitFor(Time &&time, Predicate &&predicate) {
            return mutex_.condition_.wait_for(
                lock_, std::forward<Time>(time),
                std::forward<Predicate>(predicate));
        }

    private:
        Mutex &mutex_;
        std::unique_lock<std::mutex> lock_;
    };

private:
    std::mutex mutex_;
    std::condition_variable condition_;
};

using LightMutexLocker = LightMutex::Locker;
using MutexLocker = Mutex::Locker;

} // namespace multi_thread

namespace deterministic_multi_thread {

enum RunMode { kRecord, kReplay };

extern int g_lock_id;

void Init(const std::string &log_path);
RunMode GetRunMode();

class Thread;
namespace ThisThread {
Thread *Current();
}; // namespace ThisThread

class LockOrderPersistenser;
class LockOrderParser;
class record_mutex;
class replay_mutex;

class LightMutexBase;
class RecordLightMutex;
class ReplayLightMutex;
class LightMutexFactory;

class MutexBase;
class RecordMutex;
class ReplayMutex;
class MutexFactory;

// *INDENT-OFF*
#define DEFINE_RUN_MODE(mode)                                                  \
    Common::deterministic_multi_thread::RunMode                                \
    Common::deterministic_multi_thread::GetRunMode() {                         \
        static Common::deterministic_multi_thread::RunMode run_mode = mode;    \
        return run_mode;                                                       \
    }
// *INDENT-ON*

class LockOrderPersistenser {
public:
    static LockOrderPersistenser *Instance();

    void SetLogFile(const std::string &path);
    // record this thead has acquired the lock with lock_id
    void Record(int lock_id);

private:
    LockOrderPersistenser() = default;

    ///< 单例对象, 析构时只需要处理线程资源(本类没有开启线程),
    ///< 以及手动申请的内存, 其余靠其自身析构处理即可
    ~LockOrderPersistenser() = default;

    std::ofstream os_;
    std::mutex mutex_;
};

class LockOrderParser {
public:
    static LockOrderParser *Instance();

    void ParseFromFile(const std::string &path);
    // next thread which acuired the lock
    int QueryNextLockedThread(int lock_id);
    // debug only!!!
    int QueryNextLockId(int thread_id);

    // next thread id according to lock file
    int QueryNextRunThreadId();

    void PopNextRunThreadId();

private:
    LockOrderParser(bool debug) : debug_(debug) {
    }

    ///< 单例对象, 析构时只需要处理线程资源(本类没有开启线程),
    ///< 以及手动申请的内存, 其余靠其自身析构处理即可
    ~LockOrderParser() = default;

    void LoadOne() REQUIRES(mutex_);

    class ThreadLockPairParser {
    public:
        void ParseFromFile(const std::string &path);
        bool HasMore();
        std::pair<int, int> Next();

    private:
        void ReadNext();

        std::ifstream is_;
        boost::optional<std::pair<int, int>> next_;
    };

    ThreadLockPairParser parser_ GUARDED_BY(mutex_);
    // next thread for each lock
    std::vector<std::queue<int>> next_thread_ GUARDED_BY(mutex_);
    // next lock for each thread, valid only when debug_ is true
    std::vector<std::queue<int>> next_lock_ GUARDED_BY(mutex_);

    std::queue<int> next_thread_ids_;

    const bool debug_ = false;

    std::mutex mutex_;
    std::condition_variable condition_;
};

class record_mutex : private std::mutex {
public:
    record_mutex();
    void lock();
    void unlock();
    int id() const {
        return id_;
    }

private:
    record_mutex(int id);

    static int next_id();

    const int id_;
};

class replay_mutex {
public:
    replay_mutex();

    void lock();
    void unlock();
    int id() const {
        return id_;
    }

private:
    replay_mutex(int id);

    static int next_id();
    // caller should hold the lock
    void load_next_tid();

    const int id_;
    int next_tid_ = -1;
    std::mutex mutex_;
    std::condition_variable condition_;
};

class LightMutexBase {
public:
    class Locker {
    public:
        virtual ~Locker() {
        }
    };

    virtual ~LightMutexBase() {
    }
    virtual Locker *NewScopedLocker() = 0;
};

class RecordLightMutex : public LightMutexBase {
public:
    class Locker : public LightMutexBase::Locker {
    public:
        Locker(RecordLightMutex &mutex) : mutex_(mutex.mutex_) {
            mutex_.lock();
        }
        ~Locker() {
            mutex_.unlock();
        }

    private:
        record_mutex &mutex_;
    };

    LightMutexBase::Locker *NewScopedLocker() override {
        return new Locker(*this);
    }

private:
    record_mutex mutex_;
};

class ReplayLightMutex : public LightMutexBase {
public:
    class Locker : public LightMutexBase::Locker {
    public:
        Locker(ReplayLightMutex &mutex) : mutex_(mutex.mutex_) {
            mutex_.lock();
        }
        ~Locker() {
            mutex_.unlock();
        }

    private:
        replay_mutex &mutex_;
    };

    LightMutexBase::Locker *NewScopedLocker() override {
        return new Locker(*this);
    }

private:
    replay_mutex mutex_;
};

class MutexBase {
public:
    class Locker {
    public:
        virtual ~Locker() {
        }
        virtual void Wait(const std::function<bool()> &predicate) = 0;
    };

    virtual ~MutexBase() {
    }
    virtual Locker *NewScopedLocker() = 0;
};

class RecordMutex : public MutexBase {
public:
    class Locker : public MutexBase::Locker {
    public:
        Locker(RecordMutex &mutex) : mutex_(mutex) {
            //            LOG(INFO) << "cons";
            lock();
        }
        ~Locker() {
            //            LOG(INFO) << "des";
            unlock();
            mutex_.condition_.notify_all();
        }

        void Wait(const std::function<bool()> &predicate) override {
            mutex_.condition_.wait(*this, predicate);
        }

        void lock() {
            mutex_.mutex_.lock();
        }
        void unlock() {
            //            LOG(INFO) << "unlock";
            mutex_.mutex_.unlock();
        }

    private:
        RecordMutex &mutex_;
    };
    MutexBase::Locker *NewScopedLocker() override {
        return new Locker(*this);
    }

private:
    record_mutex mutex_;
    std::condition_variable_any condition_;
};

class ReplayMutex : public MutexBase {
public:
    class Locker : public MutexBase::Locker {
    public:
        Locker(ReplayMutex &mutex) : mutex_(mutex.mutex_) {
            mutex_.lock();
        }
        ~Locker() {
            mutex_.unlock();
        }
        void Wait(const std::function<bool()> &predicate) override;

    private:
        replay_mutex &mutex_;
    };

    MutexBase::Locker *NewScopedLocker() override {
        return new Locker(*this);
    }

private:
    replay_mutex mutex_;
};

class LightMutexFactory {
public:
    static LightMutexBase *New();
};

class MutexFactory {
public:
    static MutexBase *New();
};

class CAPABILITY("mutex") LightMutex {
public:
    LightMutex() : mutex_(LightMutexFactory::New()) {
    }

    class SCOPED_CAPABILITY Locker {
    public:
        Locker(LightMutex &mutex) ACQUIRE(mutex)
            : locker_(mutex.mutex_->NewScopedLocker()) {
        }
        Locker(const Locker &) = delete;
        Locker &operator=(const Locker &) = delete;
        ~Locker() RELEASE() = default;

    private:
        std::unique_ptr<LightMutexBase::Locker> locker_;
    };

private:
    std::unique_ptr<LightMutexBase> mutex_;
};

class CAPABILITY("mutex") Mutex {
public:
    Mutex() : mutex_(MutexFactory::New()) {
    }

    class SCOPED_CAPABILITY Locker {
    public:
        Locker(Mutex &mutex) ACQUIRE(mutex)
            : locker_(mutex.mutex_->NewScopedLocker()) {
        }
        Locker(const Locker &) = delete;
        Locker &operator=(const Locker &) = delete;
        ~Locker() RELEASE() = default;

        template <typename Predicate> void Wait(Predicate &&predicate) {
            locker_->Wait(predicate);
        }

        template <typename Time, typename Predicate>
        bool WaitFor(Time && /* time */, Predicate &&predicate) {
            // ignore time, to simplify the solution
            Wait(std::forward<Predicate>(predicate));
            return true;
        }

    private:
        std::unique_ptr<MutexBase::Locker> locker_;
    };

private:
    std::unique_ptr<MutexBase> mutex_;
};

using LightMutexLocker = LightMutex::Locker;
using MutexLocker = Mutex::Locker;

} // namespace deterministic_multi_thread

#if defined DET_MULT_THREAD
using namespace deterministic_multi_thread;
#else
using namespace multi_thread;
#endif

} // namespace Common

#endif // SENSESLAM_SRC_UTILITY_MUTEX_H_
