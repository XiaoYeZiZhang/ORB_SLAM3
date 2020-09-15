#include <iostream>
#include "include/ObjectRecognition/Utility/Thread/Mutex.h"
#include "include/ObjectRecognition/Utility/Thread/Thread.h"

using namespace std;

namespace Common {

namespace deterministic_multi_thread {

bool g_initialized = false;
constexpr bool g_debug = false;

bool Initialized() {
    return g_initialized;
}

void Init(const string &log_path) {
    switch (GetRunMode()) {
    case RunMode::kRecord:
        LockOrderPersistenser::Instance()->SetLogFile(log_path);
        break;
    case RunMode::kReplay:
        LockOrderParser::Instance()->ParseFromFile(log_path);
        break;
    default:
        std::cout << "unsupported mode" << std::endl;
        break;
    }

    g_initialized = true;
}

LockOrderPersistenser *LockOrderPersistenser::Instance() {
    static LockOrderPersistenser instance;
    return &instance;
}

void LockOrderPersistenser::SetLogFile(const string &path) {
#if defined(__APPLE__) || defined(__linux__)
    if (os_.is_open()) {
        std::cout << "you have called SetLogFile" << std::endl;
    }
    os_.open(path);

    if (!os_) {
        std::cout << "can not open file " << path << std::endl;
    }
    std::cout << "In record mode" << std::endl;
#else
    LOG(INFO) << "not In Linux platform, not record lock";
#endif
}

void LockOrderPersistenser::Record(int lock_id) {
    int tid = ThisThread::Current()->Id();
    unique_lock<mutex> lk(mutex_);
#if defined(__APPLE__) || defined(__linux__)
    assert(os_.is_open());
    os_.write((char *)&tid, sizeof(int));
    os_.write((char *)&lock_id, sizeof(int));
//     os_ << tid << ' '  << lock_id << '\n';
#endif
}

LockOrderParser *LockOrderParser::Instance() {
    static LockOrderParser instance(g_debug);
    return &instance;
}

void LockOrderParser::ParseFromFile(const string &path) {
    std::cout << "In replay mode" << std::endl;
    parser_.ParseFromFile(path);
}

int LockOrderParser::QueryNextLockedThread(int lid) {
    unique_lock<mutex> lk(mutex_);

    if ((int)next_thread_.size() <= lid) {
        next_thread_.resize(lid + 1);
    }

    while (next_thread_[lid].empty() && parser_.HasMore()) {
        LoadOne();
    }

    if (next_thread_[lid].empty()) {
        return -1;
    }

    int tid = next_thread_[lid].front();
    next_thread_[lid].pop();
    return tid;
}

int LockOrderParser::QueryNextRunThreadId() {
    unique_lock<mutex> lk(mutex_);
    if (next_thread_ids_.empty())
        return -1;
    int tid = next_thread_ids_.front();
    return tid;
}

void LockOrderParser::PopNextRunThreadId() {
    unique_lock<mutex> lk(mutex_);
    assert(!next_thread_ids_.empty());
    next_thread_ids_.pop();
}

int LockOrderParser::QueryNextLockId(int tid) {
    assert(debug_);

    unique_lock<mutex> lk(mutex_);

    if ((int)next_lock_.size() <= tid) {
        next_lock_.resize(tid + 1);
    }

    while (next_lock_[tid].empty() && parser_.HasMore()) {
        LoadOne();
    }

    if (next_lock_[tid].empty()) {
        return -1;
    }

    int lid = next_lock_[tid].front();
    next_lock_[tid].pop();
    return lid;
}

void LockOrderParser::LoadOne() {
    int tid, lid;
    tie(tid, lid) = parser_.Next();
    next_thread_ids_.push(tid);
    if ((int)next_thread_.size() <= lid) {
        next_thread_.resize(lid + 1);
    }

    next_thread_[lid].push(tid);

    if (!debug_) {
        return;
    }

    if ((int)next_lock_.size() <= tid) {
        next_lock_.resize(tid + 1);
    }

    next_lock_[tid].push(lid);
}

void LockOrderParser::ThreadLockPairParser::ParseFromFile(
    const std::string &path) {
    if (is_.is_open()) {
        std::cout << "you have called ParseFromFile" << std::endl;
    }

    is_.open(path);

    if (!is_) {
        std::cout << "can not open file " << path << std::endl;
    }

    ReadNext();
}

bool LockOrderParser::ThreadLockPairParser::HasMore() {
    return static_cast<bool>(next_);
}

pair<int, int> LockOrderParser::ThreadLockPairParser::Next() {
    pair<int, int> result;

    if (next_) {
        result = std::move(*next_);
        next_.reset();
    } else {
        std::cout << "no more pair, you should call HasMore before Next"
                  << std::endl;
    }

    ReadNext();

    // DLOG(INFO) << ThisThread::Current()->Id() << " get <tid, lid>: <" <<
    // result.first << ", " <<
    //            result.second << '>';

    return result;
}

void LockOrderParser::ThreadLockPairParser::ReadNext() {
    if (next_) {
        return;
    }

    assert(is_.is_open());

    int tid, lid;
    is_.read((char *)&tid, sizeof(int));
    is_.read((char *)&lid, sizeof(int));
    //     is_ >> tid >> lid;
    next_ = make_pair(tid, lid);
}

record_mutex::record_mutex() : record_mutex(next_id()) {
}

record_mutex::record_mutex(int id) : id_(id) {
    // DLOG(INFO) << ThisThread::Current()->Id() << " create " << id;
}

int record_mutex::next_id() {
    static record_mutex mtx(0);
    static int nid = 1;
    mtx.lock();
    int id = nid++;
    mtx.unlock();
    return id;
}

void record_mutex::lock() {
    if (!Initialized()) {
        return;
    }

    std::mutex::lock();
    LockOrderPersistenser::Instance()->Record(id());
    // DLOG(INFO) << ThisThread::Current()->Id() << " acquire " << id();
}

void record_mutex::unlock() {
    if (!Initialized()) {
        return;
    }

    // DLOG(INFO) << ThisThread::Current()->Id() << " release " << id();
    std::mutex::unlock();
}

replay_mutex::replay_mutex() : replay_mutex(next_id()) {
}

replay_mutex::replay_mutex(int id) : id_(id) {
    // DLOG(INFO) << ThisThread::Current()->Id() << " create "  << id;
}

int replay_mutex::next_id() {
    static replay_mutex mtx(0);
    static int nid = 1;
    mtx.lock();
    int id = nid++;
    mtx.unlock();
    return id;
}

void replay_mutex::lock() {
    if (!Initialized()) {
        return;
    }

    int tid = ThisThread::Current()->Id();

    unique_lock<mutex> lk(mutex_);
    condition_.wait(lk, [&] {
        if (next_tid_ == -1)
            load_next_tid();
        return next_tid_ == tid;
        //        if (next_tid_ == tid ) {
        //            while (LockOrderParser::Instance()->QueryNextRunThreadId()
        //            != tid){
        //                std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        //            }
        //            LockOrderParser::Instance()->PopNextRunThreadId();
        //            return true;
        //        } else {
        //            return false;
        //        }
    });
    // DLOG(INFO) << tid << " acquire " << this->id();
}

void replay_mutex::unlock() {
    if (!Initialized()) {
        return;
    }

    unique_lock<mutex> lk(mutex_);
    next_tid_ = -1;
    // DLOG(INFO) << ThisThread::Current()->Id() << " release " << id();
    condition_.notify_all();
}

void replay_mutex::load_next_tid() {
    if (next_tid_ == -1) {
        next_tid_ = LockOrderParser::Instance()->QueryNextLockedThread(id());
    }
}

void ReplayMutex::Locker::Wait(const std::function<bool()> &predicate) {
    while (!predicate()) {
        mutex_.unlock();
        std::this_thread::yield();
        mutex_.lock();
    }
}

LightMutexBase *LightMutexFactory::New() {
    LightMutexBase *ptr = nullptr;

    switch (GetRunMode()) {
    case RunMode::kRecord:
        ptr = new RecordLightMutex();
        break;
    case RunMode::kReplay:
        ptr = new ReplayLightMutex();
        break;
    default:
        std::cout << "unsupported mode" << std::endl;
        break;
    }

    return ptr;
}

MutexBase *MutexFactory::New() {
    MutexBase *ptr = nullptr;

    switch (GetRunMode()) {
    case RunMode::kRecord:
        ptr = new RecordMutex();
        break;
    case RunMode::kReplay:
        ptr = new ReplayMutex();
        break;
    default:
        std::cout << "unsupported mode" << std::endl;
        break;
    }

    return ptr;
}
} // namespace deterministic_multi_thread

} // namespace Common