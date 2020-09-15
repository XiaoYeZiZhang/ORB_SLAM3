#include "include/ObjectRecognition/Utility/Thread/Thread.h"

namespace Common {

namespace deterministic_multi_thread {

namespace ThisThread {
thread_local Thread *this_Thread = nullptr;

Thread *Current() {
    static Thread *main_thread = Thread::FakeThread(0, "Main");

    if (this_Thread) {
        return this_Thread;
    } else {
        return main_thread;
    }
}
} // namespace ThisThread

// main thread takes the id 0
std::atomic_int Thread::g_id_(1);
} // namespace deterministic_multi_thread

} // namespace Common