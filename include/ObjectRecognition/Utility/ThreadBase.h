#ifndef ObjectRecognition_Utility_ThreadBase_H
#define ObjectRecognition_Utility_ThreadBase_H
#include <pthread.h>
#include <thread>
#include <memory>
#include <queue>
#include <mutex>
#include <glog/logging.h>
namespace ObjRecognition {

template <typename InputData> class ThreadBase {
public:
    typedef std::shared_ptr<InputData> InputDataPtr;

public:
    ThreadBase() {
        m_request_reset = false;
        m_request_stop = false;
        m_wait_end_stop = false;
        m_wait_end_reset = false;
    }
    virtual ~ThreadBase() {
        while (!m_input_queue.empty()) {
            m_input_queue.clear();
            //            m_input_queue.pop();
        }
    }

    bool StartThread() {
        typedef void *(*FUNC)(void *);
        FUNC callback = (FUNC)&ThreadBase::ThreadFun;
        int ret = pthread_create(&pid, NULL, callback, this);
        if (ret != 0) {
            return false;
        } else {
            std::lock_guard<std::mutex> lck(m_wait_end_stop_Mutex);
            m_wait_end_stop = false;
            return true;
        }
    }

    static void *ThreadFun(void *__this) {
        ThreadBase *curThread = reinterpret_cast<ThreadBase *>(__this);
        curThread->Run();
        return nullptr;
    }

    void PushData(const InputDataPtr &data) {
        std::lock_guard<std::mutex> lck(m_input_mutex);
        m_input_queue.emplace_back(data);
    }

    void RequestStop() {
        std::lock_guard<std::mutex> lck(m_state_mutex);
        m_request_stop = true;
    }

    void RequestReset() {
        std::lock_guard<std::mutex> lck(m_state_mutex);
        m_request_reset = true;
    }

    void WaitEndStop() {
        std::lock_guard<std::mutex> lck(m_wait_end_stop_Mutex);
        if (!m_wait_end_stop) {
            void *status;
            pthread_join(pid, &status);
            m_wait_end_stop = true;
        }
    }

    bool IsReseting() {
        std::lock_guard<std::mutex> lck(m_wait_end_reset_Mutex);
        return m_wait_end_reset;
    }

    void WaitEndReset() {
        while (IsReseting()) {
            using namespace std::literals::chrono_literals;
            std::this_thread::sleep_for(1ms);
        }
    }

    int Size() {
        std::lock_guard<std::mutex> lck(m_state_mutex);
        return m_input_queue.size();
    }

protected:
    void Run() {
        while (true) {
            GetNewestData();
            Process();
            if (ProcessThreadState()) {
                break;
            }
            {
                int nInputSize = 0;
                {
                    std::lock_guard<std::mutex> lck(m_input_mutex);
                    nInputSize = m_input_queue.size();
                }
                if (nInputSize == 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(static_cast<int64_t>(10.0)));
                }
            }
        }
    }

    virtual void Process() = 0;
    virtual void GetNewestData() = 0;

    bool ProcessThreadState() {
        bool bStop, bReset;
        {
            std::lock_guard<std::mutex> lck(m_state_mutex);
            bStop = m_request_stop;
            bReset = m_request_reset;
        }
        if (bStop) {
            Stop();
            {
                std::lock_guard<std::mutex> lck(m_state_mutex);
                m_request_stop = false;
            }
            return true;
        }
        if (bReset) {
            {
                std::lock_guard<std::mutex> lck(m_wait_end_reset_Mutex);
                m_wait_end_reset = true;
            }
            Reset();
            {
                std::lock_guard<std::mutex> lck(m_wait_end_reset_Mutex);
                m_wait_end_reset = false;
            }
            {
                std::lock_guard<std::mutex> lck(m_state_mutex);
                m_request_reset = false;
            }
        }
        return false;
    }

    virtual void Stop() = 0;
    virtual void Reset() = 0;

public:
    pthread_t pid;

protected:
    std::mutex m_input_mutex;
    std::vector<InputDataPtr> m_input_queue;

    std::mutex m_state_mutex;
    std::mutex m_wait_end_stop_Mutex;
    std::mutex m_wait_end_reset_Mutex;
    bool m_request_stop;
    bool m_request_reset;
    bool m_wait_end_stop;
    bool m_wait_end_reset;
};
} // namespace ObjRecognition

#endif // ObjectRecognition_Utility_ThreadBase_H
