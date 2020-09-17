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
        mbRequestReset = false;
        mbRequestStop = false;
        mbWaitEndStop = false;
        mbWaitEndReset = false;
    }
    virtual ~ThreadBase() {
        while (!mInputQueue.empty()) {
            mInputQueue.pop();
        }
    }

    bool StartThread() {
        typedef void *(*FUNC)(void *);
        FUNC callback = (FUNC)&ThreadBase::ThreadFun;
        int ret = pthread_create(&pid, NULL, callback, this);
        if (ret != 0) {
            return false;
        } else {
            std::lock_guard<std::mutex> lck(mWaitEndStopMutex);
            mbWaitEndStop = false;
            return true;
        }
    }

    static void *ThreadFun(void *__this) {
        ThreadBase *curThread = reinterpret_cast<ThreadBase *>(__this);
        curThread->Run();
        return nullptr;
    }

    void PushData(const InputDataPtr &data) {
        std::lock_guard<std::mutex> lck(mInputMutex);
        mInputQueue.emplace(data);
    }

    void RequestStop() {
        std::lock_guard<std::mutex> lck(mStateMutex);
        mbRequestStop = true;
    }

    void RequestReset() {
        std::lock_guard<std::mutex> lck(mStateMutex);
        mbRequestReset = true;
    }

    void WaitEndStop() {
        std::lock_guard<std::mutex> lck(mWaitEndStopMutex);
        if (!mbWaitEndStop) {
            void *status;
            pthread_join(pid, &status);
            mbWaitEndStop = true;
            VLOG(5) << "child Thread id:" << pid << " stop";
        }
    }

    bool IsReseting() {
        std::lock_guard<std::mutex> lck(mWaitEndResetMutex);
        return mbWaitEndReset;
    }

    void WaitEndReset() {
        while (IsReseting()) {
            using namespace std::literals::chrono_literals;
            std::this_thread::sleep_for(1ms);
        }
    }

    int Size() {
        std::lock_guard<std::mutex> lck(mStateMutex);
        return mInputQueue.size();
    }

protected:
    void Run() {
        while (true) {
            GetCurInputData();

            Process();

            if (ProcessThreadState()) {
                break;
            }
            {
                int nInputSize = 0;
                {
                    std::lock_guard<std::mutex> lck(mInputMutex);
                    nInputSize = mInputQueue.size();
                }
                if (nInputSize == 0) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(static_cast<int64_t>(10.0)));
                }
            }
        }
    }

    virtual void Process() = 0;
    virtual void GetCurInputData() = 0;
    // virtual void GetResult() = 0;

    /// return true if thread Stoped
    bool ProcessThreadState() {
        bool bStop, bReset;
        {
            std::lock_guard<std::mutex> lck(mStateMutex);
            bStop = mbRequestStop;
            bReset = mbRequestReset;
        }
        if (bStop) {
            Stop();
            {
                std::lock_guard<std::mutex> lck(mStateMutex);
                mbRequestStop = false;
            }
            return true;
        }
        if (bReset) {
            {
                std::lock_guard<std::mutex> lck(mWaitEndResetMutex);
                mbWaitEndReset = true;
            }
            Reset();
            {
                std::lock_guard<std::mutex> lck(mWaitEndResetMutex);
                mbWaitEndReset = false;
            }
            {
                std::lock_guard<std::mutex> lck(mStateMutex);
                mbRequestReset = false;
            }
        }
        return false;
    }

    virtual void Stop() = 0;
    virtual void Reset() = 0;

public:
    pthread_t pid;

protected:
    std::mutex mInputMutex;
    std::queue<InputDataPtr> mInputQueue;

    std::mutex mStateMutex;
    std::mutex mWaitEndStopMutex;
    std::mutex mWaitEndResetMutex;
    bool mbRequestStop;
    bool mbRequestReset;
    bool mbWaitEndStop;
    bool mbWaitEndReset;
};
} // namespace ObjRecognition

#endif // ObjectRecognition_Utility_ThreadBase_H
