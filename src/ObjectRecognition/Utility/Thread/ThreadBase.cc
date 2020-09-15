#include "include/ObjectRecognition/Utility/Performance.h"
#include "include/ObjectRecognition/Utility/Thread/ThreadBase.h"
#include "include/ObjectRecognition/Utility/Thread/Thread.h"

namespace Common {

ThreadBase::ThreadBase(int maxQueueSize, int isPushBlock)
    : m_cpuMask(0x00), m_currentState(STOPPED),
      m_inputDataQueue(maxQueueSize, isPushBlock) {
}

ThreadBase::~ThreadBase() {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);
    Clear_();
}

int ThreadBase::StartThread(const std::string &threadName, int cpuMask) {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);

    Clear_();
    m_cpuMask = cpuMask;
    m_threadName = threadName;
    m_workerThread = std::unique_ptr<Thread>(
        new Thread(threadName, 0, &ThreadBase::RunThread, this));
    m_currentState.store(RUNNING);
    return 0;
}

int ThreadBase::RequestStop() {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);
    return RequestStop_();
}

int ThreadBase::WaitEndStop() {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);
    return WaitEndStop_();
}

int ThreadBase::RequestReset() {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);
    STATE tmpState = RUNNING;
    if (m_currentState.compare_exchange_strong(tmpState, RESETTING)) {
        m_inputDataQueue.SendMessage(-1, std::shared_ptr<void>());
        return 0;
    }
    return -1;
}

int ThreadBase::WaitEndReset() {
    std::unique_lock<std::mutex> lockMutex(m_mutexForPublicAPI);

    while (RESETTING == CurrentState()) {
        m_waitEndReset.wait(lockMutex);
    }

    return (WAITING_FOR_START == CurrentState()) ? 0 : -1;
}

int ThreadBase::StartRunning() {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);
    STATE tmpState = WAITING_FOR_START;
    return (m_currentState.compare_exchange_strong(tmpState, RUNNING)) ? 0 : -1;
}

void ThreadBase::PushData(int dataType, const std::shared_ptr<void> &data) {
    std::lock_guard<std::mutex> lockMutex(m_mutexForPublicAPI);
    if (RUNNING == CurrentState() && (dataType >= 0)) {
        m_inputDataQueue.SendMessage(dataType, data);
    }
}

int ThreadBase::Size() const {
    return m_inputDataQueue.Size();
}

void ThreadBase::ClearDataQueue() {
    m_inputDataQueue.ClearCurrentQueue();
}

int ThreadBase::PopFront(std::shared_ptr<void> &outData) {
    if (RUNNING != CurrentState()) {
        return -1;
    }
    return m_inputDataQueue.RecvMessage(outData);
}

void ThreadBase::RunThread() {
    if (m_cpuMask > 0x00) {
        BindCore(m_cpuMask, m_threadName.c_str());
    }

    while (1) {
        STATE tmpState = CurrentState();

        // stopping --> return --> stopped
        if (STOPPING == tmpState) {
            break;
        }

        // resetting -> reset --> waiting for start
        if (RESETTING == tmpState) {
            m_inputDataQueue.ClearCurrentQueue();
            Reset();
            m_currentState.compare_exchange_strong(tmpState, WAITING_FOR_START);
            m_waitEndReset.notify_all();
            continue;
        }

        Process();
    }

    Stop();
    m_inputDataQueue.ClearCurrentQueue();
}

void ThreadBase::Clear_() {
    if (0 == RequestStop_()) {
        WaitEndStop_();
    }
}

int ThreadBase::RequestStop_() {
    if (STOPPED != CurrentState()) {
        m_currentState.store(STOPPING);
        m_inputDataQueue.SendMessage(-1, std::shared_ptr<void>());
    }
    return 0;
}

int ThreadBase::WaitEndStop_() {
    if (STOPPING == CurrentState()) {
        if (m_workerThread) {
            m_workerThread->join();
            m_workerThread.reset();
        }
        m_currentState.store(STOPPED);
        m_cpuMask = 0x00;
        m_threadName.clear();
    }
    return 0;
}
} // namespace Common