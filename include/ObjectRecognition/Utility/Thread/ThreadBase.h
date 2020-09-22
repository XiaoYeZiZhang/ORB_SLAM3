#ifndef STSLAM_THREADBASE_H_
#define STSLAM_THREADBASE_H_

#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <condition_variable>
#include <memory>
#include "../Utility.h"
#include "MessageQueue.h"
#include "Thread.h"

namespace Common {

/// 基于状态的线程模型
class ThreadBase {
public:
    typedef enum {
        STOPPED = 0,       ///< 停止状态, 初始状态
        RUNNING,           ///< 运行状态
        RESETTING,         ///< 重置中
        WAITING_FOR_START, ///< 等待重新开始
        STOPPING           ///< 停止中
    } STATE;
    virtual ~ThreadBase() = 0;

public:
    ///*** public接口均线程安全 ***///

    ///< 开始运行: STOPPED --> RUNNING
    int StartThread(
        const std::string &threadName = "ORB_SLAM3", int cpuMask = 0x00);

    ///< 请求停止: RUNNING|RESETTING|WAITING_FOR_START --> STOPPING
    int RequestStop();

    ///< 等待直到线程停止运行: STOPPING --> STOPPED
    int WaitEndStop();

    ///< 请求重置: RUNNING --> RESETTING
    int RequestReset();

    ///< 等待直到重置完成: RESETTING --> WAITING_FOR_START
    int WaitEndReset();

    ///< 开始运行: WAITING_FOR_START --> RUNNING
    int StartRunning();

    ///< 返回当前状态
    STATE CurrentState() const {
        return m_currentState.load();
    }

    ///< dataType必须>=0, 阻塞模式, 若队列满, 则等待直到可以插入队列.
    ///< 非阻塞模式, 若队列满则移除队首元素
    void PushData(int dataType, const std::shared_ptr<void> &data);

    int Size() const;

protected:
    ///< 指定队列大小, 以及push是否阻塞
    ThreadBase(int maxQueueSize, int isPushBlock);

    ///*** 内部接口, 均在工作线程执行 ***///

    ///< 清空现有队列
    void ClearDataQueue();

    ///< 返回dataType, 若dataType < 0表示线程已经准备停止, 无法再获取数据
    int PopFront(std::shared_ptr<void> &outData);

    ///*** 子类实现, 运行均在同一线程 ***///
    virtual int Reset() = 0;
    virtual int Stop() = 0;
    virtual int Process() = 0;

private:
    int m_cpuMask;
    std::string m_threadName;
    std::atomic<STATE> m_currentState;
    std::condition_variable m_waitEndReset;

    std::unique_ptr<Thread> m_workerThread;
    std::mutex m_mutexForPublicAPI;
    MessageQueue m_inputDataQueue;

    void RunThread();
    int RequestStop_();
    int WaitEndStop_();
    void Clear_();

    STSLAM_DISABLE_COPY(ThreadBase);
};

} // namespace Common
#endif