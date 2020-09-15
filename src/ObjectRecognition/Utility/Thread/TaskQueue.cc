#include <mutex>
#include <thread>
#include <condition_variable>
#include "include/ObjectRecognition/Utility/Thread/MessageQueue.h"
#include "include/ObjectRecognition/Utility/Thread/TaskQueue.h"

namespace Common {

enum { TQ_NEW_TASK_MESSAGE = 0, TQ_END_TASK_MESSAGE };

class TaskMessage {
public:
    TaskMessage(std::function<void()> task, std::function<void()> completion);

    void RunTask();

private:
    std::function<void()> m_task;
    std::function<void()> m_completion;
};

TaskMessage::TaskMessage(
    std::function<void()> task, std::function<void()> completion)
    : m_task(task), m_completion(completion) {
}

void TaskMessage::RunTask() {
    if (m_task) {
        // STID_DEBUG_DEBUG_LOG("start task");
        m_task();
        // STID_DEBUG_DEBUG_LOG("finish task");
    } else {
        // STID_DEBUG_WARN_LOG("skip null task");
    }

    if (m_completion) {
        // STID_DEBUG_DEBUG_LOG("start completion");
        m_completion();
        // STID_DEBUG_DEBUG_LOG("finish completion");
    }
}

static void RunTaskQueue(std::weak_ptr<MessageQueue> taskQueue) {
    while (1) {
        std::shared_ptr<MessageQueue> curMQ = taskQueue.lock();
        if (!curMQ) {
            // STID_DEBUG_INFO_LOG("RunTaskQueue will return soon, because
            // taskQueue was expired!");
            break;
        }

        std::shared_ptr<void> tmpMessage;
        int messageType = curMQ->RecvMessage(tmpMessage);
        curMQ.reset();

        if (TQ_END_TASK_MESSAGE == messageType) {
            // STID_DEBUG_INFO_LOG("RunTaskQueue will return soon, because recv
            // TQ_END_TASK_MESSAGE!");
            break;
        }

        std::shared_ptr<TaskMessage> curTask =
            std::static_pointer_cast<TaskMessage>(tmpMessage);
        if ((TQ_NEW_TASK_MESSAGE != messageType) || (!curTask)) {
            // STID_DEBUG_ERROR_LOG("RunTaskQueue will return soon, because recv
            // invalid message, type %d, message is %p!", messageType,
            // tmpMessage.get());
            break;
        }

        curTask->RunTask();
    }

    // STID_DEBUG_INFO_LOG("RunTaskQueue thread already returned!");

    return;
}

class TaskQueueImpl {
public:
    TaskQueueImpl(int maxQueueSize);
    ~TaskQueueImpl();

    int Size();

    void AppendTaskAsync(
        std::function<void(void)> task,
        std::function<void(void)> completion = nullptr);
    void AppendTaskSync(std::function<void(void)> task);

private:
    std::mutex m_mutexForSyncTask;
    std::condition_variable m_waitSyncTaskCompletion;
    std::shared_ptr<MessageQueue> m_taskQueue;
    std::thread m_runThread;

    STSLAM_DISABLE_COPY(TaskQueueImpl);
};

TaskQueueImpl::TaskQueueImpl(int maxQueueSize)
    : m_taskQueue(std::make_shared<MessageQueue>(maxQueueSize, true)) {
    m_runThread = std::thread(RunTaskQueue, m_taskQueue);
}

TaskQueueImpl::~TaskQueueImpl() {
    m_taskQueue->SendMessage(TQ_END_TASK_MESSAGE, std::shared_ptr<void>());
    m_runThread.join();
}

int TaskQueueImpl::Size() {
    return m_taskQueue->Size();
}

void TaskQueueImpl::AppendTaskAsync(
    std::function<void()> task, std::function<void()> completion) {
    m_taskQueue->SendMessage(
        TQ_NEW_TASK_MESSAGE, std::make_shared<TaskMessage>(task, completion));
}

void TaskQueueImpl::AppendTaskSync(std::function<void()> task) {
    std::unique_lock<std::mutex> lockWaitTask(m_mutexForSyncTask);

    bool isTaskCompletion = false;
    AppendTaskAsync(task, [&isTaskCompletion, this] {
        {
            std::lock_guard<std::mutex> lockTaskCompletion(m_mutexForSyncTask);
            isTaskCompletion = true;
        }
        m_waitSyncTaskCompletion.notify_all();
    });

    while (!isTaskCompletion) {
        m_waitSyncTaskCompletion.wait(lockWaitTask);
    }
}

TaskQueue::TaskQueue(int maxQueueSize)
    : m_pImpl(new TaskQueueImpl(maxQueueSize)) {
}

TaskQueue::~TaskQueue() {
    m_pImpl.reset();
}

int TaskQueue::Size() {
    return m_pImpl->Size();
}

void TaskQueue::AppendTaskAsync(
    std::function<void()> task, std::function<void()> completion) {
    m_pImpl->AppendTaskAsync(task, completion);
}

void TaskQueue::AppendTaskSync(std::function<void()> task) {
    m_pImpl->AppendTaskSync(task);
}

} // namespace Common
