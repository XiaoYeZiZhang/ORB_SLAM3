#include <list>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include "include/ObjectRecognition/Utility/Thread/MessageQueue.h"

namespace Common {

class MessageQueueImpl {
public:
    MessageQueueImpl(int maxQueueSize, int isBlock);

    int Size() const;
    int MaxQueueSize() const;
    int IsBlock() const;
    void ClearCurrentQueue();

    void SendMessage(int type, const std::shared_ptr<void> &message);

    bool RecvMessageFor(
        int *outMessageType, std::shared_ptr<void> &outMessage, int timeOutMs);

private:
    class MessageInfo {
    public:
        const int messageType;
        const std::shared_ptr<void> messageData;
    };

    const int m_maxQueueSize;
    const int m_isBlock;

    std::list<MessageInfo> m_queue;
    mutable std::mutex m_mutexForQueue;
    std::condition_variable m_waitQueueNotEmpty;
    std::condition_variable m_waitQueueNotFull;
};

static const int kMinQueueSize = 1;

MessageQueueImpl::MessageQueueImpl(int maxQueueSize, int isBlock)
    : m_maxQueueSize((std::max)(maxQueueSize, kMinQueueSize)),
      m_isBlock(isBlock) {
}

int MessageQueueImpl::Size() const {
    std::lock_guard<std::mutex> lockQueue(m_mutexForQueue);
    return m_queue.size();
}

int MessageQueueImpl::MaxQueueSize() const {
    return m_maxQueueSize;
}

int MessageQueueImpl::IsBlock() const {
    return m_isBlock;
}

void MessageQueueImpl::ClearCurrentQueue() {
    {
        std::lock_guard<std::mutex> lockQueue(m_mutexForQueue);
        m_queue.clear();
    }
    m_waitQueueNotFull.notify_all();
}

void MessageQueueImpl::SendMessage(
    int type, const std::shared_ptr<void> &message) {
    {
        std::unique_lock<std::mutex> lockQueue(m_mutexForQueue);

        if (m_isBlock) {
            while (m_queue.size() >= m_maxQueueSize) {
                m_waitQueueNotFull.wait(lockQueue);
            }
        } else {
            while (m_queue.size() >= m_maxQueueSize) {
                m_queue.pop_front();
            }
        }

        m_queue.push_back({type, message});
    }

    m_waitQueueNotEmpty.notify_one();
}

bool MessageQueueImpl::RecvMessageFor(
    int *outMessageType, std::shared_ptr<void> &outMessage, int timeOutMs) {
    std::unique_lock<std::mutex> lockQueue(m_mutexForQueue);

    bool result = true;

    if (timeOutMs < 0) {
        while (m_queue.empty()) {
            m_waitQueueNotEmpty.wait(lockQueue);
        }
    } else {
        result = m_waitQueueNotEmpty.wait_for(
            lockQueue, std::chrono::milliseconds(timeOutMs),
            [this]() -> bool { return !(m_queue.empty()); });
    }

    if (result) {
        const MessageInfo tmpMessage = m_queue.front();
        m_queue.pop_front();
        lockQueue.unlock();

        m_waitQueueNotFull.notify_one();
        outMessage = tmpMessage.messageData;
        if (nullptr != outMessageType) {
            *outMessageType = tmpMessage.messageType;
        }
    }
    return result;
}

MessageQueue::MessageQueue(int maxQueueSize, int isBlock)
    : m_pImpl(new MessageQueueImpl(maxQueueSize, isBlock)) {
}

MessageQueue::~MessageQueue() {
}

int MessageQueue::Size() const {
    return m_pImpl->Size();
}

int MessageQueue::MaxQueueSize() const {
    return m_pImpl->MaxQueueSize();
}

int MessageQueue::IsBlock() const {
    return m_pImpl->IsBlock();
}

void MessageQueue::ClearCurrentQueue() {
    m_pImpl->ClearCurrentQueue();
}

void MessageQueue::SendMessage(int type, const std::shared_ptr<void> &message) {
    m_pImpl->SendMessage(type, message);
}

int MessageQueue::RecvMessage(std::shared_ptr<void> &outMessage) {
    int tmpMessageType = 0;
    if (RecvMessageFor(&tmpMessageType, outMessage, -1)) {
        return tmpMessageType;
    } else {
        return -1;
    }
}

bool MessageQueue::RecvMessageFor(
    int *outMessageType, std::shared_ptr<void> &outMessage, int timeOutMs) {
    return m_pImpl->RecvMessageFor(outMessageType, outMessage, timeOutMs);
}

} // namespace Common
