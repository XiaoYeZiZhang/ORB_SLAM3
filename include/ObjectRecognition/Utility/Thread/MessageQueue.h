#ifndef STSLAM_MESSAGEQUEUE_H_
#define STSLAM_MESSAGEQUEUE_H_

#include <memory>
#include "../Utility.h"

namespace Common {

class MessageQueueImpl;

/// @class MessageQueue MessageQueue.hpp
/// @brief 有限长度的消息队列,线程安全.
class MessageQueue {
public:
    /// @brief 创建消息队列
    /// @param[in] maxQueueSize 消息队列大小
    /// @param[in] isBlock 如果为true, 则队列满时send操作会阻塞;
    /// 否则会丢弃队首消息
    MessageQueue(int maxQueueSize, int isBlock);
    virtual ~MessageQueue();

    /// @brief 返回当前队列大小
    /// @return 返回当前队列大小
    int Size() const;

    int MaxQueueSize() const;
    int IsBlock() const;

    ///< 清除当前队列数据
    void ClearCurrentQueue();

    /// @brief 发送消息, 是否阻塞由构造时的isBlock参数决定
    /// @param[in] type 消息类型, 由调用者自定义, 建议使用正数,
    /// 用于区分不同类型的消息
    /// @param[in] message 消息体
    void SendMessage(int type, const std::shared_ptr<void> &message);

    /// @brief 接收消息, 阻塞直到接收到消息
    /// @param[out] outMessage 返回消息体
    /// @return 返回消息类型
    int RecvMessage(std::shared_ptr<void> &outMessage);

    /// @brief 接收消息, 阻塞直到接收到消息或超时
    /// @param[out] outMessageType 返回消息类型
    /// @param[out] outMessage 返回消息体
    /// @param[in] timeOutMs 超时时间, 单位ms
    /// @return 成功接收到消息返回true, 超时返回false
    bool RecvMessageFor(
        int *outMessageType, std::shared_ptr<void> &outMessage, int timeOutMs);

private:
    std::unique_ptr<MessageQueueImpl> m_pImpl;

    STSLAM_DISABLE_COPY(MessageQueue);
};
} // namespace Common
#endif
