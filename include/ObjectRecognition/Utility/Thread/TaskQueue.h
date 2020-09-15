#ifndef STSLAM_TSAKQUEUE_H_
#define STSLAM_TSAKQUEUE_H_

#include <memory>
#include <functional>
#include "../Utility.h"

namespace Common {

class TaskQueueImpl;

/// @class TaskQueue TaskQueue.hpp
/// @brief 任务队列,线程安全. 每个对象中的所有任务均在同一线程按加入顺序串行执行
/// @warning 必须先析构此类的对象(会等待任务队列结束), 再析构任务中使用的对象
class TaskQueue {
public:
    /// @brief 创建任务队列
    /// @param[in] maxQueueSize 任务队列大小,
    /// 队列满时appendTaskxxx操作会阻塞直到可以加入队列
    TaskQueue(int maxQueueSize);

    /// @brief 销毁任务队列, 会阻塞直到所有任务执行完毕
    virtual ~TaskQueue();

    /// @brief 返回当前队列大小
    /// @return 返回当前队列大小
    int Size();

    /// @brief 添加任务, 会阻塞直到任务添加到队列(不会等执行完毕)
    /// @param[in] task 待添加的任务, 会持有此对象直到任务执行完毕
    /// @param[in] completion 如果非空, 任务执行完毕时会自动执行此回调
    void AppendTaskAsync(
        std::function<void(void)> task,
        std::function<void(void)> completion = nullptr);

    /// @brief 添加任务, 会阻塞直到任务执行完毕
    /// @param[in] task 待添加的任务, 会持有此对象直到任务执行完毕
    void AppendTaskSync(std::function<void(void)> task);

private:
    std::unique_ptr<TaskQueueImpl> m_pImpl;

    STSLAM_DISABLE_COPY(TaskQueue);
};
} // namespace Common
#endif
