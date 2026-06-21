#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @brief 线程安全队列模板类
 * @tparam T 存储的数据类型
 * 
 * 说明：使用互斥锁(mutex)和条件变量(condition_variable)实现的多线程安全队列，
 *       支持阻塞式pop操作，当队列为空时等待数据到来。
 *       新增最大容量限制，防止内存无限增长。
 */
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;             // 实际存储数据的队列
    mutable std::mutex mtx;                  // 保护队列的互斥锁
    std::condition_variable cond;    // 用于线程间通信的条件变量
    size_t maxSize = 10;             // 队列最大容量，防止内存爆炸

public:
    /**
     * @brief 向队列推送数据
     * @param item 要推送的数据
     * 
     * 流程：
     * 1. 获取互斥锁
     * 2. 等待队列有空间（当队列满时阻塞）
     * 3. 将数据加入队列
     * 4. 通知一个等待的线程
     */
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        // 等待队列有空位（生产者等待）
        cond.wait(lock, [this]{ return queue.size() < maxSize; });
        queue.push(item);
        cond.notify_one(); // 唤醒一个消费者
    }

    /**
     * @brief 从队列取出数据
     * @return 队列前端的数据
     * 
     * 流程：
     * 1. 获取互斥锁
     * 2. 等待队列非空（当队列空时阻塞）
     * 3. 取出并返回数据
     */
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        // 等待队列有数据（消费者等待）
        cond.wait(lock, [this]{ return !queue.empty(); });
        T item = queue.front();
        queue.pop();
        cond.notify_one(); // 唤醒可能等待的生产者
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx); // 现在 mtx 是 mutable，允许锁定
        return queue.empty();
    }

    // ... 其他方法 ...
};

#endif // THREAD_SAFE_QUEUE_H