#ifndef SAFE_QUEUE_H
#define SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    mutable std::mutex m_mutex;
    std::condition_variable conva;
    const int m_max_size = 5;

public:
    explicit SafeQueue(int size) : m_max_size(size) {}

    void push(const T& item){
        std::unique_lock<std::mutex> lock(m_mutex);
        conva.wait(lock, [this]() {
            return q.size() < m_max_size;
        });
        q.push(item);
        conva.notify_all();
    }

    T pop(){
        std::unique_lock<std::mutex> lock(m_mutex);

        conva.wait(lock, [this]() {
            return !q.empty();
        });

        T item = q.front();
        q.pop();

        conva.notify_all();

        return item;
    }

    bool empty() const{
        std::lock_guard<std::mutex> lock(m_mutex);
        return q.empty();
    }

    int size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return q.size();
    }
};

#endif 