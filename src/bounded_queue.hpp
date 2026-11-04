#pragma once

// Очередь с ограниченной ёмкостью — единственное место, где потоки
// обмениваются данными. Политика переполнения одна: вытесняем самый старый
// элемент. Для realtime это правильный выбор — нам нужен свежий кадр, а не
// очередь устаревших (иначе задержка растёт до бесконечности).

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace vt {

template <class T>
class Queue {
public:
    explicit Queue(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    // Всегда принимает элемент: при переполнении вытесняет самый старый.
    // Возвращает количество вытесненных элементов (0 в обычном случае).
    std::size_t push(T value) {
        std::size_t dropped = 0;
        {
            std::lock_guard lock(mutex_);
            if (items_.size() >= capacity_) {
                items_.pop_front();
                ++dropped;
            }
            items_.push_back(std::move(value));
        }
        not_empty_.notify_one();
        return dropped;
    }

    // Блокирующее извлечение с таймаутом: поток просыпается, чтобы проверить
    // флаг остановки, даже если данных нет.
    template <class Rep, class Period>
    std::optional<T> pop(std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock lock(mutex_);
        not_empty_.wait_for(lock, timeout, [this] { return !items_.empty(); });
        if (items_.empty()) {
            return std::nullopt;
        }
        T value = std::move(items_.front());
        items_.pop_front();
        return value;
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return items_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::deque<T> items_;
    const std::size_t capacity_;
};

}  // namespace vt
