#pragma once

// Ограниченная очередь (bounded queue) — "клей" между стадиями конвейера.
//
// Зачем ограничение: если потребитель медленнее производителя, неограниченная
// очередь превращается в утечку памяти и в растущую задержку. Для realtime-задач
// правильная стратегия — не хранить старые кадры, а выбрасывать их.
//
// Три политики переполнения:
//   Block       — производитель спит, пока не появится место (нужно для данных,
//                 которые нельзя терять: например, запись видео на диск);
//   DropOldest  — вытесняется самый старый элемент (дефолт для камеры: нам всегда
//                 нужен самый свежий кадр, а не выстроенная очередь устаревших);
//   DropNewest  — новый элемент отбрасывается (когда важнее уже накопленное).

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace vt {

enum class OverflowPolicy { Block, DropOldest, DropNewest };

template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity, OverflowPolicy policy = OverflowPolicy::DropOldest)
        : capacity_(capacity == 0 ? 1 : capacity), policy_(policy) {}

    // Копирование запрещено: внутри mutex/condition_variable (некопируемые типы),
    // да и семантика "одна очередь — один объект" здесь единственно верная.
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // Возвращает false, если элемент не попал в очередь:
    //   * очередь закрыта;
    //   * политика DropNewest и очередь полна;
    //   * политика DropOldest и пришлось вытеснить более старый элемент
    //     (элемент всё же добавлен — см. backpressure в вызывающем коде).
    bool push(T value) {
        std::unique_lock lock(mutex_);
        if (closed_) {
            return false;
        }
        if (items_.size() >= capacity_) {
            switch (policy_) {
                case OverflowPolicy::Block:
                    not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
                    if (closed_) {
                        return false;
                    }
                    break;
                case OverflowPolicy::DropOldest:
                    items_.pop_front();
                    dropped_.fetch_add(1, std::memory_order_relaxed);
                    break;
                case OverflowPolicy::DropNewest:
                    dropped_.fetch_add(1, std::memory_order_relaxed);
                    return false;
            }
        }
        items_.push_back(std::move(value));
        not_empty_.notify_one();
        return true;
    }

    // Неблокирующее извлечение.
    std::optional<T> try_pop() {
        std::unique_lock lock(mutex_);
        if (items_.empty()) {
            return std::nullopt;
        }
        T value = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return value;
    }

    // Блокирующее извлечение с таймаутом: нужно, чтобы поток мог периодически
    // проверять stop_token / флаг остановки.
    template <class Rep, class Period>
    std::optional<T> pop_wait_for(std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock lock(mutex_);
        const bool ready =
            not_empty_.wait_for(lock, timeout, [this] { return closed_ || !items_.empty(); });
        if (!ready || items_.empty()) {
            return std::nullopt;  // таймаут либо очередь закрыта и пуста
        }
        T value = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return value;
    }

    std::optional<T> pop_wait() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) {
            return std::nullopt;
        }
        T value = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return value;
    }

    // Закрытие: разбудить всех, кто ждёт, и запретить дальнейший push.
    // Единственный корректный способ "мягко" остановить потребителя.
    void close() {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return items_.size();
    }

    std::size_t capacity() const noexcept { return capacity_; }

    // Сколько элементов было отброшено/вытеснено за время жизни очереди.
    std::uint64_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> items_;
    const std::size_t capacity_;
    const OverflowPolicy policy_;
    bool closed_ = false;
    std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace vt
