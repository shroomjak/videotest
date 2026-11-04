#pragma once

// "Последнее значение" — вырожденный случай очереди ёмкостью 1 без истории.
//
// Когда нужно: UI всегда хочет показать самый свежий кадр. Очередь здесь
// избыточна — достаточно одного слота под мьютексом. Это дешевле, чем
// BoundedQueue<FramePtr>(1, DropOldest), и явнее по смыслу.

#include <mutex>
#include <optional>
#include <utility>

namespace vt {

template <class T>
class LatestValue {
public:
    void set(T value) {
        {
            std::lock_guard lock(mutex_);
            value_ = std::move(value);
            has_value_ = true;
        }
    }

    std::optional<T> get() const {
        std::lock_guard lock(mutex_);
        if (!has_value_) {
            return std::nullopt;
        }
        return value_;  // copy: для shared_ptr это инкремент счётчика ссылок
    }

private:
    mutable std::mutex mutex_;
    T value_{};
    bool has_value_ = false;
};

}  // namespace vt
