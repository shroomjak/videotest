#pragma once

#include <atomic>

// Кооперативная остановка по Ctrl+C. Обработчик сигнала делает ровно одно —
// выставляет атомарный флаг (это единственное, что разрешено в нём делать);
// рабочие потоки сами проверяют флаг в своих циклах.

namespace vt {

using StopFlag = std::atomic<bool>;

inline StopFlag& stop_flag() {
    static StopFlag flag{false};
    return flag;
}

void install_signal_handler();  // SIGINT/SIGTERM -> stop_flag() = true
bool stop_requested();

}  // namespace vt
