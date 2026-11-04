#include "core/signal_flag.hpp"

#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>

namespace vt {
namespace {

// is_always_lock_free проверяется статически: store в std::atomic безопасен в
// обработчике сигнала только если операция не берёт внутренний мьютекс.
static_assert(std::atomic<bool>::is_always_lock_free,
              "обработчик сигнала требует lock-free atomic");

std::atomic<bool> g_stop{false};

extern "C" void handle_signal(int /*signum*/) {
    // Только асинхронно-безопасное действие: атомарная запись флага.
    g_stop.store(true, std::memory_order_relaxed);
}

}  // namespace

void install_signal_handlers() {
    struct sigaction action {};
    action.sa_handler = &handle_signal;
    sigemptyset(&action.sa_mask);
    // Без SA_RESTART: прерванный poll() вернёт EINTR, поток быстрее увидит флаг.
    action.sa_flags = 0;
    ::sigaction(SIGINT, &action, nullptr);
    ::sigaction(SIGTERM, &action, nullptr);
}

void request_stop() noexcept {
    g_stop.store(true, std::memory_order_relaxed);
}

bool stop_requested() noexcept {
    return g_stop.load(std::memory_order_relaxed);
}

}  // namespace vt
