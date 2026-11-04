#include "stop_flag.hpp"

#include <csignal>

namespace vt {
namespace {
extern "C" void on_signal(int) {
    // Единственное асинхронно-безопасное действие: запись в атомарный флаг.
    stop_flag().store(true, std::memory_order_relaxed);
}
}  // namespace

void install_signal_handler() {
    std::signal(SIGINT, &on_signal);
    std::signal(SIGTERM, &on_signal);
}

bool stop_requested() {
    return stop_flag().load(std::memory_order_relaxed);
}

}  // namespace vt
