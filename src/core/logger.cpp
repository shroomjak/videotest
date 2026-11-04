#include "core/logger.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>

namespace vt {
namespace {

std::atomic<LogLevel> g_level{LogLevel::Info};
std::mutex g_write_mutex;

std::string timestamp_now() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t seconds = system_clock::to_time_t(now);
    const auto millis = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm local{};
    localtime_r(&seconds, &local);  // потокобезопасная версия localtime

    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d.%03d", local.tm_hour, local.tm_min,
                  local.tm_sec, static_cast<int>(millis.count()));
    return buffer;
}

}  // namespace

void set_log_level(LogLevel level) noexcept {
    g_level.store(level, std::memory_order_relaxed);
}

LogLevel log_level() noexcept {
    return g_level.load(std::memory_order_relaxed);
}

bool log_enabled(LogLevel level) noexcept {
    return static_cast<int>(level) >= static_cast<int>(log_level());
}

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info: return "INFO";
        case LogLevel::Warn: return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}

bool parse_log_level(std::string_view text, LogLevel& out) noexcept {
    if (text == "debug") { out = LogLevel::Debug; return true; }
    if (text == "info") { out = LogLevel::Info; return true; }
    if (text == "warn" || text == "warning") { out = LogLevel::Warn; return true; }
    if (text == "error") { out = LogLevel::Error; return true; }
    return false;
}

LogMessage::LogMessage(LogLevel level) : level_(level), enabled_(log_enabled(level)) {}

LogMessage::~LogMessage() {
    if (!enabled_) {
        return;
    }
    // Одна блокировка на строку: строки разных потоков не перемешиваются.
    std::lock_guard lock(g_write_mutex);
    std::ostream& out = (level_ == LogLevel::Error) ? std::cerr : std::cout;
    out << timestamp_now() << " [" << to_string(level_) << "] " << stream_.str() << '\n';
    out.flush();
}

}  // namespace vt
