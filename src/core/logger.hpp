#pragma once

// Минималистичный потокобезопасный логгер.
//
// Использование:
//     LOG_INFO << "камера открыта: " << path << " " << w << "x" << h;
//
// Приём: временный объект LogMessage копит текст в std::ostringstream и в
// деструкторе пишет строку целиком под мьютексом. Так строки от разных потоков
// не перемешиваются, а форматирование не требует std::format (которого нет в
// libstdc++ до GCC 13).

#include <atomic>
#include <sstream>
#include <string>
#include <string_view>

namespace vt {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void set_log_level(LogLevel level) noexcept;
LogLevel log_level() noexcept;
bool log_enabled(LogLevel level) noexcept;

std::string_view to_string(LogLevel level) noexcept;
bool parse_log_level(std::string_view text, LogLevel& out) noexcept;

class LogMessage {
public:
    explicit LogMessage(LogLevel level);
    ~LogMessage();

    LogMessage(const LogMessage&) = delete;
    LogMessage& operator=(const LogMessage&) = delete;

    template <class T>
    LogMessage& operator<<(const T& value) {
        if (enabled_) {
            stream_ << value;
        }
        return *this;
    }

private:
    LogLevel level_;
    bool enabled_;
    std::ostringstream stream_;
};

}  // namespace vt

#define LOG_DEBUG vt::LogMessage(vt::LogLevel::Debug)
#define LOG_INFO vt::LogMessage(vt::LogLevel::Info)
#define LOG_WARN vt::LogMessage(vt::LogLevel::Warn)
#define LOG_ERROR vt::LogMessage(vt::LogLevel::Error)
