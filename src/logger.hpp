#pragma once

// Минимальный потокобезопасный логгер: LOG() << "текст" << число;
// Строка копится в ostringstream и пишется целиком под мьютексом в деструкторе
// временного объекта, поэтому строки разных потоков не перемешиваются.

#include <sstream>
#include <string>

namespace vt {

class LogLine {
public:
    explicit LogLine(const char* tag) : tag_(tag) {}
    ~LogLine();

    LogLine(const LogLine&) = delete;
    LogLine& operator=(const LogLine&) = delete;

    template <class T>
    LogLine& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    const char* tag_;
    std::ostringstream stream_;
};

}  // namespace vt

#define LOG_INFO vt::LogLine("INFO")
#define LOG_WARN vt::LogLine("WARN")
#define LOG_ERROR vt::LogLine("ERR ")
