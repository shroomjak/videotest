#include "logger.hpp"

#include <iostream>
#include <mutex>

namespace vt {
namespace {
std::mutex g_mutex;
}  // namespace

LogLine::~LogLine() {
    std::lock_guard lock(g_mutex);
    std::ostream& out = (std::string(tag_) == "ERR ") ? std::cerr : std::cout;
    out << '[' << tag_ << "] " << stream_.str() << '\n';
}

}  // namespace vt
