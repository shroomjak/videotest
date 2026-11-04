#pragma once

// Минимальный интерфейс источника кадров: камера и синтетика реализуют его,
// а конвейер работает с ссылкой на него и не знает, откуда приходят пиксели.
// Три метода — этого достаточно: открыть, забрать кадр, узнать размер.

#include <string>

#include "frame.hpp"

namespace vt {

class Source {
public:
    virtual ~Source() = default;

    virtual bool open(std::string& error) = 0;
    virtual bool grab(FramePtr& out, std::string& error) = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
};

}  // namespace vt
