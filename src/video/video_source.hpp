#pragma once

// Абстракция источника кадров.
//
// Зачем интерфейс: остальной конвейер не должен знать, откуда пришли пиксели —
// из /dev/videoX (V4L2), из файла (cv::VideoCapture) или из синтетического
// генератора. Это позволяет отлаживать модель и трекинг без камеры, а также
// подменять источник в тестах.
//
// Паттерн: классический полиморфизм (virtual-функции) + фабрика. Наследование
// здесь уместно, потому что источник — это "роль", а не "тип данных":
// у него есть состояние (открытый fd, буферы) и жизненный цикл.

#include <memory>
#include <ostream>
#include <string>

#include "core/frame.hpp"

namespace vt {

// Результат попытки получить следующий кадр.
enum class NextStatus {
    Frame,        // кадр получен, out заполнен
    Timeout,      // за отведённое время камера не отдала кадр (не фатально)
    Error,        // фатальная ошибка (errno, отключение устройства)
    EndOfStream,  // файл закончился
};

// Фактические параметры потока (могут отличаться от запрошенных: драйвер
// вправе выбрать ближайший поддерживаемый режим).
struct SourceInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    std::string pixel_format;  // "MJPG", "YUYV", "BGR" (файл), ...
};

class IVideoSource {
public:
    virtual ~IVideoSource() = default;

    // Открыть источник. error заполняется при неудаче.
    virtual bool open(std::string& error) = 0;

    // Забрать следующий кадр. Блокируется не дольше внутреннего таймаута,
    // чтобы вызывающий поток мог регулярно проверять флаг остановки.
    virtual NextStatus next(FramePtr& out, std::string& error) = 0;

    virtual const SourceInfo& info() const = 0;
    virtual std::string describe() const = 0;
};

using SourcePtr = std::unique_ptr<IVideoSource>;

}  // namespace vt
