#pragma once

// Захват с /dev/videoX через Video4Linux2.
//
// Формат пикселей задаётся ЯВНО (MJPG или YUYV) и никогда не подбирается
// автоматически: если драйвер не принял запрошенный формат — это ошибка, а не
// повод молча переключиться на другой. Так проще понимать, что происходит.
//
// Используется streaming I/O с memory-mapped буферами: драйвер пишет кадры
// прямо в отображённую память, промежуточной копии через read() нет.
//
//   open -> S_FMT -> REQBUFS(MMAP) -> mmap xN -> QBUF xN -> STREAMON
//   цикл: poll -> DQBUF -> конвертация в BGR -> QBUF

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "streams/source.hpp"

namespace vt {

struct V4l2Config {
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    std::string format = "MJPG";  // "MJPG" | "YUYV" — задаётся пользователем
    int buffer_count = 4;
    int poll_timeout_ms = 500;
};

class V4l2Source final : public Source {
public:
    explicit V4l2Source(V4l2Config config);
    ~V4l2Source() override;

    V4l2Source(const V4l2Source&) = delete;
    V4l2Source& operator=(const V4l2Source&) = delete;

    bool open(std::string& error) override;

    // Забрать следующий кадр. Ложь + error — фатальная ошибка (камеру выдернули).
    // Таймаут кадра (нет данных) не считается ошибкой: вернётся пустой FramePtr.
    bool grab(FramePtr& out, std::string& error) override;

    int width() const override { return width_; }
    int height() const override { return height_; }
    const std::string& format() const { return format_; }

    void close();

private:
    bool configure(std::string& error);
    bool map_buffers(std::string& error);
    void unmap_buffers();
    void close_device();

    V4l2Config config_;
    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    int stride_ = 0;   // bytesperline: реальная длина строки, может быть > width*bpp
    std::string format_;

    std::vector<void*> buffers_;
    std::vector<std::size_t> lengths_;
    std::uint64_t next_id_ = 0;
};

}  // namespace vt
