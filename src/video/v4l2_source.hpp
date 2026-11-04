#pragma once

// Источник-камера через Video4Linux2 (/dev/videoX).
//
// Реализация использует stream I/O с memory-mapped буферами:
//
//   open(O_RDWR|O_NONBLOCK) -> VIDIOC_QUERYCAP -> VIDIOC_S_FMT -> VIDIOC_S_PARM
//   -> VIDIOC_REQBUFS(MMAP) -> mmap()xN -> VIDIOC_QBUF xN -> VIDIOC_STREAMON
//
//   цикл: poll(fd) -> VIDIOC_DQBUF -> конвертация в BGR -> VIDIOC_QBUF
//
// "Zero-copy" здесь означает, что драйвер пишет пиксели напрямую в память,
// спроектированную в наше адресное пространство: нет read() в промежуточный
// буфер и нет лишнего копирования. Копия остаётся только в конвертации
// YUYV/MJPG -> BGR (её делает OpenCV).

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "video/video_source.hpp"

namespace vt {

struct V4l2Config {
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    int buffer_count = 4;       // mmap-буферов в очереди драйвера
    int poll_timeout_ms = 500;  // сколько ждать кадр в poll()
    bool prefer_mjpeg = true;   // MJPG + сжатие в камере vs сырой YUYV
};

class V4l2Source final : public IVideoSource {
public:
    explicit V4l2Source(V4l2Config config);
    ~V4l2Source() override;

    V4l2Source(const V4l2Source&) = delete;
    V4l2Source& operator=(const V4l2Source&) = delete;

    bool open(std::string& error) override;
    NextStatus next(FramePtr& out, std::string& error) override;
    const SourceInfo& info() const override { return info_; }
    std::string describe() const override;

private:
    // Детали V4L2 (fd, mmap, формат) спрятаны за pimpl, чтобы linux/videodev2.h
    // не протекал в заголовок и не ломал сборку на других платформах.
    struct Impl;

    void close_device();
    bool set_format(std::string& error);
    bool set_frame_rate(std::string& error);
    bool init_buffers(std::string& error);
    void unmap_buffers();
    bool start_streaming(std::string& error);
    NextStatus convert_frame(const void* data, std::size_t length, std::uint64_t seq,
                             SteadyClock::time_point t_captured, FramePtr& out,
                             std::string& error);

    V4l2Config config_;
    SourceInfo info_;
    std::unique_ptr<Impl> impl_;
    std::vector<void*> buffers_{};
    std::vector<std::size_t> buffer_lengths_{};
    std::atomic<std::uint64_t> seq_{0};
    bool streaming_ = false;
};

}  // namespace vt
