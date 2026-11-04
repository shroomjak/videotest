#include "streams/v4l2_source.hpp"

#include <cerrno>
#include <cstring>
#include <utility>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"

namespace vt {
namespace {

std::string errno_text(int code) {
    // Не используем strerror_r: у GNU и XSI разные сигнатуры и типы возврата,
    // а тексты ошибок нужны только в логе — потоками здесь не пахнет.
    char buffer[256] = {};
    const char* text = std::strerror(code);
    std::snprintf(buffer, sizeof(buffer), "%s", text != nullptr ? text : "unknown error");
    return buffer;
}

// ioctl повторяем при EINTR: системный вызов может быть прерван сигналом
// (например, нашим SIGINT), и это не ошибка устройства.
template <class... Args>
int xioctl(int fd, unsigned long request, Args... args) {
    int result;
    do {
        result = ::ioctl(fd, request, args...);
    } while (result == -1 && errno == EINTR);
    return result;
}

}  // namespace

V4l2Source::V4l2Source(V4l2Config config) : config_(std::move(config)) {}

V4l2Source::~V4l2Source() {
    close();
}

bool V4l2Source::open(std::string& error) {
    // O_NONBLOCK: DQBUF не должен блокировать поток — ожидание делаем в poll().
    fd_ = ::open(config_.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {
        error = "не удалось открыть " + config_.device + ": " + errno_text(errno);
        return false;
    }
    if (!configure(error) || !map_buffers(error)) {
        close();
        return false;
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        error = "VIDIOC_STREAMON: " + errno_text(errno);
        close();
        return false;
    }
    LOG_INFO << "камера: " << config_.device << ' ' << width_ << 'x' << height_ << " @ "
             << config_.fps << " fps, формат " << format_;
    return true;
}

bool V4l2Source::configure(std::string& error) {
    v4l2_capability capability{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
        error = "VIDIOC_QUERYCAP: " + errno_text(errno);
        return false;
    }
    // У многофункциональных устройств реальные возможности узла лежат в
    // device_caps; проверять только capabilities — частая причина ложного отказа.
    const std::uint32_t caps = (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                                   ? capability.device_caps
                                   : capability.capabilities;
    if ((caps & V4L2_CAP_VIDEO_CAPTURE) == 0 || (caps & V4L2_CAP_STREAMING) == 0) {
        error = config_.device + ": нет V4L2_CAP_VIDEO_CAPTURE + V4L2_CAP_STREAMING";
        return false;
    }

    // Явный формат: MJPG (сжатый, меньше данных по USB) или YUYV (сырой, без
    // декодирования). Никакого перебора: неизвестное имя — ошибка аргументов.
    std::uint32_t pixel_format = 0;
    if (config_.format == "MJPG") {
        pixel_format = V4L2_PIX_FMT_MJPEG;
    } else if (config_.format == "YUYV") {
        pixel_format = V4L2_PIX_FMT_YUYV;
    } else {
        error = "неизвестный формат '" + config_.format + "', поддерживаются MJPG и YUYV";
        return false;
    }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = static_cast<std::uint32_t>(config_.width);
    fmt.fmt.pix.height = static_cast<std::uint32_t>(config_.height);
    fmt.fmt.pix.pixelformat = pixel_format;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        error = "VIDIOC_S_FMT " + config_.format + ": " + errno_text(errno);
        return false;
    }
    // Драйвер ВПРАВЕ скорректировать размер, но не формат. Подменённый формат —
    // это молчаливое изменение контракта: сообщаем и выходим.
    if (fmt.fmt.pix.pixelformat != pixel_format) {
        error = "драйвер не поддерживает формат " + config_.format +
                " (навязал другой fourcc) — см. v4l2-ctl --list-formats-ext";
        return false;
    }
    width_ = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);
    stride_ = static_cast<int>(fmt.fmt.pix.bytesperline);
    format_ = config_.format;
    if (width_ != config_.width || height_ != config_.height) {
        LOG_WARN << "драйвер скорректировал размер до " << width_ << 'x' << height_;
    }

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<std::uint32_t>(config_.fps);
    if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        LOG_WARN << "частота кадров не задана драйвером, продолжаем как есть";
    }
    return true;
}

bool V4l2Source::map_buffers(std::string& error) {
    v4l2_requestbuffers request{};
    request.count = static_cast<std::uint32_t>(config_.buffer_count);
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &request) < 0) {
        error = "VIDIOC_REQBUFS: " + errno_text(errno);
        return false;
    }
    if (request.count < 2) {
        error = "драйвер выделил меньше двух буферов";
        return false;
    }

    buffers_.resize(request.count);
    lengths_.resize(request.count);
    for (std::uint32_t i = 0; i < request.count; ++i) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
            error = "VIDIOC_QUERYBUF: " + errno_text(errno);
            return false;
        }
        void* address = ::mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                               static_cast<off_t>(buffer.m.offset));
        if (address == MAP_FAILED) {
            error = "mmap: " + errno_text(errno);
            return false;
        }
        buffers_[i] = address;
        lengths_[i] = buffer.length;
        // Буфер сразу возвращаем драйверу, иначе STREAMON ничего не отдаст.
        if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
            error = "VIDIOC_QBUF: " + errno_text(errno);
            return false;
        }
    }
    return true;
}

void V4l2Source::unmap_buffers() {
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
        if (buffers_[i] != nullptr && buffers_[i] != MAP_FAILED) {
            ::munmap(buffers_[i], lengths_[i]);
        }
    }
    buffers_.clear();
    lengths_.clear();
}

void V4l2Source::close_device() {
    if (fd_ >= 0) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
        // munmap строго до close: память закрытого fd использовать нельзя.
        unmap_buffers();
        ::close(fd_);
        fd_ = -1;
    } else {
        unmap_buffers();
    }
}

void V4l2Source::close() {
    close_device();
}

bool V4l2Source::grab(FramePtr& out, std::string& error) {
    out.reset();
    if (fd_ < 0) {
        error = "устройство не открыто";
        return false;
    }

    pollfd descriptor{};
    descriptor.fd = fd_;
    descriptor.events = POLLIN;
    const int ready = ::poll(&descriptor, 1, config_.poll_timeout_ms);
    if (ready == 0) {
        return true;  // кадра пока нет — не ошибка
    }
    if (ready < 0) {
        if (errno == EINTR) {
            return true;  // сигнал: цикл снаружи проверит флаг остановки
        }
        error = "poll: " + errno_text(errno);
        return false;
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        error = "устройство отключено или недоступно";
        return false;
    }

    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
        if (errno == EAGAIN) {
            return true;
        }
        error = "VIDIOC_DQBUF: " + errno_text(errno);
        return false;
    }

    // Время фиксируем сразу после DQBUF: тогда метрика «камера -> разметка»
    // не включает ожидание в poll.
    const auto captured = std::chrono::steady_clock::now();
    cv::Mat image;
    bool converted = true;
    const bool has_buffer = buffer.index < buffers_.size() && buffer.bytesused > 0;

    if (!has_buffer) {
        converted = false;  // пустой кадр (бывает при сбросе экспозиции)
    } else if (format_ == "MJPG") {
        // Кадр MJPG — это самостоятельный JPEG-файл в памяти.
        const cv::Mat encoded(1, static_cast<int>(buffer.bytesused), CV_8UC1,
                              buffers_[buffer.index]);
        converted = !cv::imdecode(encoded, cv::IMREAD_COLOR, &image).empty();
    } else {
        // YUYV: оборачиваем память драйвера без копирования. step обязателен —
        // bytesperline может быть больше width*2 из-за выравнивания строк.
        const int step = stride_ > 0 ? stride_ : width_ * 2;
        const cv::Mat yuyv(height_, width_, CV_8UC2, buffers_[buffer.index], step);
        cv::cvtColor(yuyv, image, cv::COLOR_YUV2BGR_YUY2);
    }

    // Буфер возвращаем ВСЕГДА — иначе через buffer_count кадров камера встанет.
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
        error = "VIDIOC_QBUF после обработки: " + errno_text(errno);
        return false;
    }
    if (!converted || image.empty()) {
        return true;  // битый кадр пропускаем, но конвейер не роняем
    }

    auto frame = std::make_shared<Frame>();
    frame->image = std::move(image);
    frame->id = next_id_++;
    frame->captured = captured;
    out = std::move(frame);
    return true;
}

}  // namespace vt
