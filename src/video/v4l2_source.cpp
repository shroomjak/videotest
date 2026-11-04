#include "video/v4l2_source.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/logger.hpp"

namespace vt {
namespace {

// RAII-обёртка над системным вызовом, который возвращает int и сигналит об
// ошибке значением -1 через errno. Шаблон с variadic args — чтобы не писать
// проверку errno после каждого ioctl.
template <class... Args>
int xioctl(int fd, unsigned long request, Args&&... args) {
    int result;
    do {
        result = ::ioctl(fd, request, std::forward<Args>(args)...);
    } while (result == -1 && errno == EINTR);  // ioctl может быть прерван сигналом
    return result;
}

std::string errno_text(int code) {
    char buffer[256];
    // XSI-версия strerror_r возвращает int, GNU — char*. #ifdef делает код
    // переносимым между libc.
#ifdef _GNU_SOURCE
    char* message = ::strerror_r(code, buffer, sizeof(buffer));
    return message;
#else
    if (::strerror_r(code, buffer, sizeof(buffer)) != 0) {
        return "неизвестная ошибка";
    }
    return buffer;
#endif
}

std::string fourcc_to_string(std::uint32_t fmt) {
    std::string text(4, ' ');
    text[0] = static_cast<char>(fmt & 0xFF);
    text[1] = static_cast<char>((fmt >> 8) & 0xFF);
    text[2] = static_cast<char>((fmt >> 16) & 0xFF);
    text[3] = static_cast<char>((fmt >> 24) & 0xFF);
    return text;
}

}  // namespace

struct V4l2Source::Impl {
    int fd = -1;
    std::uint32_t pixel_format = 0;
    int stride = 0;  // bytesperline: реальная длина строки в байтах
};

V4l2Source::V4l2Source(V4l2Config config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {}

V4l2Source::~V4l2Source() {
    close_device();
}

std::string V4l2Source::describe() const {
    std::ostringstream out;
    out << config_.device << " (" << info_.width << 'x' << info_.height << " @ " << info_.fps
        << " fps, " << info_.pixel_format << ')';
    return out.str();
}

bool V4l2Source::open(std::string& error) {
    // O_NONBLOCK: open() не должен висеть, если камера "занята"; далее режим
    // нужен, чтобы DQBUF не блокировал поток — ожидание делаем в poll().
    impl_->fd = ::open(config_.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (impl_->fd < 0) {
        error = "не удалось открыть " + config_.device + ": " + errno_text(errno);
        return false;
    }

    v4l2_capability capability{};
    if (xioctl(impl_->fd, VIDIOC_QUERYCAP, &capability) < 0) {
        error = "VIDIOC_QUERYCAP: " + errno_text(errno);
        close_device();
        return false;
    }
    // Важная деталь: у многофункциональных устройств (UVC-камера с микрофоном,
    // TV-тюнер и т.п.) поле `capabilities` описывает ВСЁ устройство целиком, а
    // реальные возможности конкретного узла /dev/videoX лежат в `device_caps`.
    // Если проверять только `capabilities`, часть камер «не поддерживает захват»
    // по ошибке. Поэтому при выставленном V4L2_CAP_DEVICE_CAPS берём device_caps.
    const std::uint32_t caps =
        (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0 ? capability.device_caps
                                                             : capability.capabilities;

    // Одноплоскостной API (VIDIOC_*BUFFER, `fmt.pix`) работает только с
    // V4L2_CAP_VIDEO_CAPTURE. Устройства только с MPLANE требуют другого набора
    // структур — это отдельная тема, здесь честно сообщаем об ограничении.
    const bool can_capture = (caps & V4L2_CAP_VIDEO_CAPTURE) != 0;
    const bool can_stream = (caps & V4L2_CAP_STREAMING) != 0;
    if (!can_capture) {
        error = config_.device +
                " не поддерживает V4L2_CAP_VIDEO_CAPTURE (возможно, это только "
                "multi-planar устройство или узел метаданных)";
        LOG_ERROR << "подсказка: v4l2-ctl -d " << config_.device << " --all";
        close_device();
        return false;
    }
    if (!can_stream) {
        error = config_.device + " не поддерживает streaming I/O (VIDIOC_STREAMON)";
        close_device();
        return false;
    }

    LOG_INFO << "камера: " << reinterpret_cast<const char*>(capability.card) << " / "
             << reinterpret_cast<const char*>(capability.driver);

    if (!set_format(error) || !set_frame_rate(error) || !init_buffers(error) ||
        !start_streaming(error)) {
        close_device();
        return false;
    }
    LOG_INFO << "открыт источник: " << describe();
    return true;
}

bool V4l2Source::set_format(std::string& error) {
    // Порядок предпочтений: MJPEG даёт больший fps на USB 2.0 (меньше данных),
    // но добавляет декодирование на CPU; YUYV не требует декодирования.
    std::vector<std::uint32_t> candidates;
    if (config_.prefer_mjpeg) {
        candidates.push_back(V4L2_PIX_FMT_MJPEG);
    }
    candidates.push_back(V4L2_PIX_FMT_YUYV);
    if (!config_.prefer_mjpeg) {
        candidates.push_back(V4L2_PIX_FMT_MJPEG);
    }
    candidates.push_back(V4L2_PIX_FMT_UYVY);

    for (std::uint32_t format : candidates) {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = static_cast<std::uint32_t>(config_.width);
        fmt.fmt.pix.height = static_cast<std::uint32_t>(config_.height);
        fmt.fmt.pix.pixelformat = format;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        if (xioctl(impl_->fd, VIDIOC_S_FMT, &fmt) < 0) {
            continue;  // формат не поддержан — пробуем следующий
        }
        if (fmt.fmt.pix.pixelformat != format) {
            continue;  // драйвер молча подменил формат — не наш случай
        }
        impl_->pixel_format = fmt.fmt.pix.pixelformat;
        impl_->stride = static_cast<int>(fmt.fmt.pix.bytesperline);
        info_.width = static_cast<int>(fmt.fmt.pix.width);
        info_.height = static_cast<int>(fmt.fmt.pix.height);
        info_.pixel_format = fourcc_to_string(impl_->pixel_format);
        if (info_.width != config_.width || info_.height != config_.height) {
            LOG_WARN << "драйвер скорректировал размер: " << config_.width << 'x'
                     << config_.height << " -> " << info_.width << 'x' << info_.height;
        }
        return true;
    }
    error = "не удалось согласовать формат пикселей (пробовали MJPG/YUYV/UYVY)";
    return false;
}

bool V4l2Source::set_frame_rate(std::string& error) {
    (void)error;  // частота не критична: при отказе продолжаем с текущей
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<std::uint32_t>(config_.fps);
    if (xioctl(impl_->fd, VIDIOC_S_PARM, &parm) < 0) {
        // Не фатальна: многие UVC-камеры игнорируют частоту (она задаётся
        // форматом). Продолжаем с тем, что есть.
        info_.fps = config_.fps;
        return true;
    }
    const auto& interval = parm.parm.capture.timeperframe;
    info_.fps = (interval.numerator != 0)
                    ? static_cast<double>(interval.denominator) / interval.numerator
                    : config_.fps;
    if (info_.fps <= 0.0) {
        info_.fps = config_.fps;
    }
    return true;
}

bool V4l2Source::init_buffers(std::string& error) {
    v4l2_requestbuffers request{};
    request.count = static_cast<std::uint32_t>(config_.buffer_count);
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(impl_->fd, VIDIOC_REQBUFS, &request) < 0) {
        error = "VIDIOC_REQBUFS: " + errno_text(errno);
        return false;
    }
    if (request.count < 2) {
        error = "драйвер выделил меньше двух буферов";
        return false;
    }

    buffers_.resize(request.count);
    buffer_lengths_.resize(request.count);
    for (std::uint32_t index = 0; index < request.count; ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (xioctl(impl_->fd, VIDIOC_QUERYBUF, &buffer) < 0) {
            error = "VIDIOC_QUERYBUF: " + errno_text(errno);
            return false;
        }
        void* address = ::mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                               impl_->fd, static_cast<off_t>(buffer.m.offset));
        if (address == MAP_FAILED) {
            error = "mmap: " + errno_text(errno);
            return false;
        }
        buffers_[index] = address;
        buffer_lengths_[index] = buffer.length;

        // Сразу вернуть буфер драйверу — иначе STREAMON не начнёт поток.
        if (xioctl(impl_->fd, VIDIOC_QBUF, &buffer) < 0) {
            error = "VIDIOC_QBUF: " + errno_text(errno);
            return false;
        }
    }
    LOG_DEBUG << "выделено mmap-буферов: " << request.count;
    return true;
}

bool V4l2Source::start_streaming(std::string& error) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(impl_->fd, VIDIOC_STREAMON, &type) < 0) {
        error = "VIDIOC_STREAMON: " + errno_text(errno);
        return false;
    }
    streaming_ = true;
    return true;
}

void V4l2Source::unmap_buffers() {
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
        if (buffers_[i] != nullptr && buffers_[i] != MAP_FAILED) {
            ::munmap(buffers_[i], buffer_lengths_[i]);
        }
    }
    buffers_.clear();
    buffer_lengths_.clear();
}

void V4l2Source::close_device() {
    if (impl_ && impl_->fd >= 0) {
        if (streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(impl_->fd, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        // munmap делаем до close: отображать память закрытого fd нельзя.
        unmap_buffers();
        ::close(impl_->fd);
        impl_->fd = -1;
    } else {
        unmap_buffers();
    }
}

NextStatus V4l2Source::next(FramePtr& out, std::string& error) {
    if (impl_->fd < 0) {
        error = "устройство не открыто";
        return NextStatus::Error;
    }

    // poll — это и есть ожидание кадра: поток спит в ядре, не занимая CPU.
    pollfd descriptor{};
    descriptor.fd = impl_->fd;
    descriptor.events = POLLIN;
    const int ready = ::poll(&descriptor, 1, config_.poll_timeout_ms);
    if (ready == 0) {
        return NextStatus::Timeout;
    }
    if (ready < 0) {
        if (errno == EINTR) {
            return NextStatus::Timeout;  // сигнал (Ctrl+C) — вернёмся в цикл
        }
        error = "poll: " + errno_text(errno);
        return NextStatus::Error;
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        error = "устройство отключено или недоступно";
        return NextStatus::Error;
    }

    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (xioctl(impl_->fd, VIDIOC_DQBUF, &buffer) < 0) {
        if (errno == EAGAIN) {
            return NextStatus::Timeout;
        }
        error = "VIDIOC_DQBUF: " + errno_text(errno);
        return NextStatus::Error;
    }

    // bytesused == 0 бывает при «пустом» кадре (например, пропуск при смене
    // экспозиции) — конвертировать нечего, но буфер вернуть всё равно надо.
    if (buffer.bytesused == 0 || buffer.index >= buffers_.size()) {
        if (xioctl(impl_->fd, VIDIOC_QBUF, &buffer) < 0) {
            LOG_ERROR << "VIDIOC_QBUF (пустой кадр): " << errno_text(errno);
        }
        return NextStatus::Timeout;
    }

    // Время «получения буфера от драйвера» фиксируем сразу после DQBUF: тогда
    // разница t_decoded − t_captured честно покажет стоимость декодирования.
    const auto t_captured = SteadyClock::now();
    const std::uint64_t seq = seq_.fetch_add(1, std::memory_order_relaxed);
    const NextStatus status = convert_frame(buffers_[buffer.index], buffer.bytesused, seq,
                                            t_captured, out, error);

    // Буфер обязательно возвращаем драйверу, даже если декодирование упало:
    // иначе через buffer_count кадров камера встанет.
    if (xioctl(impl_->fd, VIDIOC_QBUF, &buffer) < 0) {
        LOG_ERROR << "VIDIOC_QBUF после обработки: " << errno_text(errno);
    }
    return status;
}

NextStatus V4l2Source::convert_frame(const void* data, std::size_t length, std::uint64_t seq,
                                     SteadyClock::time_point t_captured, FramePtr& out,
                                     std::string& error) {
    cv::Mat bgr;

    switch (impl_->pixel_format) {
        case V4L2_PIX_FMT_YUYV: {
            // Оборачиваем память драйвера в cv::Mat без копирования: 2 байта на
            // пиксель (Y0 U Y1 V), step = bytesperline может быть больше width*2
            // из-за выравнивания строк — это надо уважать, иначе картинка "поедет".
            const int step = impl_->stride > 0 ? impl_->stride : info_.width * 2;
            const cv::Mat yuyv(info_.height, info_.width, CV_8UC2, const_cast<void*>(data), step);
            cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
            break;
        }
        case V4L2_PIX_FMT_UYVY: {
            const int step = impl_->stride > 0 ? impl_->stride : info_.width * 2;
            const cv::Mat uyvy(info_.height, info_.width, CV_8UC2, const_cast<void*>(data), step);
            cv::cvtColor(uyvy, bgr, cv::COLOR_YUV2BGR_UYVY);
            break;
        }
        case V4L2_PIX_FMT_MJPEG:
        case V4L2_PIX_FMT_JPEG: {
            // Кадр MJPEG — это самостоятельный JPEG-файл в памяти.
            const cv::Mat encoded(1, static_cast<int>(length), CV_8UC1,
                                  const_cast<void*>(data));
            cv::imdecode(encoded, cv::IMREAD_COLOR, &bgr);
            if (bgr.empty()) {
                error = "не удалось декодировать MJPEG-кадр";
                return NextStatus::Error;
            }
            break;
        }
        case V4L2_PIX_FMT_RGB24: {
            const int step = impl_->stride > 0 ? impl_->stride : info_.width * 3;
            const cv::Mat rgb(info_.height, info_.width, CV_8UC3, const_cast<void*>(data), step);
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            break;
        }
        default:
            error = "неподдерживаемый формат " + info_.pixel_format;
            return NextStatus::Error;
    }

    if (bgr.empty()) {
        error = "пустое изображение после конвертации";
        return NextStatus::Error;
    }

    auto frame = std::make_shared<Frame>();
    frame->image = std::make_shared<const cv::Mat>(std::move(bgr));
    frame->seq = seq;
    frame->t_captured = t_captured;
    frame->t_decoded = SteadyClock::now();
    out = std::move(frame);
    return NextStatus::Frame;
}

}  // namespace vt
