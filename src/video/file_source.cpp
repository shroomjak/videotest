#include "video/file_source.hpp"

#include <cmath>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "core/logger.hpp"

namespace vt {

// ---------------------------------------------------------------- FileSource

FileSource::FileSource(std::string path, bool loop, bool pace_realtime)
    : path_(std::move(path)), loop_(loop), pace_realtime_(pace_realtime) {}

bool FileSource::open(std::string& error) {
    if (!capture_.open(path_)) {
        error = "не удалось открыть видеофайл: " + path_;
        return false;
    }
    info_.width = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.fps = capture_.get(cv::CAP_PROP_FPS);
    if (!(info_.fps > 1.0) || info_.fps > 240.0) {
        info_.fps = 30.0;  // у некоторых контейнеров fps отсутствует
    }
    info_.pixel_format = "BGR";
    start_ = SteadyClock::now();
    LOG_INFO << "открыт файл: " << describe();
    return true;
}

std::string FileSource::describe() const {
    std::ostringstream out;
    out << path_ << " (" << info_.width << 'x' << info_.height << " @ " << info_.fps << " fps"
        << (loop_ ? ", зациклен" : "") << ')';
    return out.str();
}

NextStatus FileSource::next(FramePtr& out, std::string& error) {
    if (!capture_.read(scratch_) || scratch_.empty()) {
        if (loop_) {
            capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!capture_.read(scratch_) || scratch_.empty()) {
                error = "файл закончился и не перемотался";
                return NextStatus::EndOfStream;
            }
        } else {
            return NextStatus::EndOfStream;
        }
    }

    // Держим темп источника: иначе файл "пробежит" за миллисекунды и мы получим
    // unrealistic-нагрузку на конвейер.
    if (pace_realtime_ && info_.fps > 0.0) {
        const auto interval = std::chrono::duration<double>(1.0 / info_.fps);
        const auto now = SteadyClock::now();
        if (next_deadline_ < start_) {
            next_deadline_ = now;
        }
        next_deadline_ += std::chrono::duration_cast<SteadyClock::duration>(interval);
        if (next_deadline_ > now) {
            std::this_thread::sleep_until(next_deadline_);
        } else {
            next_deadline_ = now;  // отстали — не копим долг
        }
    }

    auto frame = std::make_shared<Frame>();
    // Копия нужна: следующий read() перезапишет scratch_. Дальше кадр живёт в
    // shared_ptr<const cv::Mat> и между потоками уже не копируется.
    frame->image = std::make_shared<const cv::Mat>(scratch_.clone());
    frame->seq = seq_.fetch_add(1, std::memory_order_relaxed);
    frame->t_captured = SteadyClock::now();
    frame->t_decoded = frame->t_captured;
    ++frames_emitted_;
    out = std::move(frame);
    return NextStatus::Frame;
}

// ----------------------------------------------------------- SyntheticSource

SyntheticSource::SyntheticSource(int width, int height, double fps) {
    info_.width = width;
    info_.height = height;
    info_.fps = fps;
    info_.pixel_format = "BGR";
}

bool SyntheticSource::open(std::string& error) {
    (void)error;
    start_ = SteadyClock::now();
    next_deadline_ = start_;
    LOG_INFO << "открыт синтетический источник: " << describe();
    return true;
}

std::string SyntheticSource::describe() const {
    std::ostringstream out;
    out << "synthetic (" << info_.width << 'x' << info_.height << " @ " << info_.fps << " fps)";
    return out.str();
}

NextStatus SyntheticSource::next(FramePtr& out, std::string& error) {
    (void)error;
    const auto now = SteadyClock::now();
    const double interval = 1.0 / (info_.fps > 0 ? info_.fps : 30.0);
    next_deadline_ += std::chrono::duration_cast<SteadyClock::duration>(
        std::chrono::duration<double>(interval));
    if (next_deadline_ > now) {
        std::this_thread::sleep_until(next_deadline_);
    } else {
        next_deadline_ = now;
    }

    // Рисуем два "объекта", которые двигаются по синусоидам: удобно проверять,
    // что разметка совпадает с картинкой по координатам и по времени.
    auto canvas = std::make_shared<cv::Mat>(info_.height, info_.width, CV_8UC3,
                                            cv::Scalar(38, 38, 44));
    const double t = seconds_between(start_, SteadyClock::now());
    const cv::Point a(static_cast<int>(info_.width * (0.5 + 0.35 * std::sin(t * 1.1))),
                      static_cast<int>(info_.height * (0.5 + 0.30 * std::cos(t * 0.9))));
    const cv::Point b(static_cast<int>(info_.width * (0.5 + 0.28 * std::cos(t * 0.6))),
                      static_cast<int>(info_.height * (0.5 + 0.32 * std::sin(t * 1.4))));
    cv::circle(*canvas, a, info_.height / 7, cv::Scalar(80, 190, 255), cv::FILLED, cv::LINE_AA);
    cv::circle(*canvas, b, info_.height / 9, cv::Scalar(120, 255, 140), cv::FILLED, cv::LINE_AA);
    cv::putText(*canvas, "synthetic source", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(230, 230, 230), 2, cv::LINE_AA);

    auto frame = std::make_shared<Frame>();
    frame->image = std::move(canvas);
    frame->seq = seq_.fetch_add(1, std::memory_order_relaxed);
    frame->t_captured = SteadyClock::now();
    frame->t_decoded = frame->t_captured;
    out = std::move(frame);
    return NextStatus::Frame;
}

}  // namespace vt
