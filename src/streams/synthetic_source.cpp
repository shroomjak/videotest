#include "streams/synthetic_source.hpp"

#include <cmath>
#include <thread>

#include <opencv2/imgproc.hpp>

namespace vt {

SyntheticSource::SyntheticSource(SyntheticConfig config) : config_(config) {}

bool SyntheticSource::open(std::string& error) {
    error.clear();
    start_ = std::chrono::steady_clock::now();
    next_ = start_;
    return true;
}

bool SyntheticSource::grab(FramePtr& out, std::string& error) {
    error.clear();
    // Держим темп источника, как настоящая камера: иначе конвейер получит
    // тысячи кадров в секунду и очередь будет всё время вытеснять их.
    const double interval = 1.0 / static_cast<double>(config_.fps > 0 ? config_.fps : 30);
    next_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(interval));
    const auto now = std::chrono::steady_clock::now();
    if (next_ > now) {
        std::this_thread::sleep_until(next_);
    } else {
        next_ = now;
    }

    auto image = std::make_shared<cv::Mat>(config_.height, config_.width, CV_8UC3,
                                           cv::Scalar(40, 40, 46));
    const double t = std::chrono::duration<double>(now - start_).count();
    // Два круга с разной скоростью: трекер должен вести два независимых трека.
    const cv::Point a(static_cast<int>(config_.width * (0.5 + 0.32 * std::sin(t * 0.9))),
                      static_cast<int>(config_.height * (0.5 + 0.28 * std::cos(t * 0.7))));
    const cv::Point b(static_cast<int>(config_.width * (0.5 + 0.25 * std::cos(t * 1.4))),
                      static_cast<int>(config_.height * (0.5 + 0.30 * std::sin(t * 1.1))));
    cv::circle(*image, a, config_.height / 8, cv::Scalar(60, 180, 250), cv::FILLED, cv::LINE_AA);
    cv::circle(*image, b, config_.height / 10, cv::Scalar(120, 240, 120), cv::FILLED, cv::LINE_AA);

    auto frame = std::make_shared<Frame>();
    frame->image = std::move(*image);
    frame->id = next_id_++;
    frame->captured = now;
    out = std::move(frame);
    return true;
}

}  // namespace vt
