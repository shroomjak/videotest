#pragma once

// Кадр — единица данных, которая путешествует между потоками конвейера.
//
// Ключевая идея по владению:
//   * cv::Mat внутри shared_ptr<const cv::Mat> — копирование кадра = копирование
//     указателя (O(1)), пиксели не дублируются;
//   * const гарантирует, что ни один потребитель не изменит данные, пока другой
//     поток их читает. Так мы получаем потокобезопасность "по построению",
//     без мьютексов вокруг самих пикселей.

#include <chrono>
#include <cstdint>
#include <memory>

#include <opencv2/core.hpp>

namespace vt {

using SteadyClock = std::chrono::steady_clock;

struct Frame {
    std::shared_ptr<const cv::Mat> image;  // CV_8UC3, BGR
    std::uint64_t seq = 0;                 // монотонный номер кадра от источника
    SteadyClock::time_point t_captured{};  // момент получения буфера от драйвера
    SteadyClock::time_point t_decoded{};   // момент, когда пиксели готовы в cv::Mat
};

using FramePtr = std::shared_ptr<const Frame>;

inline double seconds_between(SteadyClock::time_point a, SteadyClock::time_point b) noexcept {
    return std::chrono::duration<double>(b - a).count();
}

inline double ms_between(SteadyClock::time_point a, SteadyClock::time_point b) noexcept {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace vt
