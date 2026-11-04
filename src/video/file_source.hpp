#pragma once

// Источники без камеры:
//   * FileSource      — видеофайл/картинка через cv::VideoCapture, с
//                       "проигрыванием" в темпе fps и опциональным зацикливанием;
//   * SyntheticSource — генератор движущейся картинки (шары на фоне), который
//                       работает всегда и удобен для отладки конвейера и модели.
//
// Оба класса — хороший пример того, зачем нужен IVideoSource: конвейер,
// модель и визуализация не меняются ни на строку.

#include <atomic>
#include <cstdint>
#include <string>

#include <opencv2/videoio.hpp>

#include "video/video_source.hpp"

namespace vt {

class FileSource final : public IVideoSource {
public:
    FileSource(std::string path, bool loop, bool pace_realtime);

    bool open(std::string& error) override;
    NextStatus next(FramePtr& out, std::string& error) override;
    const SourceInfo& info() const override { return info_; }
    std::string describe() const override;

private:
    std::string path_;
    bool loop_;
    bool pace_realtime_;
    SourceInfo info_;
    cv::VideoCapture capture_;
    cv::Mat scratch_;  // переиспользуемый буфер декодирования (без аллокаций на кадр)
    std::atomic<std::uint64_t> seq_{0};
    SteadyClock::time_point start_{};
    SteadyClock::time_point next_deadline_{};
    std::uint64_t frames_emitted_ = 0;
};

class SyntheticSource final : public IVideoSource {
public:
    SyntheticSource(int width, int height, double fps);

    bool open(std::string& error) override;
    NextStatus next(FramePtr& out, std::string& error) override;
    const SourceInfo& info() const override { return info_; }
    std::string describe() const override;

private:
    SourceInfo info_;
    SteadyClock::time_point start_{};
    SteadyClock::time_point next_deadline_{};
    std::atomic<std::uint64_t> seq_{0};
};

}  // namespace vt
