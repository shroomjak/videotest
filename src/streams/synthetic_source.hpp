#pragma once

// Синтетический источник: рисует два движущихся «объекта».
// Нужен ровно для одного: прогнать конвейер и трекер без камеры (отладка,
// демонстрация, смоук-тест). Интерфейс такой же, как у V4l2Source, поэтому
// подменяется одной строкой в main.

#include <chrono>
#include <cstdint>
#include <string>

#include "streams/source.hpp"

namespace vt {

struct SyntheticConfig {
    int width = 640;
    int height = 480;
    int fps = 30;
};

class SyntheticSource final : public Source {
public:
    explicit SyntheticSource(SyntheticConfig config);

    bool open(std::string& error) override;
    bool grab(FramePtr& out, std::string& error) override;
    int width() const override { return config_.width; }
    int height() const override { return config_.height; }
    const std::string& format() const { return name_; }

private:
    SyntheticConfig config_;
    std::string name_ = "SYNTH";
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point next_;
    std::uint64_t next_id_ = 0;
};

}  // namespace vt
