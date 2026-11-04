#pragma once

// Счётчики конвейера. Пишут их рабочие потоки, читает UI-поток.
// Всё, что пересекает границу потоков, — std::atomic: чтение метрик не должно
// требовать блокировок и не должно тормозить обработку кадров.

#include <atomic>
#include <cstdint>

#include "core/frame.hpp"

namespace vt {

struct PipelineStats {
    std::atomic<std::uint64_t> frames_captured{0};   // получено от источника
    std::atomic<std::uint64_t> frames_dropped{0};    // вытеснено из очереди кадров
    std::atomic<std::uint64_t> grab_timeouts{0};     // источник не отдал кадр за таймаут
    std::atomic<std::uint64_t> frames_inferred{0};   // прошло через модель
    std::atomic<std::uint64_t> results_dropped{0};   // вытеснено из очереди результатов
    std::atomic<std::uint64_t> instances_total{0};   // суммарно найденных объектов
    std::atomic<std::int64_t> preprocess_us{0};
    std::atomic<std::int64_t> inference_us{0};
    std::atomic<std::int64_t> postprocess_us{0};
    std::atomic<std::int64_t> latency_us{0};  // от получения буфера до готового результата

    void reset() noexcept {
        frames_captured = 0;
        frames_dropped = 0;
        grab_timeouts = 0;
        frames_inferred = 0;
        results_dropped = 0;
        instances_total = 0;
        preprocess_us = 0;
        inference_us = 0;
        postprocess_us = 0;
        latency_us = 0;
    }
};

// Сглаженный измеритель скорости (кадров в секунду).
//
// Живёт в одном потоке (UI), поэтому без атомиков. Идея: на каждом вызове
// считаем мгновенную скорость между двумя замерами и сглаживаем её
// экспоненциальным средним, чтобы цифра на экране не дёргалась.
class RateMeter {
public:
    explicit RateMeter(double smoothing = 0.2) : smoothing_(smoothing) {}

    double update(std::uint64_t total, SteadyClock::time_point now) {
        if (total < last_total_) {  // счётчик сбросили
            last_total_ = total;
            last_time_ = now;
            return value_;
        }
        const double dt = seconds_between(last_time_, now);
        if (dt > 1e-4) {
            const double instant = static_cast<double>(total - last_total_) / dt;
            value_ = has_value_ ? (value_ * (1.0 - smoothing_) + instant * smoothing_) : instant;
            has_value_ = true;
            last_total_ = total;
            last_time_ = now;
        }
        return value_;
    }

    double value() const noexcept { return value_; }

private:
    double smoothing_;
    std::uint64_t last_total_ = 0;
    SteadyClock::time_point last_time_ = SteadyClock::now();
    double value_ = 0.0;
    bool has_value_ = false;
};

}  // namespace vt
