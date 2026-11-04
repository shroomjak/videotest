#include "pipeline/pipeline.hpp"

#include <chrono>
#include <thread>
#include <utility>

#include "core/logger.hpp"
#include "core/signal_flag.hpp"

namespace vt {
namespace {

FramePtr nothing;

}  // namespace

Pipeline::Pipeline(SourcePtr source, SegmenterPtr segmenter, PipelineConfig config)
    : source_(std::move(source)),
      segmenter_(std::move(segmenter)),
      config_(config),
      capture_queue_(config.capture_queue_capacity, OverflowPolicy::DropOldest),
      result_queue_(config.result_queue_capacity, OverflowPolicy::DropOldest) {}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (running_.exchange(true)) {
        return;
    }
    capture_thread_ = std::thread([this] { capture_loop(); });
    inference_thread_ = std::thread([this] { inference_loop(); });
    LOG_INFO << "конвейер запущен: камера -> модель -> отображение";
}

void Pipeline::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    // Порядок важен: сначала закрываем входную очередь (потребитель увидит
    // пустую закрытую очередь и выйдет), затем ждём потоки.
    capture_queue_.close();
    result_queue_.close();
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    LOG_INFO << "конвейер остановлен";
}

void Pipeline::capture_loop() {
    // Счётчик подряд идущих ошибок: одиночный сбой (например, камера
    // переоткрылась) не должен убивать поток, но бесконечный цикл ошибок — тоже
    // плохо. После N подряд — останавливаемся, чтобы не жечь CPU.
    int consecutive_errors = 0;
    constexpr int kMaxConsecutiveErrors = 30;

    while (running_.load(std::memory_order_relaxed) && !stop_requested()) {
        FramePtr frame;
        std::string error;
        const NextStatus status = source_->next(frame, error);

        switch (status) {
            case NextStatus::Frame: {
                consecutive_errors = 0;
                stats_.frames_captured.fetch_add(1, std::memory_order_relaxed);
                const std::uint64_t before = capture_queue_.dropped();
                if (!capture_queue_.push(std::move(frame))) {
                    break;  // очередь закрыта — выходим по условию цикла
                }
                // DropOldest вытесняет старый кадр: обновляем счётчик потерь.
                const std::uint64_t after = capture_queue_.dropped();
                if (after != before) {
                    stats_.frames_dropped.fetch_add(after - before, std::memory_order_relaxed);
                }
                break;
            }
            case NextStatus::Timeout:
                stats_.grab_timeouts.fetch_add(1, std::memory_order_relaxed);
                break;
            case NextStatus::EndOfStream:
                LOG_INFO << "источник закончился";
                request_stop();
                break;
            case NextStatus::Error: {
                ++consecutive_errors;
                LOG_ERROR << "источник: " << error << " (ошибка " << consecutive_errors << '/'
                          << kMaxConsecutiveErrors << ')';
                // Небольшая пауза, чтобы не залить лог при мгновенных ошибках.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (consecutive_errors >= kMaxConsecutiveErrors) {
                    LOG_ERROR << "слишком много ошибок подряд — останавливаем конвейер";
                    request_stop();
                }
                break;
            }
        }
    }
    // Если вышли не по Ctrl+C, всё равно будим потребителя.
    capture_queue_.close();
}

void Pipeline::inference_loop() {
    // Ошибки модели считаем подряд: одна плохая камера/кадр не должна убивать
    // конвейер, но вечный цикл «ошибка — пропуск кадра» бесполезен и заливает лог.
    int consecutive_errors = 0;
    constexpr int kMaxConsecutiveErrors = 20;

    while (running_.load(std::memory_order_relaxed) && !stop_requested()) {
        // Блокирующее ожидание с таймаутом: так поток просыпается, чтобы
        // проверить флаг остановки, даже если кадры не приходят.
        auto maybe_frame = capture_queue_.pop_wait_for(std::chrono::milliseconds(100));
        if (!maybe_frame) {
            if (capture_queue_.is_closed()) {
                break;
            }
            continue;
        }
        const FramePtr& frame = *maybe_frame;

        std::vector<Instance> instances;
        std::string error;
        const auto t_start = SteadyClock::now();
        const bool ok = segmenter_->segment(*frame->image, instances, error);
        const auto t_end = SteadyClock::now();

        if (!ok) {
            ++consecutive_errors;
            LOG_ERROR << "инференс: " << error << " (ошибка " << consecutive_errors << '/'
                      << kMaxConsecutiveErrors << ')';
            if (consecutive_errors >= kMaxConsecutiveErrors) {
                LOG_ERROR << "модель стабильно возвращает ошибку — останавливаем конвейер";
                request_stop();
                break;
            }
            continue;
        }
        consecutive_errors = 0;

        auto result = std::make_shared<InferenceResult>();
        result->frame = frame;
        result->instances = std::move(instances);
        result->input_width = segmenter_->info().input_width;
        result->input_height = segmenter_->info().input_height;
        result->timings = segmenter_->last_timings();
        result->timings.inference_ms = result->timings.inference_ms > 0.0
                                           ? result->timings.inference_ms
                                           : ms_between(t_start, t_end);
        result->latency_ms = ms_between(frame->t_captured, t_end);

        // Опциональная задержка: показывает, что будет, если модель медленнее
        // камеры — очередь кадров начнёт вытеснять старые кадры.
        if (config_.artificial_infer_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.artificial_infer_delay_ms));
        }

        stats_.frames_inferred.fetch_add(1, std::memory_order_relaxed);
        stats_.instances_total.fetch_add(result->instances.size(), std::memory_order_relaxed);
        stats_.inference_us.store(static_cast<std::int64_t>(result->timings.inference_ms * 1000.0),
                                  std::memory_order_relaxed);
        stats_.latency_us.store(static_cast<std::int64_t>(result->latency_ms * 1000.0),
                                std::memory_order_relaxed);

        const std::uint64_t before = result_queue_.dropped();
        result_queue_.push(std::move(result));
        const std::uint64_t after = result_queue_.dropped();
        if (after != before) {
            stats_.results_dropped.fetch_add(after - before, std::memory_order_relaxed);
        }
    }
    result_queue_.close();
}

}  // namespace vt
