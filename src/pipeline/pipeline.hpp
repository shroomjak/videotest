#pragma once

// Конвейер реального времени: источник -> очередь -> модель -> очередь -> UI.
//
// Потоки:
//   * capture_loop     — блокируется в poll()/read источника, кладёт frame в capture_queue_;
//   * inference_loop   — берёт самый свежий кадр, считает модель, кладёт результат;
//   * (вызывающий) UI  — забирает результаты из result_queue_ и рисует.
//
// Почему именно так, а не "один поток на всё":
//   камера и модель имеют разную и нестабильную скорость. Если делать всё
//   последовательно, любая задержка инференса начинает тормозить захват кадров:
//   буферы драйвера переполняются, и мы получаем либо потерю кадров с рывками,
//   либо растущую задержку "камера -> экран". Разделение на потоки + очереди
//   фиксированной ёмкости с вытеснением старых кадров даёт постоянную задержку
//   (~1 кадр) и максимальную частоту обновления картинки.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "core/bounded_queue.hpp"
#include "core/frame.hpp"
#include "core/stats.hpp"
#include "infer/detection.hpp"
#include "video/video_source.hpp"

namespace vt {

struct PipelineConfig {
    // Ёмкость 2 означает: «текущий кадр + один на замену». Меньше — начнём терять
    // кадры даже при равных скоростях, больше — вырастет задержка.
    std::size_t capture_queue_capacity = 2;
    std::size_t result_queue_capacity = 2;
    // Искусственная задержка после инференса — учебный инструмент: позволяет
    // увидеть, как ведёт себя система, когда модель медленнее камеры.
    int artificial_infer_delay_ms = 0;
};

class Pipeline {
public:
    Pipeline(SourcePtr source, SegmenterPtr segmenter, PipelineConfig config);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void start();
    void stop();  // идемпотентна, безопасна из деструктора

    bool running() const noexcept { return running_.load(std::memory_order_relaxed); }

    BoundedQueue<FramePtr>& capture_queue() noexcept { return capture_queue_; }
    BoundedQueue<ResultPtr>& result_queue() noexcept { return result_queue_; }
    const BoundedQueue<FramePtr>& capture_queue() const noexcept { return capture_queue_; }
    const BoundedQueue<ResultPtr>& result_queue() const noexcept { return result_queue_; }
    const PipelineStats& stats() const noexcept { return stats_; }
    const IVideoSource& source() const noexcept { return *source_; }
    const ISegmenter& segmenter() const noexcept { return *segmenter_; }

private:
    void capture_loop();
    void inference_loop();

    SourcePtr source_;
    SegmenterPtr segmenter_;
    PipelineConfig config_;

    BoundedQueue<FramePtr> capture_queue_;
    BoundedQueue<ResultPtr> result_queue_;
    PipelineStats stats_;

    std::atomic<bool> running_{false};
    std::thread capture_thread_;
    std::thread inference_thread_;
};

}  // namespace vt
