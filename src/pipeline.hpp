#pragma once

// Конвейер реального времени из двух потоков:
//
//   поток 1 (захват):  камера -> Queue<FramePtr>(2) ->
//   поток 2 (обработка): decode/модель/трекер -> Queue<ResultPtr>(2) ->
//   main-поток: рисует два потока рядом + HUD, читает клавиши.
//
// Очереди ёмкостью 2 нужны, чтобы камера никогда не ждала модель: если модель
// не успевает, старая картинка вытесняется новой, и задержка не растёт.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "bounded_queue.hpp"
#include "frame.hpp"
#include "model/segmenter.hpp"
#include "streams/source.hpp"
#include "track/tracker.hpp"

namespace vt {

struct PipelineStats {
    std::atomic<std::uint64_t> captured{0};   // кадров получено от источника
    std::atomic<std::uint64_t> dropped{0};    // вытеснено из очереди кадров
    std::atomic<std::uint64_t> processed{0};  // кадров с готовой разметкой
};

class Pipeline {
public:
    Pipeline(Source& source, Segmenter& model, Tracker& tracker);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void start();
    void stop();

    Queue<ResultPtr>& results() { return results_; }
    const PipelineStats& stats() const { return stats_; }
    bool running() const { return running_.load(std::memory_order_relaxed); }

private:
    void capture_loop();
    void process_loop();

    Source& source_;
    Segmenter& model_;
    Tracker& tracker_;

    Queue<FramePtr> frames_{2};
    Queue<ResultPtr> results_{2};
    PipelineStats stats_;

    std::atomic<bool> running_{false};
    std::thread capture_thread_;
    std::thread process_thread_;
};

}  // namespace vt
