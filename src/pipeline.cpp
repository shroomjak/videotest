#include "pipeline.hpp"

#include <chrono>
#include <utility>

#include "logger.hpp"
#include "stop_flag.hpp"

namespace vt {

Pipeline::Pipeline(Source& source, Segmenter& model, Tracker& tracker)
    : source_(source), model_(model), tracker_(tracker) {}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (running_.exchange(true)) {
        return;
    }
    capture_thread_ = std::thread([this] { capture_loop(); });
    process_thread_ = std::thread([this] { process_loop(); });
    LOG_INFO << "конвейер запущен: источник -> очередь -> модель -> трекер -> очередь";
}

void Pipeline::stop() {
    // Порядок важен: сначала снимаем флаг, потом join. Иначе рабочие циклы
    // (которые проверяют running_) не увидят остановку, join будет ждать
    // вечно, а поток, у которого цикл завершился сам, оставит joinable()
    // поток с уже завершённой функцией.
    running_.store(false, std::memory_order_relaxed);
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (process_thread_.joinable()) {
        process_thread_.join();
    }
}

void Pipeline::capture_loop() {
    // Ошибки подряд: одиночный сбой не должен убивать конвейер, но вечный
    // цикл ошибок бесполезен — после 30 подряд останавливаемся.
    int errors = 0;
    while (running_.load() && !stop_requested()) {
        FramePtr frame;
        std::string error;
        if (!source_.grab(frame, error)) {
            if (++errors >= 30) {
                LOG_ERROR << "источник: " << error << " — останавливаем конвейер";
                break;
            }
            LOG_WARN << "источник: " << error;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        errors = 0;
        if (!frame) {
            continue;  // таймаут: кадра пока нет
        }
        stats_.captured.fetch_add(1, std::memory_order_relaxed);
        const std::size_t dropped = frames_.push(std::move(frame));
        if (dropped > 0) {
            stats_.dropped.fetch_add(dropped, std::memory_order_relaxed);
        }
    }
}

void Pipeline::process_loop() {
    int errors = 0;
    while (running_.load() && !stop_requested()) {
        auto maybe_frame = frames_.pop(std::chrono::milliseconds(100));
        if (!maybe_frame) {
            continue;
        }
        FramePtr frame = std::move(*maybe_frame);

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Instance> instances;
        double infer_ms = 0.0;
        std::string error;
        if (!model_.infer(frame->image, instances, &infer_ms, error)) {
            if (++errors >= 20) {
                LOG_ERROR << "модель: " << error << " — останавливаем конвейер";
                break;
            }
            LOG_ERROR << "модель: " << error;
            continue;
        }
        errors = 0;

        // Трекинг: проставляем track_id. На пустой разметке трекер тоже
        // вызывается (instances пуст) — просто старит существующие треки.
        tracker_.update(instances);

        auto result = std::make_shared<Result>();
        result->frame = std::move(frame);
        result->instances = std::move(instances);
        result->infer_ms = infer_ms;
        result->latency_ms = ms_between(result->frame->captured,
                                        std::chrono::steady_clock::now());
        stats_.processed.fetch_add(1, std::memory_order_relaxed);
        (void)t0;
        results_.push(std::move(result));
    }
}

}  // namespace vt
