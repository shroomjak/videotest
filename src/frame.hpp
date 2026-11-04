#pragma once

// Кадр и результат разметки — данные, которые ходят между потоками конвейера.

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace vt {

// Один найденный объект. track_id присваивает трекер; -1 — ещё не присвоен.
struct Instance {
    cv::Rect box;         // целые пиксели: быстрее и достаточно для рисования
    float score = 0.0f;
    int class_id = 0;
    int track_id = -1;
    cv::Mat mask;         // CV_8UC1 0/255 в размере кадра; может быть пустой
};

// Кадр от источника. cv::Mat копируется как дескриптор (счётчик ссылок),
// а shared_ptr<const Frame> гарантирует, что никто не изменит пиксели, пока
// их читает другой поток. Ключевой приём всей потокобезопасности проекта.
struct Frame {
    cv::Mat image;                                  // BGR, CV_8UC3
    std::uint64_t id = 0;                           // номер кадра от источника
    std::chrono::steady_clock::time_point captured; // момент получения от драйвера
};

// Результат обработки кадра: та же картинка + разметка.
// instances пустой => «пустая разметка»: размеченный поток совпадает с исходным.
struct Result {
    std::shared_ptr<const Frame> frame;
    std::vector<Instance> instances;
    double infer_ms = 0.0;
    double latency_ms = 0.0;
};

using FramePtr = std::shared_ptr<Frame>;
using ResultPtr = std::shared_ptr<Result>;

inline double ms_between(std::chrono::steady_clock::time_point a,
                         std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace vt
