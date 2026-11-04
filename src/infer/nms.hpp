#pragma once

// Non-Maximum Suppression (NMS) и IoU — вынесены в отдельный заголовок, потому
// что это чистая математика без зависимостей: её легко читать, тестировать и
// переиспользовать (например, в трекере, где IoU нужен для сопоставления
// объектов между кадрами).
//
// Зачем NMS вообще: сырой выход YOLO — это 8400 «кандидатов» на трёх уровнях
// сетки. Одна и та же собака даёт десятки перекрывающихся боксов, и после
// порога по уверенности нужно оставить по одному на объект.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "infer/detection.hpp"

namespace vt {

// IoU = площадь пересечения / площадь объединения.
inline float box_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter_w = std::max(0.0f, x2 - x1);
    const float inter_h = std::max(0.0f, y2 - y1);
    const float intersection = inter_w * inter_h;
    const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    const float uni = area_a + area_b - intersection;
    return uni > 0.0f ? intersection / uni : 0.0f;
}

// Greedy class-aware NMS.
//
// Алгоритм:
//   1. отсортировать индексы по убыванию уверенности;
//   2. взять самый уверенный непогашенный бокс — он остаётся;
//   3. погасить все следующие боксы того же класса с IoU > threshold;
//   4. повторять до исчерпания или до max_detections.
//
// «Class-aware» — принципиально: объекты разных классов часто перекрываются
// (человек держит чашку), и подавление по всем классам склеило бы их в один.
//
// detections и coeffs перемещаются синхронно: коэффициенты масок должны
// остаться привязанными к своим боксам.
inline void class_aware_nms(std::vector<Instance>& detections,
                            std::vector<std::vector<float>>& coeffs, float iou_threshold,
                            int max_detections) {
    const std::size_t count = detections.size();
    if (count < 2 || max_detections <= 0) {
        return;
    }

    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return detections[a].score > detections[b].score;
    });

    std::vector<char> suppressed(count, 0);
    std::vector<Instance> kept;
    std::vector<std::vector<float>> kept_coeffs;
    const std::size_t limit = std::min<std::size_t>(count, static_cast<std::size_t>(max_detections));
    kept.reserve(limit);
    kept_coeffs.reserve(limit);

    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t index = order[i];
        if (suppressed[index] != 0) {
            continue;
        }
        kept.push_back(std::move(detections[index]));
        kept_coeffs.push_back(std::move(coeffs[index]));
        if (kept.size() >= limit) {
            break;
        }
        const Instance& keeper = kept.back();
        for (std::size_t k = i + 1; k < count; ++k) {
            const std::size_t other = order[k];
            if (suppressed[other] != 0 || detections[other].class_id != keeper.class_id) {
                continue;
            }
            if (box_iou(keeper.box, detections[other].box) > iou_threshold) {
                suppressed[other] = 1;
            }
        }
    }
    detections = std::move(kept);
    coeffs = std::move(kept_coeffs);
}

}  // namespace vt
