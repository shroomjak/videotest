#pragma once

// Визуализация: два потока рядом — исходный кадр и тот же кадр с разметкой.
// Если instances пуст (в том числе когда модели нет), правая половина
// побайтно совпадает с левой: отдельных режимов «без модели» не нужно.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "frame.hpp"

namespace vt {

struct ViewOptions {
    float mask_alpha = 0.45f;  // прозрачность заливки маски
    bool show_masks = true;
    bool show_boxes = true;
};

// Рисует маску/бокс/подпись с track_id поверх копии кадра.
cv::Mat draw_overlay(const cv::Mat& bgr, const std::vector<Instance>& instances,
                     const ViewOptions& options);

// Склеивает два кадра в [исходный | размеченный] с разделителем и подписями.
cv::Mat compose_side_by_side(const cv::Mat& raw, const cv::Mat& overlay);

// Строка-панель со счётчиками (fps, задержка, потери, число треков).
void draw_hud(cv::Mat& canvas, const std::vector<std::string>& lines);

// Устойчивый цвет класса (BGR), чтобы один класс всегда рисовался одним цветом.
cv::Scalar class_color(int class_id);

}  // namespace vt
