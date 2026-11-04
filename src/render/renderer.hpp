#pragma once

// Визуализация: "сырой" поток камеры рядом с потоком, размеченным масками
// instance-сегментации.
//
// Рендерер сознательно отделён от конвейера: он работает в UI-потоке и знает
// только о InferenceResult. Это позволяет менять стиль отрисовки, не трогая
// нити обработки, и наоборот — тестировать рендеринг без камеры.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "infer/detection.hpp"

namespace vt {

enum class Layout {
    SideBySide,  // два окна рядом: камера | сегментация
    Stacked,     // друг под другом
    OverlayOnly, // только размеченный поток
};

struct RenderOptions {
    bool show_masks = true;
    bool show_boxes = true;
    bool show_labels = true;
    bool show_contours = true;
    float mask_alpha = 0.45f;
    float score_threshold = 0.30f;
};

class Renderer {
public:
    explicit Renderer(RenderOptions options) : options_(options) {}

    // Рисует маски/боксы/подписи поверх кадра. Кадр НЕ модифицируется: метод
    // делает копию (нужно для side-by-side, где оригинал идёт рядом).
    cv::Mat draw(const cv::Mat& bgr, const InferenceResult& result,
                 const std::vector<std::string>& class_names) const;

    // Склеивает два потока в один кадр-«мозаику» и подписывает половинки.
    cv::Mat compose(const cv::Mat& raw, const cv::Mat& overlay, Layout layout) const;

    // Полупрозрачная панель с метриками в левом верхнем углу.
    static void draw_hud(cv::Mat& canvas, const std::vector<std::string>& lines);

    // Заглушка-кадр на случай, когда данных ещё/уже нет.
    static cv::Mat placeholder(const cv::Size& size, const std::string& text);

    // Устойчивый цвет для класса: один и тот же класс всегда одного цвета.
    static cv::Scalar class_color(int class_id);

private:
    void draw_instance(cv::Mat& canvas, const Instance& instance, const std::string& label,
                       float score) const;

    RenderOptions options_;
};

}  // namespace vt
