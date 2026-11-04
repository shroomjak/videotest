#include "render/renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace vt {
namespace {

// Палитра из 20 различимых цветов (BGR), взята из классической схемы для
// семантической сегментации. Дальше цвета повторяются по кругу.
// Обычный const, а не constexpr: cv::Scalar не является literal-типом
// (внутри cv::Matx), поэтому в constexpr он не годится.
const std::array<cv::Scalar, 20> kPalette = {
    cv::Scalar(56, 56, 255),   cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
    cv::Scalar(29, 178, 255),  cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146),  cv::Scalar(134, 219, 61),  cv::Scalar(52, 147, 26),
    cv::Scalar(187, 212, 0),   cv::Scalar(168, 153, 44),  cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52),   cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0),
    cv::Scalar(255, 56, 132),  cv::Scalar(133, 0, 82),    cv::Scalar(255, 56, 203),
    cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};

double text_scale_for(const cv::Mat& canvas) {
    return std::max(0.4, canvas.cols / 1280.0);
}

// Текст с фоном: на светлом кадре белые буквы без подложки не читаются.
void put_label(cv::Mat& canvas, const std::string& text, const cv::Point& origin,
               const cv::Scalar& color, double scale) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    cv::Point top_left(origin.x, origin.y - size.height - 4);
    // Не даём подписи уехать за край кадра.
    if (top_left.y < 0) {
        top_left.y = 0;
    }
    if (top_left.x < 0) {
        top_left.x = 0;
    }
    // Подложка под текст: у ROI берём отдельный Mat, чтобы не задеть пиксели
    // за пределами рамки.
    const cv::Rect box(top_left.x, top_left.y,
                       std::min(size.width + 8, canvas.cols - top_left.x),
                       std::min(size.height + 6, canvas.rows - top_left.y));
    if (box.width > 0 && box.height > 0) {
        cv::rectangle(canvas, box, cv::Scalar(0, 0, 0), cv::FILLED);
    }
    cv::putText(canvas, text, cv::Point(top_left.x + 4, top_left.y + size.height),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

}  // namespace

cv::Scalar Renderer::class_color(int class_id) {
    // Отрицательные/огромные id не должны ломать отрисовку.
    const int index = ((class_id % static_cast<int>(kPalette.size())) +
                       static_cast<int>(kPalette.size())) %
                      static_cast<int>(kPalette.size());
    return kPalette[static_cast<std::size_t>(index)];
}

void Renderer::draw_instance(cv::Mat& canvas, const Instance& instance, const std::string& label,
                             float score) const {
    const cv::Scalar color = class_color(instance.class_id);
    const double scale = text_scale_for(canvas);

    if (options_.show_boxes) {
        const cv::Rect box = instance.box;
        cv::rectangle(canvas, box, color, std::max(1, canvas.cols / 480), cv::LINE_AA);
    }
    if (options_.show_labels) {
        char text[160];
        std::snprintf(text, sizeof(text), "%s %.2f", label.c_str(), score);
        put_label(canvas, text, cv::Point(static_cast<int>(instance.box.x),
                                          static_cast<int>(instance.box.y)),
                  cv::Scalar(255, 255, 255), scale);
    }
}

cv::Mat Renderer::draw(const cv::Mat& bgr, const InferenceResult& result,
                       const std::vector<std::string>& class_names) const {
    cv::Mat canvas = bgr.clone();

    // 1. Маски. Красим все маски в один слой и смешиваем его с кадром один раз:
    //    так альфа-наложение не накапливается там, где объекты перекрываются.
    if (options_.show_masks) {
        cv::Mat layer = canvas.clone();
        bool painted = false;
        for (const auto& instance : result.instances) {
            if (instance.score < options_.score_threshold || instance.mask.empty()) {
                continue;
            }
            layer.setTo(class_color(instance.class_id), instance.mask);
            painted = true;
        }
        if (painted) {
            cv::addWeighted(canvas, 1.0 - options_.mask_alpha, layer, options_.mask_alpha, 0.0,
                            canvas);
        }
    }

    // 2. Контуры масок — чёткая граница объекта поверх заливки.
    if (options_.show_contours) {
        for (const auto& instance : result.instances) {
            if (instance.mask.empty() || instance.score < options_.score_threshold) {
                continue;
            }
            std::vector<std::vector<cv::Point>> contours;
            // findContours портит вход, поэтому берём копию маски.
            cv::findContours(instance.mask.clone(), contours, cv::RETR_EXTERNAL,
                             cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(canvas, contours, -1, class_color(instance.class_id),
                             std::max(1, canvas.cols / 640), cv::LINE_AA);
        }
    }

    // 3. Боксы и подписи.
    for (const auto& instance : result.instances) {
        if (instance.score < options_.score_threshold) {
            continue;
        }
        const std::string name =
            (instance.class_id >= 0 && static_cast<std::size_t>(instance.class_id) < class_names.size())
                ? class_names[static_cast<std::size_t>(instance.class_id)]
                : ("class_" + std::to_string(instance.class_id));
        draw_instance(canvas, instance, name, instance.score);
    }
    return canvas;
}

cv::Mat Renderer::compose(const cv::Mat& raw, const cv::Mat& overlay, Layout layout) const {
    const cv::Size size = raw.size();
    cv::Mat canvas;
    if (layout == Layout::SideBySide) {
        // Горизонтальная склейка: hconcat требует одинаковый тип и высоту.
        cv::Mat right = overlay;
        if (overlay.size() != size) {
            cv::resize(overlay, right, size);
        }
        cv::hconcat(raw, right, canvas);
    } else if (layout == Layout::Stacked) {
        cv::Mat right = overlay;
        if (overlay.size() != size) {
            cv::resize(overlay, right, size);
        }
        cv::vconcat(raw, right, canvas);
    } else {
        canvas = overlay.clone();
    }

    if (layout != Layout::OverlayOnly) {
        // Разделитель и подписи половинок.
        if (layout == Layout::SideBySide) {
            cv::line(canvas, cv::Point(size.width, 0), cv::Point(size.width, canvas.rows),
                     cv::Scalar(200, 200, 200), 2);
        } else {
            cv::line(canvas, cv::Point(0, size.height), cv::Point(canvas.cols, size.height),
                     cv::Scalar(200, 200, 200), 2);
        }
        // Подписи половин ставим в НИЖНИЙ левый угол каждой половины: сверху
        // слева живёт HUD, и подписи не должны с ним пересекаться. Нижний угол
        // обычно свободен (объекты редко занимают самый низ кадра).
        const double scale = text_scale_for(canvas);
        const int label_y = canvas.rows - 8;
        if (layout == Layout::SideBySide) {
            put_label(canvas, "CAMERA / raw", cv::Point(8, label_y), cv::Scalar(230, 230, 230),
                      scale);
            put_label(canvas, "INSTANCE SEGMENTATION",
                      cv::Point(size.width + 8, label_y), cv::Scalar(230, 230, 230), scale);
        } else {
            put_label(canvas, "CAMERA / raw", cv::Point(8, size.height - 8),
                      cv::Scalar(230, 230, 230), scale);
            put_label(canvas, "INSTANCE SEGMENTATION", cv::Point(8, label_y),
                      cv::Scalar(230, 230, 230), scale);
        }
    }
    return canvas;
}

void Renderer::draw_hud(cv::Mat& canvas, const std::vector<std::string>& lines) {
    if (lines.empty()) {
        return;
    }
    // Масштаб ограничен сверху: на очень широкой мозаике (side-by-side двух
    // 1280-х кадров) пропорциональный шрифт вырос бы настолько, что HUD занял
    // бы треть экрана.
    const double scale = std::clamp(canvas.cols / 1900.0, 0.45, 0.9);
    const int line_height = static_cast<int>(24 * scale) + 4;
    int max_width = 0;
    for (const auto& line : lines) {
        const cv::Size size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, scale, 1, nullptr);
        max_width = std::max(max_width, size.width);
    }
    const cv::Rect panel(0, 0, std::min(canvas.cols, max_width + 20),
                         std::min(canvas.rows, line_height * static_cast<int>(lines.size()) + 14));
    if (panel.width <= 0 || panel.height <= 0) {
        return;
    }
    // Полупрозрачная чёрная подложка: умножаем ROI на 0.45.
    cv::Mat roi = canvas(panel);
    roi.convertTo(roi, -1, 0.45, 0.0);

    int y = static_cast<int>(line_height * 1.1);
    for (const auto& line : lines) {
        cv::putText(canvas, line, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                    cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        y += line_height;
    }
}

cv::Mat Renderer::placeholder(const cv::Size& size, const std::string& text) {
    cv::Mat canvas(size, CV_8UC3, cv::Scalar(28, 28, 32));
    const double scale = std::max(0.6, size.width / 900.0);
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    const cv::Point origin((size.width - text_size.width) / 2,
                           (size.height + text_size.height) / 2);
    cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(180, 180, 180), 1,
                cv::LINE_AA);
    return canvas;
}

}  // namespace vt
