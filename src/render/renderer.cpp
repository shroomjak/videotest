#include "render/renderer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>

#include <opencv2/imgproc.hpp>

#include "model/class_names.hpp"

namespace vt {
namespace {

// 20 различимых цветов BGR; дальше повторяются по кругу.
const std::array<cv::Scalar, 20> kPalette = {
    cv::Scalar(56, 56, 255),   cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
    cv::Scalar(29, 178, 255),  cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146),  cv::Scalar(134, 219, 61),  cv::Scalar(52, 147, 26),
    cv::Scalar(187, 212, 0),   cv::Scalar(168, 153, 44),  cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52),   cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0),
    cv::Scalar(255, 56, 132),  cv::Scalar(133, 0, 82),    cv::Scalar(255, 56, 203),
    cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};

// 0.82 -> "0.82" без форматирования потока (snprintf дешевле и без iostream).
std::string score_text(float score) {
    char buffer[8];
    std::snprintf(buffer, sizeof(buffer), "%.2f", static_cast<double>(score));
    return buffer;
}

// Текст с чёрной подложкой: на светлом кадре белые буквы иначе не читаются.
void put_label(cv::Mat& canvas, const std::string& text, cv::Point origin,
               const cv::Scalar& color, double scale) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    cv::Point top(std::max(0, origin.x), std::max(0, origin.y - size.height - 4));
    const cv::Rect box(top.x, top.y, std::min(size.width + 8, canvas.cols - top.x),
                       std::min(size.height + 6, canvas.rows - top.y));
    if (box.width <= 0 || box.height <= 0) {
        return;
    }
    cv::rectangle(canvas, box, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, text, cv::Point(top.x + 4, top.y + size.height),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

}  // namespace

cv::Scalar class_color(int class_id) {
    const int count = static_cast<int>(kPalette.size());
    const int index = ((class_id % count) + count) % count;
    return kPalette[static_cast<std::size_t>(index)];
}

cv::Mat draw_overlay(const cv::Mat& bgr, const std::vector<Instance>& instances,
                     const ViewOptions& options) {
    cv::Mat canvas = bgr.clone();
    const double scale = std::max(0.4, canvas.cols / 1600.0);
    const int thickness = std::max(1, canvas.cols / 640);

    // 1. Маски: красим все в один слой и смешиваем его с кадром ОДИН раз —
    //    иначе полупрозрачные заливки накладываются друг на друга и перекрытия
    //    становятся темнее (альфа складывается).
    //
    //    ВАЖНО про производительность: маска хранится в размере ВСЕГО кадра, но
    //    ненулевые пиксели у неё только внутри рамки объекта. Поэтому все
    //    операции делаются над ROI (прямоугольником рамки), а не над кадром:
    //    при 1280x720 и 300 объектах разница — сотни миллисекунд на кадр.
    if (options.show_masks) {
        cv::Mat layer = canvas.clone();
        bool painted = false;
        for (const auto& instance : instances) {
            const cv::Rect roi = instance.mask.empty() ? cv::Rect()
                                                       : instance.box & cv::Rect(0, 0, canvas.cols, canvas.rows);
            if (roi.width <= 0 || roi.height <= 0) {
                continue;
            }
            // Работаем только внутри рамки: и маска, и слой режутся одним ROI.
            cv::Mat mask_roi = instance.mask(roi);
            cv::Mat layer_roi = layer(roi);
            layer_roi.setTo(class_color(instance.class_id), mask_roi);
            painted = true;
        }
        if (painted) {
            cv::addWeighted(canvas, 1.0 - options.mask_alpha, layer, options.mask_alpha, 0.0,
                            canvas);
        }
    }

    // 2. Контуры масок — чёткая граница объекта. Контуры ищутся в ROI, а точки
    //    затем смещаются обратно в координаты кадра.
    for (const auto& instance : instances) {
        const cv::Rect roi = instance.mask.empty() ? cv::Rect()
                                                   : instance.box & cv::Rect(0, 0, canvas.cols, canvas.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            continue;
        }
        std::vector<std::vector<cv::Point>> contours;
        // findContours модифицирует вход, поэтому нужна копия ROI.
        cv::findContours(instance.mask(roi).clone(), contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        for (auto& contour : contours) {
            for (auto& point : contour) {
                point.x += roi.x;
                point.y += roi.y;
            }
        }
        cv::drawContours(canvas, contours, -1, class_color(instance.class_id), thickness,
                         cv::LINE_AA);
    }

    // 3. Бокс и подпись: «#track_id class score».
    for (const auto& instance : instances) {
        const cv::Scalar color = class_color(instance.class_id);
        if (options.show_boxes) {
            cv::rectangle(canvas, instance.box, color, thickness, cv::LINE_AA);
        }
        // Подпись: "#id имя score". Имя берём из списка классов; если его нет
        // (меньше классов в файле, чем предсказала модель) — падаем обратно на
        // "c<id>", чтобы подпись не пропала совсем.
        const std::string label =
            std::string("#") + std::to_string(instance.track_id) + " " +
            class_label(options.class_names, instance.class_id) + " " + score_text(instance.score);
        put_label(canvas, label, cv::Point(instance.box.x, instance.box.y),
                  cv::Scalar(255, 255, 255), scale);
    }
    return canvas;
}

cv::Mat compose_side_by_side(const cv::Mat& raw, const cv::Mat& overlay) {
    cv::Mat canvas;
    // hconcat требует одинаковый тип и высоту; размеры у нас всегда равны,
    // потому что оба кадра получены из одного Frame.
    cv::hconcat(raw, overlay, canvas);
    cv::line(canvas, cv::Point(raw.cols, 0), cv::Point(raw.cols, canvas.rows),
             cv::Scalar(200, 200, 200), 2);

    const double scale = std::max(0.5, canvas.cols / 2200.0);
    put_label(canvas, "CAMERA", cv::Point(8, canvas.rows - 8), cv::Scalar(235, 235, 235), scale);
    put_label(canvas, "SEGMENTATION + TRACKING", cv::Point(raw.cols + 8, canvas.rows - 8),
              cv::Scalar(235, 235, 235), scale);
    return canvas;
}

void draw_hud(cv::Mat& canvas, const std::vector<std::string>& lines) {
    if (lines.empty()) {
        return;
    }
    const double scale = std::clamp(canvas.cols / 2200.0, 0.45, 0.8);
    const int line_height = static_cast<int>(22 * scale) + 5;
    int width = 0;
    for (const auto& line : lines) {
        width = std::max(width, cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, scale, 1,
                                                nullptr).width);
    }
    const cv::Rect panel(0, 0, std::min(canvas.cols, width + 16),
                         std::min(canvas.rows, line_height * static_cast<int>(lines.size()) + 10));
    if (panel.width <= 0 || panel.height <= 0) {
        return;
    }
    // Полупрозрачная подложка: ROI — это «вид» на оригинал, поэтому convertTo
    // меняет сам canvas.
    canvas(panel).convertTo(canvas(panel), -1, 0.45, 0.0);
    int y = line_height;
    for (const auto& line : lines) {
        cv::putText(canvas, line, cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                    cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        y += line_height;
    }
}

}  // namespace vt
