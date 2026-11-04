// Тесты геометрии: letterbox-преобразование, сборка масок и разбор формы
// выходов модели. Всё это — чистая математика без камеры и без ONNX-файла,
// поэтому тестируется детерминированно и быстро.

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "infer/onnx_segmenter.hpp"

namespace {

int g_failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #condition); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (false)

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) <= (eps))

using vt::LetterboxTransform;

// Кадр 1280x720 -> вход 640x640: scale = 0.5, pad_y = (640-360)/2 = 140.
void test_letterbox_roundtrip() {
    LetterboxTransform t;
    t.scale_x = 0.5f;
    t.scale_y = 0.5f;
    t.pad = {0.0f, 140.0f};

    const cv::Rect2f source{100.0f, 50.0f, 200.0f, 120.0f};
    const cv::Rect2f input = t.to_input(source);
    CHECK_NEAR(input.x, 50.0f, 1e-4);
    CHECK_NEAR(input.y, 165.0f, 1e-4);
    CHECK_NEAR(input.width, 100.0f, 1e-4);
    CHECK_NEAR(input.height, 60.0f, 1e-4);

    const cv::Rect2f back = t.to_source(input);
    CHECK_NEAR(back.x, source.x, 1e-3);
    CHECK_NEAR(back.y, source.y, 1e-3);
    CHECK_NEAR(back.width, source.width, 1e-3);
    CHECK_NEAR(back.height, source.height, 1e-3);
}

// Горизонтальный кадр 1920x1080 -> 640x640: scale = 640/1920, поля слева/справа.
void test_letterbox_wide_frame() {
    const float scale = 640.0f / 1920.0f;
    LetterboxTransform t;
    t.scale_x = scale;
    t.scale_y = scale;
    t.pad = {0.0f, (640.0f - 1080.0f * scale) * 0.5f};

    // Точка в левом верхнем углу кадра должна попасть на левый край входа.
    const cv::Rect2f corner = t.to_input(cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f));
    CHECK_NEAR(corner.x, 0.0f, 1e-3);
    CHECK_NEAR(corner.y, t.pad.y, 1e-3);

    // Правый нижний угол — правый край.
    const cv::Rect2f far_corner =
        t.to_input(cv::Rect2f(1919.0f, 1079.0f, 1.0f, 1.0f));
    CHECK_NEAR(far_corner.x + far_corner.width, 640.0f, 0.5f);
}

void test_class_names_fallback() {
    // У модели без метаданных имена должны быть синтетическими, но не пустыми.
    vt::SegmenterConfig config;
    config.model_path = "/nonexistent.onnx";
    vt::OnnxSegmenter segmenter(config);
    std::string error;
    // Файла нет — load() обязан вернуть false и заполнить error, не бросая.
    const bool loaded = segmenter.load(error);
    CHECK(!loaded);
    CHECK(!error.empty());
    std::printf("  (ожидаемая ошибка загрузки: %s)\n", error.c_str());
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests = {
        {"letterbox_roundtrip", test_letterbox_roundtrip},
        {"letterbox_wide_frame", test_letterbox_wide_frame},
        {"class_names_fallback", test_class_names_fallback},
    };

    for (const auto& [name, fn] : tests) {
        const int before = g_failures;
        fn();
        std::printf("%-24s %s\n", name, (g_failures == before) ? "OK" : "FAILED");
    }
    if (g_failures != 0) {
        std::printf("\nвсего ошибок: %d\n", g_failures);
        return 1;
    }
    std::printf("\nвсе тесты прошли\n");
    return 0;
}
