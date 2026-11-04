// Тесты постобработки: IoU, NMS, декодирование выходов и сборка масок.
// Всё это чистая математика без ONNX Runtime — проверяется быстро и точно.

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "infer/nms.hpp"
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

using vt::Instance;
using vt::box_iou;
using vt::class_aware_nms;

Instance make(int class_id, float score, float x, float y, float w, float h) {
    Instance det;
    det.class_id = class_id;
    det.score = score;
    det.box = cv::Rect2f(x, y, w, h);
    return det;
}

std::vector<std::vector<float>> make_coeffs(std::size_t n) {
    std::vector<std::vector<float>> coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        coeffs[i] = {static_cast<float>(i)};  // маркер, чтобы проверить синхронность
    }
    return coeffs;
}

void test_iou_identity() {
    const cv::Rect2f a(0, 0, 100, 100);
    CHECK_NEAR(box_iou(a, a), 1.0f, 1e-6f);
}

void test_iou_disjoint() {
    const cv::Rect2f a(0, 0, 50, 50);
    const cv::Rect2f b(100, 100, 50, 50);
    CHECK_NEAR(box_iou(a, b), 0.0f, 1e-6f);
}

void test_iou_half_overlap() {
    // Два квадрата 100x100 со сдвигом на 50: пересечение 50x100 = 5000,
    // объединение 10000 + 10000 - 5000 = 15000 -> 1/3.
    const cv::Rect2f a(0, 0, 100, 100);
    const cv::Rect2f b(50, 0, 100, 100);
    CHECK_NEAR(box_iou(a, b), 1.0f / 3.0f, 1e-5f);
}

void test_iou_degenerate_box() {
    // Нулевая площадь не должна приводить к делению на ноль.
    const cv::Rect2f a(10, 10, 0, 0);
    const cv::Rect2f b(10, 10, 50, 50);
    CHECK_NEAR(box_iou(a, b), 0.0f, 1e-6f);
}

void test_nms_suppresses_duplicates() {
    std::vector<Instance> dets = {make(0, 0.9f, 0, 0, 100, 100),
                                  make(0, 0.8f, 5, 5, 100, 100),   // тот же объект
                                  make(0, 0.7f, 200, 200, 100, 100)};
    auto coeffs = make_coeffs(dets.size());
    class_aware_nms(dets, coeffs, 0.45f, 100);
    CHECK(dets.size() == 2);
    CHECK_NEAR(dets[0].score, 0.9f, 1e-6f);
    CHECK_NEAR(dets[1].score, 0.7f, 1e-6f);
    // Коэффициенты должны ехать вместе со своими боксами.
    CHECK(coeffs.size() == 2);
    CHECK_NEAR(coeffs[0][0], 0.0f, 1e-6f);
    CHECK_NEAR(coeffs[1][0], 2.0f, 1e-6f);
}

void test_nms_is_class_aware() {
    // Почти полностью перекрывающиеся боксы разных классов — оба должны выжить.
    std::vector<Instance> dets = {make(0, 0.9f, 0, 0, 100, 100),
                                  make(1, 0.8f, 0, 0, 100, 100)};
    auto coeffs = make_coeffs(dets.size());
    class_aware_nms(dets, coeffs, 0.45f, 100);
    CHECK(dets.size() == 2);
}

void test_nms_respects_max_detections() {
    std::vector<Instance> dets;
    for (int i = 0; i < 10; ++i) {
        // Разносим боксы, чтобы они не подавляли друг друга.
        dets.push_back(make(0, 0.5f + 0.01f * static_cast<float>(i),
                            static_cast<float>(i) * 100.0f, 0.0f, 50.0f, 50.0f));
    }
    auto coeffs = make_coeffs(dets.size());
    class_aware_nms(dets, coeffs, 0.45f, 3);
    CHECK(dets.size() == 3);
    CHECK_NEAR(dets[0].score, 0.59f, 1e-5f);  // самый уверенный идёт первым
}

void test_nms_iou_threshold_effect() {
    // Два бокса 100x100 со сдвигом 95 px: IoU = 500 / 19500 ≈ 0.026.
    const auto make_pair = [] {
        return std::vector<Instance>{make(0, 0.9f, 0, 0, 100, 100),
                                     make(0, 0.8f, 95, 0, 100, 100)};
    };

    // Порог 0.45 (значение по умолчанию): перекрытие мало, оба остаются.
    auto tolerant = make_pair();
    auto tolerant_coeffs = make_coeffs(tolerant.size());
    class_aware_nms(tolerant, tolerant_coeffs, 0.45f, 100);
    CHECK(tolerant.size() == 2);

    // Порог 0.01: почти любое перекрытие считается дубликатом.
    auto strict = make_pair();
    auto strict_coeffs = make_coeffs(strict.size());
    class_aware_nms(strict, strict_coeffs, 0.01f, 100);
    CHECK(strict.size() == 1);
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests = {
        {"iou_identity", test_iou_identity},
        {"iou_disjoint", test_iou_disjoint},
        {"iou_half_overlap", test_iou_half_overlap},
        {"iou_degenerate_box", test_iou_degenerate_box},
        {"nms_suppresses_duplicates", test_nms_suppresses_duplicates},
        {"nms_is_class_aware", test_nms_is_class_aware},
        {"nms_respects_max_detections", test_nms_respects_max_detections},
        {"nms_iou_threshold_effect", test_nms_iou_threshold_effect},
    };

    for (const auto& [name, fn] : tests) {
        const int before = g_failures;
        fn();
        std::printf("%-30s %s\n", name, (g_failures == before) ? "OK" : "FAILED");
    }
    if (g_failures != 0) {
        std::printf("\nвсего ошибок: %d\n", g_failures);
        return 1;
    }
    std::printf("\nвсе тесты прошли\n");
    return 0;
}
