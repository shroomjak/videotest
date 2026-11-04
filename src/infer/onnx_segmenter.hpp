#pragma once

// ONNX-модель instance-сегментации (YOLOv8-seg / YOLO11-seg / YOLO26-seg).
//
// Почему ONNX Runtime, а не cv::dnn:
//   * полная поддержка операторов и новых архитектур (end-to-end NMS, DFL и т.п.);
//   * явное управление потоками (intra-op threads) и провайдерами (CPU/CUDA);
//   * владельческие типы (Ort::Value, Ort::Session) и понятные ошибки (Ort::Exception).
//
// Форматы вывода, которые понимаем:
//   1) "сырой" YOLOv8/11-seg:  prediction [1, 4+nc+32, N]  + protos [1, 32, 160, 160];
//   2) end-to-end (YOLO26-seg): [1, N, 6+32] — NMS уже выполнен в графе,
//      столбцы: cx,cy,w,h, score, class, 32 коэффициента масок.
// Вариант определяется по форме выхода автоматически.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "infer/detection.hpp"

namespace vt {

struct SegmenterConfig {
    std::string model_path;
    int input_size = 640;       // используется, только если у модели динамический вход
    float score_threshold = 0.35f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.5f;
    int max_detections = 100;
    int intra_op_threads = 0;   // 0 — значение ORT по умолчанию
    std::string device = "cpu"; // "cpu" | "cuda"
};

// Геометрическое преобразование кадра к входу сети: масштаб по осям + отступы
// letterbox. Хранится одним объектом, чтобы препроцессинг и обратное
// преобразование координат гарантированно использовали одни и те же числа.
struct LetterboxTransform {
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    cv::Point2f pad{0.0f, 0.0f};

    // из координат исходного кадра в координаты входа сети
    cv::Rect2f to_input(const cv::Rect2f& box) const {
        return {box.x * scale_x + pad.x, box.y * scale_y + pad.y, box.width * scale_x,
                box.height * scale_y};
    }
    // и обратно
    cv::Rect2f to_source(const cv::Rect2f& box) const {
        return {(box.x - pad.x) / scale_x, (box.y - pad.y) / scale_y, box.width / scale_x,
                box.height / scale_y};
    }
};

class OnnxSegmenter final : public ISegmenter {
public:
    explicit OnnxSegmenter(SegmenterConfig config);
    ~OnnxSegmenter() override;

    OnnxSegmenter(const OnnxSegmenter&) = delete;
    OnnxSegmenter& operator=(const OnnxSegmenter&) = delete;

    bool load(std::string& error) override;
    bool segment(const cv::Mat& bgr, std::vector<Instance>& out, std::string& error) override;

    const std::vector<std::string>& class_names() const override { return class_names_; }
    ModelInfo info() const override { return info_; }
    InferTimings last_timings() const override { return timings_; }

private:
    struct Impl;  // Ort::Session/Env спрятаны, чтобы onnxruntime_cxx_api.h не тянулся в каждый TU

    // letterbox + нормализация + NCHW.
    // Выход — непрерывный CV_32F blob формы 1×3×H×W: его память отдаётся в
    // Ort::Value напрямую, без промежуточного std::vector и без копирования.
    LetterboxTransform make_input(const cv::Mat& bgr, cv::Mat& blob) const;

    enum class PredictionLayout { ChannelsFirst, RowsPerDetection };

    // Разбор выхода сети в боксы и коэффициенты масок (координаты входа сети).
    PredictionLayout decode_outputs(const float* prediction, const std::vector<int64_t>& shape,
                                   std::vector<Instance>& detections,
                                   std::vector<std::vector<float>>& coeffs, std::string& error) const;

    // Greedy NMS по классам (нужен только для "сырого" выхода).
    void apply_nms(std::vector<Instance>& detections,
                   std::vector<std::vector<float>>& coeffs) const;

    // Сборка бинарных масок: mask = sigmoid(protos^T · coeffs), затем crop по боксу
    // и обратное преобразование в координаты исходного кадра.
    void build_masks(const std::vector<Instance>& detections,
                     const std::vector<std::vector<float>>& coeffs, const float* protos,
                     const std::vector<int64_t>& proto_shape, const LetterboxTransform& transform,
                     const cv::Size& original_size, std::vector<Instance>& out) const;

    SegmenterConfig config_;
    std::vector<std::string> class_names_;
    ModelInfo info_;
    InferTimings timings_;
    std::unique_ptr<Impl> impl_;
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_cache_;
    std::vector<const char*> output_names_cache_;
};

}  // namespace vt
