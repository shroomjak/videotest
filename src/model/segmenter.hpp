#pragma once

// ONNX-модель instance-сегментации (YOLO*-seg).
//
// Соглашения о формате ЗАДАЮТСЯ ЯВНО и не определяются автоматически:
//   * вход:  NCHW float32, RGB, [0,1], размер input_size x input_size,
//            letterbox (пропорции сохраняются, поля серые 114);
//   * выход: out_layout = "raw" — [1, 4+nc+32, N] (YOLOv8/11-seg):
//            NMS выполняется в C++;
//            out_layout = "nms" — [1, N, 6+32] (YOLO26 end-to-end):
//            NMS уже внутри графа, C++ его не повторяет.
// Число классов (nc) тоже задаётся пользователем, а не угадывается.
//
// Если файл модели не задан, используется пустая модель: она возвращает ноль
// объектов, и «размеченный» поток совпадает с исходным. Отдельного режима
// «без модели» нет — конвейер всегда работает одинаково.

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "frame.hpp"

namespace vt {

struct ModelConfig {
    std::string path;              // пусто => пустая модель (нет разметки)
    int input_size = 640;          // квадратный вход сети
    std::string out_layout = "raw";// "raw" | "nms"
    int class_count = 80;          // сколько классов у модели
    float score_threshold = 0.35f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.5f;
    int threads = 2;               // intra-op потоков ONNX Runtime
};

class Segmenter {
public:
    explicit Segmenter(ModelConfig config);
    ~Segmenter();

    Segmenter(const Segmenter&) = delete;
    Segmenter& operator=(const Segmenter&) = delete;

    bool load(std::string& error);

    // BGR-кадр -> объекты в координатах исходного кадра. Для пустой модели
    // возвращает пустой вектор.
    bool infer(const cv::Mat& bgr, std::vector<Instance>& out, double* infer_ms,
               std::string& error);

    bool empty() const { return empty_; }

private:
    // Преобразование кадра к входу сети: считает масштаб и отступы letterbox и
    // заполняет blob формы 1x3xSxS. Возвращаемые scale/pad нужны, чтобы
    // перевести координаты обратно в пиксели исходного кадра.
    void preprocess(const cv::Mat& bgr, cv::Mat& blob, float& scale, cv::Point2f& pad) const;

    void decode_raw(const float* pred, int64_t channels, int64_t anchors,
                    std::vector<Instance>& dets, std::vector<std::vector<float>>& coeffs) const;
    void decode_nms(const float* pred, int64_t rows, int64_t cols,
                    std::vector<Instance>& dets, std::vector<std::vector<float>>& coeffs) const;
    void suppress(std::vector<Instance>& dets, std::vector<std::vector<float>>& coeffs) const;
    void make_masks(const std::vector<Instance>& dets,
                    const std::vector<std::vector<float>>& coeffs, const float* protos,
                    const std::vector<int64_t>& proto_shape, float scale, const cv::Point2f& pad,
                    const cv::Size& frame_size, std::vector<Instance>& out) const;

    ModelConfig config_;
    bool empty_ = false;
    bool loaded_ = false;
    int input_size_ = 0;

    // Всё, что относится к ONNX Runtime, спрятано в .cpp (pimpl «руками»):
    // заголовок модели не должен тянуть onnxruntime_cxx_api.h во все файлы.
    struct OrtHolder;
    OrtHolder* ort_ = nullptr;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_ptrs_;
    std::vector<const char*> output_ptrs_;
};

}  // namespace vt
