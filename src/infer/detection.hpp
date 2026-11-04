#pragma once

// Общие типы для всех моделей детекции/сегментации.
//
// Важный момент: InferenceResult — это immutable-снимок (только const-поля),
// который перелетает из потока инференса в UI-поток через shared_ptr. Поэтому
// разметку можно рисовать в одном потоке, а инференс уже считать следующий
// кадр — данные не разъедутся и не потребуют блокировок.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/frame.hpp"

namespace vt {

// Один найденный объект (instance).
struct Instance {
    int class_id = -1;
    float score = 0.0f;
    cv::Rect2f box;  // пиксели исходного кадра; Rect2f — дробный, чтобы не терять 0.5 px при resize
    // Бинарная маска (CV_8UC1, значения 0/255) в размере исходного кадра.
    // Может быть пустой, если модель детекции без масок.
    cv::Mat mask;
};

struct InferTimings {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms() const noexcept { return preprocess_ms + inference_ms + postprocess_ms; }
};

struct InferenceResult {
    std::shared_ptr<const Frame> frame;  // исходный кадр, к которому относится разметка
    std::vector<Instance> instances;
    InferTimings timings;
    double latency_ms = 0.0;  // от получения буфера камеры до готовой разметки
    int input_width = 0;      // размер входа модели (для справки в UI)
    int input_height = 0;
};

using ResultPtr = std::shared_ptr<const InferenceResult>;

// Описание загруженной модели — показываем в UI и в логе.
struct ModelInfo {
    std::string path;
    std::string task = "instance-segmentation";  // или "detection"
    std::string input_layout = "NCHW";
    std::string output_layout;
    int input_width = 0;
    int input_height = 0;
    int class_count = 0;
    std::string providers;  // какие execution providers реально активны
};

class ISegmenter {
public:
    virtual ~ISegmenter() = default;

    virtual bool load(std::string& error) = 0;

    // Принимает BGR-кадр (CV_8UC3), отдаёт найденные объекты в координатах
    // исходного кадра. Метод вызывается из одного потока — потокобезопасность
    // обеспечивается архитектурой (один поток на модель), а не мьютексом.
    virtual bool segment(const cv::Mat& bgr, std::vector<Instance>& out, std::string& error) = 0;

    virtual const std::vector<std::string>& class_names() const = 0;
    virtual ModelInfo info() const = 0;

    // Разбивка времени последнего вызова segment(). Читается тем же потоком,
    // который вызвал segment(), поэтому мьютекс не нужен: это не разделяемое
    // между потоками состояние, а «выходной параметр» с отложенным чтением.
    virtual InferTimings last_timings() const { return {}; }
};

using SegmenterPtr = std::unique_ptr<ISegmenter>;

}  // namespace vt
