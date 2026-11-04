#include "infer/onnx_segmenter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cctype>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

// Из cv::dnn здесь нужен только blobFromImage (препроцессинг), поэтому
// тянуть <opencv2/dnn/dnn.hpp> не требуется.

#include "core/logger.hpp"
#include "infer/nms.hpp"

namespace vt {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// Ultralytics кладёт имена классов в metadata["names"] python-dict'ом:
//   {0: 'person', 1: 'bicycle'}
// Приводим кавычки к JSON-виду и разбираем вручную, без внешних зависимостей.
std::unordered_map<int, std::string> parse_class_names(const std::string& text) {
    std::unordered_map<int, std::string> names;
    std::string json = text;
    std::replace(json.begin(), json.end(), '\'', '"');

    std::size_t pos = 0;
    while ((pos = json.find(':', pos)) != std::string::npos) {
        std::size_t key_end = pos;
        while (key_end > 0 && std::isspace(static_cast<unsigned char>(json[key_end - 1]))) {
            --key_end;
        }
        std::size_t key_start = key_end;
        while (key_start > 0 && std::isdigit(static_cast<unsigned char>(json[key_start - 1]))) {
            --key_start;
        }
        if (key_start == key_end) {  // не числовой ключ
            ++pos;
            continue;
        }
        const int id = std::stoi(json.substr(key_start, key_end - key_start));

        const std::size_t value_start = json.find('"', pos);
        if (value_start == std::string::npos) {
            break;
        }
        const std::size_t value_end = json.find('"', value_start + 1);
        if (value_end == std::string::npos) {
            break;
        }
        names[id] = json.substr(value_start + 1, value_end - value_start - 1);
        pos = value_end + 1;
    }
    return names;
}

int find_output_index(const std::vector<std::string>& names, const std::vector<std::string>& keys) {
    for (std::size_t i = 0; i < names.size(); ++i) {
        for (const auto& key : keys) {
            if (names[i].find(key) != std::string::npos) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream text;
    text << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text << (i ? "," : "") << shape[i];
    }
    text << ']';
    return text.str();
}

}  // namespace

struct OnnxSegmenter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "videotest"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info{"Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault};

    int prediction_output = 0;  // индексы в списке выходов модели
    int protos_output = -1;
    bool has_protos = false;
};

OnnxSegmenter::OnnxSegmenter(SegmenterConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {}

OnnxSegmenter::~OnnxSegmenter() = default;

bool OnnxSegmenter::load(std::string& error) {
    try {
        // --- опции сессии -------------------------------------------------------
        impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config_.intra_op_threads > 0) {
            impl_->options.SetIntraOpNumThreads(config_.intra_op_threads);
        }
        if (config_.device == "cuda") {
            try {
                OrtCUDAProviderOptions cuda{};
                impl_->options.AppendExecutionProvider_CUDA(cuda);
                info_.providers = "CUDA";
            } catch (const Ort::Exception& e) {
                // Сборка ORT может быть без CUDA — это не повод падать: считаем на CPU.
                LOG_WARN << "CUDA provider недоступен (" << e.what() << "), работаем на CPU";
                info_.providers = "CPU";
            }
        } else {
            info_.providers = "CPU";
        }

        impl_->session =
            std::make_unique<Ort::Session>(impl_->env, config_.model_path.c_str(), impl_->options);

        // --- имена входов/выходов ----------------------------------------------
        const std::size_t input_count = impl_->session->GetInputCount();
        const std::size_t output_count = impl_->session->GetOutputCount();
        for (std::size_t i = 0; i < input_count; ++i) {
            Ort::AllocatedStringPtr name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
            input_names_storage_.emplace_back(name.get());
        }
        for (std::size_t i = 0; i < output_count; ++i) {
            Ort::AllocatedStringPtr name =
                impl_->session->GetOutputNameAllocated(i, impl_->allocator);
            output_names_storage_.emplace_back(name.get());
        }
        for (const auto& name : input_names_storage_) input_names_cache_.push_back(name.c_str());
        for (const auto& name : output_names_storage_) output_names_cache_.push_back(name.c_str());

        // --- форма входа --------------------------------------------------------
        auto input_shape = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() != 4) {
            error = "ожидался 4D-вход (NCHW), получено измерений: " +
                    std::to_string(input_shape.size());
            return false;
        }
        if (input_shape[2] <= 0 || input_shape[3] <= 0) {
            LOG_WARN << "у модели динамический размер входа — используем " << config_.input_size;
            input_shape[2] = config_.input_size;
            input_shape[3] = config_.input_size;
        }
        info_.input_height = static_cast<int>(input_shape[2]);
        info_.input_width = static_cast<int>(input_shape[3]);
        info_.input_layout = (input_shape[1] == 3) ? "NCHW" : "NHWC(?)";
        info_.path = config_.model_path;

        // --- назначение выходов -------------------------------------------------
        impl_->prediction_output =
            find_output_index(output_names_storage_, {"output0", "pred", "detect"});
        impl_->protos_output =
            find_output_index(output_names_storage_, {"output1", "proto", "mask", "coeff"});
        if (impl_->prediction_output < 0) {
            impl_->prediction_output = 0;
        }
        impl_->has_protos = (output_count > 1) && (impl_->protos_output >= 0) &&
                           (impl_->protos_output != impl_->prediction_output);
        if (!impl_->has_protos) {
            impl_->protos_output = -1;
            info_.task = "detection";
            LOG_WARN << "выходов-прототипов масок нет — модель используется как детектор";
        }

        for (std::size_t i = 0; i < output_names_storage_.size(); ++i) {
            const auto shape =
                impl_->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            const std::string text = output_names_storage_[i] + shape_to_string(shape);
            LOG_INFO << "выход модели: " << text;
            info_.output_layout += (info_.output_layout.empty() ? "" : ", ") + text;
        }

        // --- имена классов из метаданных ---------------------------------------
        std::unordered_map<int, std::string> metadata_names;
        try {
            Ort::ModelMetadata metadata = impl_->session->GetModelMetadata();
            Ort::AllocatedStringPtr names =
                metadata.LookupCustomMetadataMapAllocated("names", impl_->allocator);
            if (names != nullptr) {
                metadata_names = parse_class_names(names.get());
            }
        } catch (const Ort::Exception&) {
            // метаданных может не быть — это нормально
        }

        // --- число классов из формы выхода -------------------------------------
        const auto pred_shape = impl_->session->GetOutputTypeInfo(impl_->prediction_output)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape();
        const int coeffs = impl_->has_protos ? 32 : 0;
        int class_count = 0;
        if (pred_shape.size() == 3) {
            if (pred_shape[1] < pred_shape[2]) {
                class_count = static_cast<int>(pred_shape[1]) - 4 - coeffs;  // [1, 4+nc+32, N]
            } else {
                class_count = static_cast<int>(pred_shape[2]) - 6 - coeffs;  // [1, N, 6+32]
            }
        }
        class_count = std::max(class_count, static_cast<int>(metadata_names.size()));
        class_count = std::max(class_count, 1);
        if (impl_->has_protos && pred_shape.size() == 3 && pred_shape[1] > pred_shape[2] &&
            metadata_names.empty()) {
            // [1, N, 6+32] без метаданных: число классов принципиально неизвестно.
            LOG_WARN << "у end-to-end модели нет имён классов в метаданных — "
                        "объекты будут подписаны как class_N";
        }

        class_names_.resize(static_cast<std::size_t>(class_count));
        for (int i = 0; i < class_count; ++i) {
            const auto it = metadata_names.find(i);
            class_names_[static_cast<std::size_t>(i)] =
                (it != metadata_names.end()) ? it->second : ("class_" + std::to_string(i));
        }
        info_.class_count = class_count;

        LOG_INFO << "модель загружена: " << config_.model_path << " | вход " << info_.input_width
                 << 'x' << info_.input_height << " | классов " << class_count << " | провайдер "
                 << info_.providers;
        return true;
    } catch (const Ort::Exception& e) {
        error = std::string("ONNX Runtime: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        error = std::string("ошибка загрузки модели: ") + e.what();
        return false;
    }
}

LetterboxTransform OnnxSegmenter::make_input(const cv::Mat& bgr, cv::Mat& blob) const {
    // Letterbox: вписываем кадр в квадрат с сохранением пропорций, поля
    // заполняем серым 114 (как при обучении YOLO). Без этого объекты
    // «растягиваются», и обратное преобразование масок даёт смещение.
    const int in_w = info_.input_width;
    const int in_h = info_.input_height;

    LetterboxTransform transform;
    transform.scale_x = static_cast<float>(in_w) / static_cast<float>(bgr.cols);
    transform.scale_y = static_cast<float>(in_h) / static_cast<float>(bgr.rows);
    const float scale = std::min(transform.scale_x, transform.scale_y);
    transform.scale_x = scale;
    transform.scale_y = scale;
    transform.pad = {static_cast<float>(in_w) - static_cast<float>(bgr.cols) * scale,
                     static_cast<float>(in_h) - static_cast<float>(bgr.rows) * scale};
    transform.pad.x *= 0.5f;
    transform.pad.y *= 0.5f;

    cv::Mat canvas(in_h, in_w, CV_8UC3, cv::Scalar(114, 114, 114));
    const int new_w = static_cast<int>(std::lround(static_cast<float>(bgr.cols) * scale));
    const int new_h = static_cast<int>(std::lround(static_cast<float>(bgr.rows) * scale));
    cv::Mat resized;
    // INTER_LINEAR — компромисс скорости и качества; для уменьшения в 2+ раза
    // INTER_AREA даёт меньше алиасинга, но заметно дороже.
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(static_cast<int>(transform.pad.x),
                                   static_cast<int>(transform.pad.y), new_w, new_h)));

    // blobFromImage: BGR->RGB (swapRB), HWC->NCHW, /255 -> непрерывный
    // CV_32F blob формы (1,3,H,W). Единственное место, где мы платим за
    // переупаковку пикселей под модель.
    //
    // Важно: выходной параметр — cv::Mat, а не std::vector<float>:
    // blobFromImageWithParams требует Mat/UMat (проверка внутри OpenCV),
    // а вектор пришлось бы потом ещё раз копировать.
    cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(in_w, in_h), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false, CV_32F);
    return transform;
}

OnnxSegmenter::PredictionLayout OnnxSegmenter::decode_outputs(
    const float* prediction, const std::vector<int64_t>& shape, std::vector<Instance>& detections,
    std::vector<std::vector<float>>& coeffs, std::string& error) const {
    detections.clear();
    coeffs.clear();

    if (shape.size() != 3 || shape[0] != 1) {
        error = "неожиданная форма выхода: " + shape_to_string(shape);
        return PredictionLayout::ChannelsFirst;
    }

    const int num_classes = info_.class_count;
    const int coeff_count = impl_->has_protos ? 32 : 0;
    const std::int64_t dim1 = shape[1];
    const std::int64_t dim2 = shape[2];

    // Какой из двух форматов перед нами? Ориентация тензора сама по себе
    // ненадёжный признак: [1, 37, 8400] (1 класс, сырой выход) и [1, 300, 38]
    // (end-to-end) отличаются только числами. Поэтому сравниваем измерения с
    // ожидаемым числом каналов/столбцов, а к эвристике по размерам прибегаем
    // только если ни одно не совпало.
    const std::int64_t raw_channels = 4 + num_classes + coeff_count;
    const std::int64_t e2e_columns = 6 + coeff_count;
    bool channels_first;
    if (dim1 == raw_channels) {
        channels_first = true;
    } else if (dim2 == e2e_columns) {
        channels_first = false;
    } else {
        channels_first = dim1 < dim2;
    }

    if (channels_first) {
        // --- [1, 4+nc+32, N]: сырой выход YOLOv8/11-seg -------------------------
        // Матрица хранится "канал-мажорно": значение канала c для якоря a —
        // prediction[c * N + a]. Поэтому обход по якорям — с шагом 1.
        const std::int64_t anchors = dim2;
        const std::int64_t channels = dim1;
        if (channels < 4 + num_classes) {
            error = "в выходе меньше каналов, чем 4 + число классов";
            return PredictionLayout::ChannelsFirst;
        }

        const float* cx_row = prediction;
        const float* cy_row = prediction + anchors;
        const float* w_row = prediction + 2 * anchors;
        const float* h_row = prediction + 3 * anchors;

        for (std::int64_t a = 0; a < anchors; ++a) {
            float best_score = 0.0f;
            int best_class = -1;
            for (int c = 0; c < num_classes; ++c) {
                const float value = prediction[static_cast<std::int64_t>(4 + c) * anchors + a];
                if (value > best_score) {
                    best_score = value;
                    best_class = c;
                }
            }
            if (best_class < 0 || best_score < config_.score_threshold) {
                continue;
            }
            const float w = w_row[a];
            const float h = h_row[a];
            if (w <= 0.0f || h <= 0.0f) {
                continue;
            }

            Instance det;
            det.class_id = best_class;
            det.score = best_score;
            det.box = cv::Rect2f(cx_row[a] - w * 0.5f, cy_row[a] - h * 0.5f, w, h);

            std::vector<float> det_coeffs;
            if (coeff_count > 0 && channels >= 4 + num_classes + coeff_count) {
                det_coeffs.resize(static_cast<std::size_t>(coeff_count));
                for (int k = 0; k < coeff_count; ++k) {
                    det_coeffs[static_cast<std::size_t>(k)] =
                        prediction[static_cast<std::int64_t>(4 + num_classes + k) * anchors + a];
                }
            }
            detections.push_back(std::move(det));
            coeffs.push_back(std::move(det_coeffs));
        }
        return PredictionLayout::ChannelsFirst;
    }

    // --- [1, N, 6+32]: end-to-end выход (NMS выполнен в графе, YOLO26) ---------
    // Столбцы: x1, y1, x2, y2, score, class_id, 32 коэффициента масок.
    //
    // Замечание: экспортёры разных версий иногда отдают здесь cx,cy,w,h.
    // Отличить можно по большинству строк: для xyxy x2 >= x1 и y2 >= y1.
    // Определяем формат один раз по первой «осмысленной» строке — так модель
    // не придётся перекомпилировать под каждый вариант экспорта.
    const std::int64_t rows = dim1;
    const std::int64_t cols = dim2;
    if (cols < 6) {
        error = "end-to-end выход должен иметь минимум 6 столбцов";
        return PredictionLayout::RowsPerDetection;
    }

    bool xyxy_format = true;
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = prediction + r * cols;
        if (row[2] <= 0.0f || row[3] <= 0.0f) {
            continue;  // пустая строка (padding после top-k)
        }
        xyxy_format = (row[2] >= row[0]) && (row[3] >= row[1]);
        break;
    }

    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = prediction + r * cols;
        const float score = row[4];
        if (score < config_.score_threshold) {
            continue;
        }
        Instance det;
        // В end-to-end выходе класс — просто номер в столбце, а не one-hot, поэтому
        // число классов из формы тензора вывести нельзя. Не зажимаем id: если
        // метаданных с именами нет, рендерер подпишет объект как class_N.
        det.class_id = static_cast<int>(row[5]);
        if (det.class_id < 0) {
            det.class_id = 0;
        }
        det.score = score;
        det.box = xyxy_format
                      ? cv::Rect2f(row[0], row[1], row[2] - row[0], row[3] - row[1])
                      : cv::Rect2f(row[0] - row[2] * 0.5f, row[1] - row[3] * 0.5f, row[2], row[3]);

        std::vector<float> det_coeffs;
        if (coeff_count > 0 && cols >= 6 + coeff_count) {
            det_coeffs.assign(row + 6, row + 6 + coeff_count);
        }
        detections.push_back(std::move(det));
        coeffs.push_back(std::move(det_coeffs));
    }
    return PredictionLayout::RowsPerDetection;
}

void OnnxSegmenter::apply_nms(std::vector<Instance>& detections,
                              std::vector<std::vector<float>>& coeffs) const {
    // Сама реализация — в infer/nms.hpp: это чистая функция, которую можно
    // тестировать без модели и без ONNX Runtime.
    class_aware_nms(detections, coeffs, config_.nms_threshold, config_.max_detections);
}

void OnnxSegmenter::build_masks(const std::vector<Instance>& detections,
                                const std::vector<std::vector<float>>& coeffs, const float* protos,
                                const std::vector<int64_t>& proto_shape,
                                const LetterboxTransform& transform, const cv::Size& original_size,
                                std::vector<Instance>& out) const {
    // protos: [1, 32, ph, pw] — базис из 32 «мягких масок».
    // Маска объекта = sigmoid(сумма_k coeff[k] * proto[k]).
    out.clear();
    out.reserve(detections.size());
    if (proto_shape.size() != 4) {
        return;
    }
    const int proto_count = static_cast<int>(proto_shape[1]);
    const int proto_h = static_cast<int>(proto_shape[2]);
    const int proto_w = static_cast<int>(proto_shape[3]);
    const std::size_t plane = static_cast<std::size_t>(proto_h) * proto_w;

    for (std::size_t index = 0; index < detections.size(); ++index) {
        const Instance& det = detections[index];
        const auto& det_coeffs = coeffs[index];
        if (static_cast<int>(det_coeffs.size()) != proto_count) {
            continue;  // коэффициентов нет — объект останется без маски
        }

        // 1. Линейная комбинация прототипов -> float-маска в размере protos.
        cv::Mat combined = cv::Mat::zeros(proto_h, proto_w, CV_32F);
        for (int k = 0; k < proto_count; ++k) {
            const float* channel = protos + static_cast<std::size_t>(k) * plane;
            const float coeff = det_coeffs[static_cast<std::size_t>(k)];
            if (coeff == 0.0f) {
                continue;
            }
            for (int y = 0; y < proto_h; ++y) {
                const float* src = channel + static_cast<std::size_t>(y) * proto_w;
                float* dst = combined.ptr<float>(y);
                for (int x = 0; x < proto_w; ++x) {
                    dst[x] += coeff * src[x];  // accumulate в один проход
                }
            }
        }

        // 2. Сигмоида + обрезка по боксу, переведённому в координаты protos.
        const float ratio_x = static_cast<float>(proto_w) / static_cast<float>(info_.input_width);
        const float ratio_y = static_cast<float>(proto_h) / static_cast<float>(info_.input_height);

        cv::Rect2f box = det.box;
        cv::Rect2f clipped = box & cv::Rect2f(0.0f, 0.0f, static_cast<float>(info_.input_width),
                                              static_cast<float>(info_.input_height));
        cv::Rect crop(static_cast<int>(std::floor(clipped.x * ratio_x)),
                      static_cast<int>(std::floor(clipped.y * ratio_y)),
                      std::max(1, static_cast<int>(std::ceil(clipped.width * ratio_x))),
                      std::max(1, static_cast<int>(std::ceil(clipped.height * ratio_y))));
        crop &= cv::Rect(0, 0, proto_w, proto_h);

        cv::Rect2f source_box = transform.to_source(box);
        source_box &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(original_size.width),
                                 static_cast<float>(original_size.height));

        cv::Mat mask;
        // Объект может оказаться за границей кадра после обратного преобразования.
        // Тогда он остаётся в списке, но без маски: бокс рисуется, заливки нет.
        if (source_box.width < 1.0f || source_box.height < 1.0f) {
            Instance instance;
            instance.class_id = det.class_id;
            instance.score = det.score;
            instance.box = source_box;
            out.push_back(std::move(instance));
            continue;
        }

        if (crop.width > 0 && crop.height > 0) {
            cv::Mat crop_view = combined(crop);
            cv::Mat resized;
            cv::resize(crop_view, resized,
                       cv::Size(std::max(1, static_cast<int>(std::lround(source_box.width))),
                                std::max(1, static_cast<int>(std::lround(source_box.height)))),
                       0, 0, cv::INTER_LINEAR);

            // 3. Сигмоида и порог: sigmoid(x) > t  эквивалентно  x > logit(t),
            // поэтому вместо поэлементной exp() + деления достаточно одного
            // сравнения с порогом в логитах. Битово совпадает с «честной»
            // сигмоидой по знаку разности, но в разы дешевле на больших масках.
            const float threshold = config_.mask_threshold;
            const float logit_threshold = std::log(threshold / (1.0f - threshold));
            cv::Mat binary;
            cv::compare(resized, logit_threshold, binary, cv::CMP_GT);

            // 4. Возвращаем маску в размер исходного кадра.
            mask = cv::Mat::zeros(original_size, CV_8UC1);
            const int x = std::max(0, static_cast<int>(std::lround(source_box.x)));
            const int y = std::max(0, static_cast<int>(std::lround(source_box.y)));
            const int w = std::min(binary.cols, mask.cols - x);
            const int h = std::min(binary.rows, mask.rows - y);
            if (w > 0 && h > 0) {
                binary(cv::Rect(0, 0, w, h)).copyTo(mask(cv::Rect(x, y, w, h)));
            }
        }

        Instance instance;
        instance.class_id = det.class_id;
        instance.score = det.score;
        instance.box = source_box;
        instance.mask = std::move(mask);
        out.push_back(std::move(instance));
    }
}

bool OnnxSegmenter::segment(const cv::Mat& bgr, std::vector<Instance>& out, std::string& error) {
    if (!impl_->session) {
        error = "модель не загружена";
        return false;
    }
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        error = "на вход модели ожидался непустой BGR-кадр CV_8UC3";
        return false;
    }

    try {
        out.clear();
        const auto t0 = Clock::now();

        // --- 1. препроцессинг ---------------------------------------------------
        cv::Mat input_blob;
        const LetterboxTransform transform = make_input(bgr, input_blob);
        const auto t1 = Clock::now();
        if (input_blob.empty() || !input_blob.isContinuous()) {
            error = "не удалось подготовить вход модели";
            return false;
        }

        const std::array<int64_t, 4> input_shape{1, 3, info_.input_height, info_.input_width};
        // Память blob'а живёт до конца вызова segment(), а Ort::Value не копирует
        // данные — он лишь ссылается на них на время Run().
        Ort::Value input = Ort::Value::CreateTensor<float>(
            impl_->memory_info, input_blob.ptr<float>(), static_cast<std::size_t>(input_blob.total()),
            input_shape.data(), input_shape.size());

        // --- 2. инференс --------------------------------------------------------
        std::vector<Ort::Value> outputs = impl_->session->Run(
            Ort::RunOptions{nullptr}, input_names_cache_.data(), &input, 1,
            output_names_cache_.data(), output_names_cache_.size());
        const auto t2 = Clock::now();

        // --- 3. декодирование выходов ------------------------------------------
        const std::size_t pred_index = static_cast<std::size_t>(impl_->prediction_output);
        const auto pred_info = outputs[pred_index].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> pred_shape = pred_info.GetShape();
        const float* pred_data = outputs[pred_index].GetTensorData<float>();

        std::vector<Instance> detections;
        std::vector<std::vector<float>> coeffs;
        const PredictionLayout layout =
            decode_outputs(pred_data, pred_shape, detections, coeffs, error);
        if (!error.empty()) {
            return false;
        }
        // End-to-end выход уже прошёл NMS внутри модели (YOLO26: one-to-one head),
        // поэтому второй раз подавлять не нужно — можно случайно удалить
        // легитимно перекрывающиеся объекты. Но ограничение на количество
        // применяем всегда: оно страхует и стоимость сборки масок, и UI.
        if (layout == PredictionLayout::ChannelsFirst && detections.size() > 1) {
            apply_nms(detections, coeffs);
        } else if (detections.size() > static_cast<std::size_t>(config_.max_detections)) {
            detections.resize(static_cast<std::size_t>(config_.max_detections));
            coeffs.resize(detections.size());
        }

        const float* proto_data = nullptr;
        std::vector<int64_t> proto_shape;
        if (impl_->has_protos) {
            const std::size_t proto_index = static_cast<std::size_t>(impl_->protos_output);
            proto_shape = outputs[proto_index].GetTensorTypeAndShapeInfo().GetShape();
            proto_data = outputs[proto_index].GetTensorData<float>();
        }

        // --- 4. маски и обратное преобразование координат -----------------------
        if (proto_data != nullptr) {
            build_masks(detections, coeffs, proto_data, proto_shape, transform, bgr.size(), out);
        } else {
            for (auto& det : detections) {
                det.box = transform.to_source(det.box);
                det.box &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(bgr.cols),
                                      static_cast<float>(bgr.rows));
                out.push_back(std::move(det));
            }
        }

        const auto t3 = Clock::now();
        timings_.preprocess_ms = elapsed_ms(t0, t1);
        timings_.inference_ms = elapsed_ms(t1, t2);
        timings_.postprocess_ms = elapsed_ms(t2, t3);
        LOG_DEBUG << "инференс: pre " << elapsed_ms(t0, t1) << " мс, run " << elapsed_ms(t1, t2)
                  << " мс, post " << elapsed_ms(t2, t3) << " мс, объектов " << out.size();
        return true;
    } catch (const Ort::Exception& e) {
        error = std::string("ONNX Runtime: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        error = std::string("ошибка инференса: ") + e.what();
        return false;
    }
}

}  // namespace vt
