#include "model/segmenter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "logger.hpp"

namespace vt {
namespace {

constexpr int kMaskCoeffs = 32;  // сколько коэффициентов масок отдаёт YOLO-seg

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// IoU = площадь пересечения / площадь объединения.
float iou(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int w = std::max(0, x2 - x1);
    const int h = std::max(0, y2 - y1);
    const float intersection = static_cast<float>(w * h);
    const float area_a = static_cast<float>(a.width * a.height);
    const float area_b = static_cast<float>(b.width * b.height);
    const float uni = area_a + area_b - intersection;
    return uni > 0.0f ? intersection / uni : 0.0f;
}

}  // namespace

struct Segmenter::OrtHolder {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "videotest"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory{"Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault};
};

Segmenter::Segmenter(ModelConfig config) : config_(std::move(config)) {
    empty_ = config_.path.empty();
}

Segmenter::~Segmenter() {
    if (ort_ != nullptr) {
        delete ort_;
    }
}

bool Segmenter::load(std::string& error) {
    // Пустая модель: ничего не загружаем, infer() вернёт ноль объектов, и
    // визуализация покажет два одинаковых потока. Отдельного режима не нужно.
    if (empty_) {
        LOG_WARN << "модель не задана: разметка будет пустой, потоки совпадут";
        return true;
    }
    if (config_.out_layout != "raw" && config_.out_layout != "nms") {
        error = "out_layout должен быть raw или nms, получено '" + config_.out_layout + "'";
        return false;
    }

    ort_ = new OrtHolder();
    try {
        ort_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_->options.SetIntraOpNumThreads(config_.threads);
        ort_->session =
            std::make_unique<Ort::Session>(ort_->env, config_.path.c_str(), ort_->options);

        for (std::size_t i = 0; i < ort_->session->GetInputCount(); ++i) {
            input_names_.emplace_back(
                ort_->session->GetInputNameAllocated(i, ort_->allocator).get());
        }
        for (std::size_t i = 0; i < ort_->session->GetOutputCount(); ++i) {
            output_names_.emplace_back(
                ort_->session->GetOutputNameAllocated(i, ort_->allocator).get());
        }
        for (const auto& name : input_names_) input_ptrs_.push_back(name.c_str());
        for (const auto& name : output_names_) output_ptrs_.push_back(name.c_str());

        const auto shape =
            ort_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_size_ = (shape.size() == 4 && shape[2] > 0) ? static_cast<int>(shape[2])
                                                          : config_.input_size;
    } catch (const Ort::Exception& e) {
        error = std::string("ONNX Runtime: ") + e.what();
        return false;
    }

    // Печатаем фактические формы выходов: если они не совпадают с выбранным
    // out_layout, ошибка будет видна сразу, а не в виде «пустой разметки».
    for (std::size_t i = 0; i < output_names_.size(); ++i) {
        const auto s = ort_->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::string text;
        for (const auto dim : s) text += (text.empty() ? "" : ",") + std::to_string(dim);
        LOG_INFO << "выход " << output_names_[i] << " [" << text << ']';
    }
    LOG_INFO << "модель: " << config_.path << ", вход " << input_size_ << "x" << input_size_
             << ", layout " << config_.out_layout << ", классов " << config_.class_count;
    loaded_ = true;
    return true;
}

void Segmenter::preprocess(const cv::Mat& bgr, cv::Mat& blob, float& scale,
                           cv::Point2f& pad) const {
    const int size = input_size_;
    // Letterbox: вписываем кадр в квадрат с сохранением пропорций, поля серые
    // (114 — как при обучении YOLO). Без этого маски после обратного
    // преобразования «уедут» по одной из осей.
    scale = std::min(static_cast<float>(size) / static_cast<float>(bgr.cols),
                     static_cast<float>(size) / static_cast<float>(bgr.rows));
    const int new_w = static_cast<int>(std::lround(static_cast<float>(bgr.cols) * scale));
    const int new_h = static_cast<int>(std::lround(static_cast<float>(bgr.rows) * scale));
    pad = {static_cast<float>(size - new_w) * 0.5f, static_cast<float>(size - new_h) * 0.5f};

    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));
    resized.copyTo(canvas(cv::Rect(static_cast<int>(pad.x), static_cast<int>(pad.y), new_w, new_h)));

    // BGR->RGB (swapRB), HWC->NCHW, /255. Выход — непрерывный CV_32F blob,
    // память которого отдаётся в Ort::Value без промежуточной копии.
    cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(size, size), cv::Scalar(),
                           /*swapRB=*/true, /*crop=*/false, CV_32F);
}

void Segmenter::decode_raw(const float* pred, int64_t channels, int64_t anchors,
                           std::vector<Instance>& dets,
                           std::vector<std::vector<float>>& coeffs) const {
    // Раскладка [1, 4+nc+32, N] — канал-мажорная: значение канала c для якоря a
    // лежит по индексу c*N + a. Порядок столбцов: cx, cy, w, h, nc вероятностей
    // классов, 32 коэффициента масок.
    if (channels < 4 + config_.class_count) {
        return;
    }
    for (int64_t a = 0; a < anchors; ++a) {
        float best = 0.0f;
        int best_class = -1;
        for (int c = 0; c < config_.class_count; ++c) {
            const float value = pred[(4 + c) * anchors + a];
            if (value > best) {
                best = value;
                best_class = c;
            }
        }
        if (best_class < 0 || best < config_.score_threshold) {
            continue;
        }
        const float w = pred[2 * anchors + a];
        const float h = pred[3 * anchors + a];
        if (w <= 0.0f || h <= 0.0f) {
            continue;
        }
        Instance det;
        det.score = best;
        det.class_id = best_class;
        det.box = cv::Rect(cvRound(pred[a] - w * 0.5f), cvRound(pred[anchors + a] - h * 0.5f),
                           cvRound(w), cvRound(h));

        std::vector<float> det_coeffs;
        if (channels >= 4 + config_.class_count + kMaskCoeffs) {
            const int64_t base = (4 + config_.class_count) * anchors;
            det_coeffs.resize(kMaskCoeffs);
            for (int k = 0; k < kMaskCoeffs; ++k) {
                det_coeffs[static_cast<std::size_t>(k)] = pred[base + k * anchors + a];
            }
        }
        dets.push_back(std::move(det));
        coeffs.push_back(std::move(det_coeffs));
    }
}

void Segmenter::decode_nms(const float* pred, int64_t rows, int64_t cols,
                           std::vector<Instance>& dets,
                           std::vector<std::vector<float>>& coeffs) const {
    // Раскладка [1, N, 6+32]: NMS уже выполнен в графе (YOLO26, one-to-one head).
    // Столбцы: x1, y1, x2, y2, score, class_id, 32 коэффициента масок.
    if (cols < 6) {
        return;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = pred + r * cols;
        const float score = row[4];
        if (score < config_.score_threshold) {
            continue;
        }
        Instance det;
        det.score = score;
        det.class_id = static_cast<int>(row[5]);
        det.box = cv::Rect(cvRound(row[0]), cvRound(row[1]), cvRound(row[2] - row[0]),
                           cvRound(row[3] - row[1]));

        std::vector<float> det_coeffs;
        if (cols >= 6 + kMaskCoeffs) {
            det_coeffs.assign(row + 6, row + 6 + kMaskCoeffs);
        }
        dets.push_back(std::move(det));
        coeffs.push_back(std::move(det_coeffs));
    }
}

void Segmenter::suppress(std::vector<Instance>& dets,
                         std::vector<std::vector<float>>& coeffs) const {
    // Жадный class-aware NMS: сортируем по уверенности, оставляем лучший бокс и
    // гасим перекрывающиеся с ним ТОГО ЖЕ класса. Объекты разных классов не
    // подавляют друг друга (человек и чашка часто перекрываются).
    if (dets.size() < 2) {
        return;
    }
    std::vector<std::size_t> order(dets.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return dets[a].score > dets[b].score; });

    std::vector<bool> dead(dets.size(), false);
    std::vector<Instance> kept;
    std::vector<std::vector<float>> kept_coeffs;
    for (const std::size_t i : order) {
        if (dead[i]) {
            continue;
        }
        kept.push_back(std::move(dets[i]));
        kept_coeffs.push_back(std::move(coeffs[i]));
        for (const std::size_t k : order) {
            if (k != i && !dead[k] && dets[k].class_id == kept.back().class_id &&
                iou(kept.back().box, dets[k].box) > config_.nms_threshold) {
                dead[k] = true;
            }
        }
    }
    dets = std::move(kept);
    coeffs = std::move(kept_coeffs);
}

void Segmenter::make_masks(const std::vector<Instance>& dets,
                           const std::vector<std::vector<float>>& coeffs, const float* protos,
                           const std::vector<int64_t>& proto_shape, float scale,
                           const cv::Point2f& pad, const cv::Size& frame_size,
                           std::vector<Instance>& out) const {
    // Прототипы: [1, 32, ph, pw]. Маска объекта = sigmoid(сумма_k coeff[k]*proto[k]),
    // затем обрезка по боксу, масштабирование в кадр и порог.
    if (proto_shape.size() != 4 || proto_shape[1] < kMaskCoeffs) {
        out = dets;  // прототипов недостаточно — оставляем только боксы
        return;
    }
    const int ph = static_cast<int>(proto_shape[2]);
    const int pw = static_cast<int>(proto_shape[3]);
    const std::size_t plane = static_cast<std::size_t>(ph) * pw;

    for (std::size_t i = 0; i < dets.size(); ++i) {
        Instance instance = dets[i];
        if (coeffs[i].size() != kMaskCoeffs) {
            out.push_back(std::move(instance));  // маски нет, бокс остаётся
            continue;
        }

        // 1. Линейная комбинация 32 прототипов -> float-маска размера прототипов.
        cv::Mat combined = cv::Mat::zeros(ph, pw, CV_32F);
        for (int k = 0; k < kMaskCoeffs; ++k) {
            const float coeff = coeffs[i][static_cast<std::size_t>(k)];
            if (coeff == 0.0f) {
                continue;
            }
            const float* src = protos + static_cast<std::size_t>(k) * plane;
            for (int y = 0; y < ph; ++y) {
                const float* row_src = src + static_cast<std::size_t>(y) * pw;
                float* row_dst = combined.ptr<float>(y);
                for (int x = 0; x < pw; ++x) {
                    row_dst[x] += coeff * row_src[x];
                }
            }
        }

        // 2. Бокс из координат сети -> координаты исходного кадра.
        const cv::Rect2f box_in_frame(
            (static_cast<float>(instance.box.x) - pad.x) / scale,
            (static_cast<float>(instance.box.y) - pad.y) / scale,
            static_cast<float>(instance.box.width) / scale,
            static_cast<float>(instance.box.height) / scale);
        const cv::Rect2f clipped = box_in_frame & cv::Rect2f(0, 0, frame_size.width,
                                                            frame_size.height);
        if (clipped.width < 1.0f || clipped.height < 1.0f) {
            out.push_back(std::move(instance));
            continue;
        }

        // 3. Обрезка прототипа по боксу (в масштабе прототипов) и ресайз к боксу.
        const float ratio_x = static_cast<float>(pw) / static_cast<float>(input_size_);
        const float ratio_y = static_cast<float>(ph) / static_cast<float>(input_size_);
        const cv::Rect crop(
            static_cast<int>(std::floor((static_cast<float>(instance.box.x)) * ratio_x)),
            static_cast<int>(std::floor((static_cast<float>(instance.box.y)) * ratio_y)),
            std::max(1, static_cast<int>(std::ceil(static_cast<float>(instance.box.width) * ratio_x))),
            std::max(1, static_cast<int>(std::ceil(static_cast<float>(instance.box.height) * ratio_y))));
        const cv::Rect safe_crop = crop & cv::Rect(0, 0, pw, ph);
        if (safe_crop.width < 1 || safe_crop.height < 1) {
            out.push_back(std::move(instance));
            continue;
        }
        cv::Mat resized;
        cv::resize(combined(safe_crop), resized,
                   cv::Size(std::max(1, cvRound(clipped.width)),
                            std::max(1, cvRound(clipped.height))));

        // 4. Порог в логитах: sigmoid(z) > t  <=>  z > ln(t/(1-t)).
        // Сравнивать надо именно логиты (resized), а не вероятности — так на
        // каждый пиксель приходится одно сравнение вместо exp().
        const float logit = std::log(config_.mask_threshold / (1.0f - config_.mask_threshold));
        cv::Mat binary;
        cv::compare(resized, logit, binary, cv::CMP_GT);

        // 5. Вставляем маску в пустое изображение размера кадра.
        cv::Mat mask = cv::Mat::zeros(frame_size, CV_8UC1);
        const int x = std::max(0, cvRound(clipped.x));
        const int y = std::max(0, cvRound(clipped.y));
        const int w = std::min(binary.cols, mask.cols - x);
        const int h = std::min(binary.rows, mask.rows - y);
        if (w > 0 && h > 0) {
            binary(cv::Rect(0, 0, w, h)).copyTo(mask(cv::Rect(x, y, w, h)));
        }
        instance.mask = std::move(mask);
        instance.box = cv::Rect(cvRound(clipped.x), cvRound(clipped.y), cvRound(clipped.width),
                                cvRound(clipped.height));
        out.push_back(std::move(instance));
    }
}

bool Segmenter::infer(const cv::Mat& bgr, std::vector<Instance>& out, double* infer_ms,
                      std::string& error) {
    out.clear();
    if (infer_ms != nullptr) {
        *infer_ms = 0.0;
    }
    if (empty_) {
        return true;  // «пустая разметка»: поток останется исходным
    }
    if (!loaded_ || ort_ == nullptr || !ort_->session) {
        error = "модель не загружена";
        return false;
    }

    try {
        const double t0 = now_ms();
        cv::Mat blob;
        float scale = 1.0f;
        cv::Point2f pad{0.0f, 0.0f};
        preprocess(bgr, blob, scale, pad);

        const std::array<int64_t, 4> shape{1, 3, input_size_, input_size_};
        // Ort::Value ссылается на память blob'а (без копирования) и живёт до Run().
        Ort::Value input = Ort::Value::CreateTensor<float>(
            ort_->memory, blob.ptr<float>(), static_cast<std::size_t>(blob.total()), shape.data(),
            shape.size());
        std::vector<Ort::Value> outputs =
            ort_->session->Run(Ort::RunOptions{nullptr}, input_ptrs_.data(), &input, 1,
                               output_ptrs_.data(), output_ptrs_.size());

        const auto pred_shape =
            outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const float* pred = outputs[0].GetTensorData<float>();
        std::vector<Instance> dets;
        std::vector<std::vector<float>> coeffs;

        if (config_.out_layout == "raw") {
            if (pred_shape.size() != 3) {
                error = "layout raw ожидает выход [1, C, N], получено другое число измерений";
                return false;
            }
            decode_raw(pred, pred_shape[1], pred_shape[2], dets, coeffs);
            suppress(dets, coeffs);  // в raw-выходе дубликаты ещё не подавлены
        } else {
            if (pred_shape.size() != 3) {
                error = "layout nms ожидает выход [1, N, C]";
                return false;
            }
            decode_nms(pred, pred_shape[1], pred_shape[2], dets, coeffs);
        }

        // Прототипы масок — второй выход. Если его нет \emph{или} он не четырёхмерный,
        // это не повод терять детекции: отдаём хотя бы боксы. Иначе объекты молча
        // исчезали бы, и причина была бы не видна.
        const bool has_protos =
            outputs.size() > 1 && outputs[1].GetTensorTypeAndShapeInfo().GetShape().size() == 4;
        if (has_protos) {
            const auto proto_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            if (proto_shape[1] != kMaskCoeffs) {
                LOG_WARN << "прототипы: ожидалось " << kMaskCoeffs << " каналов, получено "
                         << proto_shape[1] << " — маски будут пустыми";
            }
            make_masks(dets, coeffs, outputs[1].GetTensorData<float>(), proto_shape, scale, pad,
                       bgr.size(), out);
        } else {
            if (outputs.size() > 1) {
                LOG_WARN << "второй выход не является картой прототипов — только боксы";
            }
            out = std::move(dets);  // модель без прототипов: только боксы
        }

        if (infer_ms != nullptr) {
            *infer_ms = now_ms() - t0;
        }
        return true;
    } catch (const Ort::Exception& e) {
        // Исключение не должно вылетать из потока: std::thread превратит его
        // в std::terminate(). Переводим в код возврата + текст.
        error = std::string("ONNX Runtime: ") + e.what();
        return false;
    }
}

}  // namespace vt
