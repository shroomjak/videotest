// videotest: камера (/dev/videoX) -> ONNX instance segmentation -> визуализация.
//
// Точка входа намеренно тонкая: разбор аргументов, создание объектов, цикл
// отображения. Вся предметная логика — в модулях video/, infer/, pipeline/,
// render/, чтобы её можно было читать и тестировать по отдельности.

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "app/cli.hpp"
#include "core/logger.hpp"
#include "core/signal_flag.hpp"
#include "core/stats.hpp"
#include "infer/onnx_segmenter.hpp"
#include "pipeline/pipeline.hpp"
#include "render/renderer.hpp"
#include "video/file_source.hpp"
#include "video/v4l2_source.hpp"

namespace {

using namespace vt;

// «Пустая модель»: позволяет запустить конвейер без ONNX-файла — удобно, чтобы
// сначала проверить камеру, а уже потом подключать сеть.
class NullSegmenter final : public ISegmenter {
public:
    NullSegmenter() {
        info_.path = "<none>";
        info_.task = "capture-only";
        info_.providers = "-";
    }
    bool load(std::string&) override { return true; }
    bool segment(const cv::Mat&, std::vector<Instance>& out, std::string&) override {
        out.clear();
        return true;
    }
    const std::vector<std::string>& class_names() const override { return empty_; }
    ModelInfo info() const override { return info_; }

private:
    std::vector<std::string> empty_;
    ModelInfo info_;
};

SourcePtr make_source(const AppConfig& config) {
    if (config.synthetic) {
        return std::make_unique<SyntheticSource>(config.width, config.height, config.fps);
    }
    if (!config.video_file.empty()) {
        return std::make_unique<FileSource>(config.video_file, config.loop_file, config.pace_file);
    }
    V4l2Config v4l2;
    v4l2.device = config.device;
    v4l2.width = config.width;
    v4l2.height = config.height;
    v4l2.fps = config.fps;
    v4l2.prefer_mjpeg = config.prefer_mjpeg;
    return std::make_unique<V4l2Source>(v4l2);
}

SegmenterPtr make_segmenter(const AppConfig& config, std::string& error) {
    if (config.model_path.empty()) {
        LOG_WARN << "модель не задана (--model) — будет только поток камеры без разметки";
        return std::make_unique<NullSegmenter>();
    }
#if !defined(VIDEOTEST_WITH_ONNX)
    // Сборка без ONNX Runtime (-DVIDEOTEST_WITH_ONNX=OFF): инференс недоступен,
    // но камера, конвейер и отрисовка работают. Такой режим удобен, чтобы
    // сначала убедиться, что камера открывается и картинка идёт.
    (void)config;
    error = "программа собрана без ONNX Runtime (VIDEOTEST_WITH_ONNX=OFF), "
            "инференс недоступен — пересоберите с поддержкой ONNX";
    return nullptr;
#else
    SegmenterConfig segmenter_config;
    segmenter_config.model_path = config.model_path;
    segmenter_config.score_threshold = config.score_threshold;
    segmenter_config.nms_threshold = config.nms_threshold;
    segmenter_config.mask_threshold = config.mask_threshold;
    segmenter_config.intra_op_threads = config.infer_threads;
    segmenter_config.device = config.device_for_infer;
    auto segmenter = std::make_unique<OnnxSegmenter>(segmenter_config);
    if (!segmenter->load(error)) {
        return nullptr;
    }
    return segmenter;
#endif
}

RenderOptions render_options_from(const AppConfig& config) {
    RenderOptions options;
    options.show_masks = config.show_masks;
    options.show_boxes = config.show_boxes;
    options.show_labels = true;
    options.show_contours = config.show_contours;
    options.mask_alpha = 0.45f;
    // Порог отрисовки чуть ниже порога модели: рамка «на грани» видна, но
    // бледная — это помогает настраивать --score, не перезапуская программу.
    options.score_threshold = config.score_threshold * 0.85f;
    return options;
}

// Подбирает кодек по расширению файла. VideoWriter молча не открывается, если
// нужного кодека нет в сборке OpenCV (например, пакет собран без FFmpeg —
// тогда нет ни mp4v, ни XVID, но MJPG пишется). Поэтому пробуем список
// кандидатов по порядку и берём первый, который реально открылся.
bool open_recorder(cv::VideoWriter& writer, const std::string& path, double fps,
                   const cv::Size& size) {
    const std::string lower = [&path] {
        std::string text = path;
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }();
    const bool mp4 = lower.size() > 4 && lower.compare(lower.size() - 4, 4, ".mp4") == 0;

    // Для .mp4 предпочитаем H.264/mp4v, для остальных контейнеров — MJPG:
    // он есть почти в любой сборке OpenCV (внутренний JPEG-кодек).
    std::vector<int> candidates;
    if (mp4) {
        candidates.push_back(cv::VideoWriter::fourcc('m', 'p', '4', 'v'));
        candidates.push_back(cv::VideoWriter::fourcc('a', 'v', 'c', '1'));
        candidates.push_back(cv::VideoWriter::fourcc('H', '2', '6', '4'));
    }
    candidates.push_back(cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!mp4) {
        candidates.push_back(cv::VideoWriter::fourcc('X', 'V', 'I', 'D'));
    }

    for (const int codec : candidates) {
        if (writer.open(path, cv::CAP_ANY, codec, fps, size)) {
            char text[5] = {static_cast<char>(codec & 0xFF), static_cast<char>((codec >> 8) & 0xFF),
                            static_cast<char>((codec >> 16) & 0xFF),
                            static_cast<char>((codec >> 24) & 0xFF), '\0'};
            LOG_INFO << "кодек: " << text;
            return true;
        }
    }
    return false;
}

// Пытается создать окно и не падает, если у HighGUI нет backend'а.
bool createWindowOk(const std::string& window, const SourceInfo& info, Layout layout) {
    try {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        const cv::Size initial((layout == Layout::SideBySide) ? info.width * 2 : info.width,
                               (layout == Layout::Stacked) ? info.height * 2 : info.height);
        cv::resizeWindow(window, initial);
        return true;
    } catch (const cv::Exception& e) {
        LOG_DEBUG << "namedWindow: " << e.what();
        return false;
    }
}

std::string format_seconds(double seconds) {
    const int total = static_cast<int>(seconds);
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02d:%02d", total / 60, total % 60);
    return buffer;
}

// Строки HUD — одна точка правды для того, что видит пользователь.
std::vector<std::string> make_hud_lines(const Pipeline& pipeline, const RateMeter& capture_rate,
                                        const RateMeter& infer_rate, double elapsed,
                                        const SourceInfo& info, std::size_t class_count) {
    const PipelineStats& stats = pipeline.stats();
    std::vector<std::string> lines;
    char line[192];

    std::snprintf(line, sizeof(line), "capture %5.1f fps | infer %5.1f fps | captured %llu",
                  capture_rate.value(), infer_rate.value(),
                  static_cast<unsigned long long>(
                      stats.frames_captured.load(std::memory_order_relaxed)));
    lines.emplace_back(line);

    std::snprintf(line, sizeof(line), "source %dx%d @%.0f %s | uptime %s", info.width, info.height,
                  info.fps, info.pixel_format.c_str(), format_seconds(elapsed).c_str());
    lines.emplace_back(line);

    const double infer_ms =
        static_cast<double>(stats.inference_us.load(std::memory_order_relaxed)) / 1000.0;
    const double latency_ms =
        static_cast<double>(stats.latency_us.load(std::memory_order_relaxed)) / 1000.0;
    std::snprintf(line, sizeof(line), "infer %5.2f ms | latency %5.1f ms | dropped %llu/%llu",
                  infer_ms, latency_ms,
                  static_cast<unsigned long long>(
                      stats.frames_dropped.load(std::memory_order_relaxed)),
                  static_cast<unsigned long long>(
                      stats.results_dropped.load(std::memory_order_relaxed)));
    lines.emplace_back(line);

    std::snprintf(line, sizeof(line), "queue %zu/%zu | objects %llu | classes %zu",
                  pipeline.capture_queue().size(), pipeline.capture_queue().capacity(),
                  static_cast<unsigned long long>(
                      stats.instances_total.load(std::memory_order_relaxed)),
                  class_count);
    lines.emplace_back(line);
    return lines;
}

}  // namespace

int main(int argc, char** argv) {
    AppConfig config;
    std::string error;
    if (!parse_args(argc, argv, config, error)) {
        std::cerr << "ошибка аргументов: " << error << "\n\n";
        print_help(std::cerr, argv[0]);
        return 2;
    }
    if (config.show_help) {
        print_help(std::cout, argv[0]);
        return 0;
    }
    set_log_level(config.log_level);
    LOG_INFO << "videotest: " << describe_config(config);

    install_signal_handlers();

    // --- источник -------------------------------------------------------------
    SourcePtr source = make_source(config);
    if (!source->open(error)) {
        LOG_ERROR << "источник: " << error;
        return 1;
    }
    const SourceInfo source_info = source->info();

    // --- модель ---------------------------------------------------------------
    SegmenterPtr segmenter = make_segmenter(config, error);
    if (!segmenter) {
        LOG_ERROR << "модель: " << error;
        return 1;
    }
    const ModelInfo model_info = segmenter->info();

    // --- конвейер -------------------------------------------------------------
    PipelineConfig pipeline_config;
    pipeline_config.artificial_infer_delay_ms = config.artificial_delay_ms;
    Pipeline pipeline(std::move(source), std::move(segmenter), pipeline_config);
    pipeline.start();

    Renderer renderer(render_options_from(config));

    const std::string window = "videotest";
    bool show_window = !config.headless;
    if (show_window) {
        // createWindowOk(): HighGUI может быть собран без backend'а (сервер без
        // X11/GTK, минимальная сборка OpenCV). Тогда namedWindow бросает
        // cv::Exception — это не повод падать: продолжаем без окна, оставив
        // возможность записи в файл.
        if (!createWindowOk(window, source_info, config.layout)) {
            LOG_WARN << "окно недоступно (HighGUI без backend'а) — продолжаем в режиме "
                        "--headless; используйте --record/--snapshot, чтобы увидеть результат";
            show_window = false;
        }
    }

    // Запись: размер мозаики зависит от раскладки, поэтому открываем writer
    // лениво — по первому готовому кадру, когда фактический размер известен.
    cv::VideoWriter recorder;
    bool recorder_tried = false;

    RateMeter capture_rate;
    RateMeter infer_rate;
    const auto started = SteadyClock::now();
    std::uint64_t displayed = 0;

    LOG_INFO << "старт визуализации: q/Esc — выход, s — снимок, m/b/c/h — слои, 1/2/3 — раскладка";
    while (!stop_requested() && pipeline.running()) {
        // --- забираем результат, не блокируясь надолго -------------------------
        ResultPtr result;
        if (auto maybe = pipeline.result_queue().pop_wait_for(std::chrono::milliseconds(10))) {
            result = std::move(*maybe);
        }

        const auto now = SteadyClock::now();
        capture_rate.update(pipeline.stats().frames_captured.load(std::memory_order_relaxed), now);
        infer_rate.update(pipeline.stats().frames_inferred.load(std::memory_order_relaxed), now);

        if (!result) {
            // Кадров ещё нет (или модель медленно стартует) — обрабатываем
            // клавиши, чтобы приложение оставалось отзывчивым.
            if (show_window) {
                int key = -1;
                try {
                    key = cv::waitKey(10) & 0xFF;
                } catch (const cv::Exception&) {
                    show_window = false;  // окно пропало — переходим в headless
                }
                if (key == 'q' || key == 27) {
                    break;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            continue;
        }

        // --- исходный кадр ----------------------------------------------------
        cv::Mat raw = result->frame->image ? *result->frame->image : cv::Mat();
        if (raw.empty()) {
            continue;
        }
        if (config.mirror) {
            cv::flip(raw, raw, 1);
            // Зеркалим и геометрию результата, чтобы разметка не «съехала».
            InferenceResult mirrored = *result;
            for (auto& instance : mirrored.instances) {
                if (!instance.mask.empty()) {
                    cv::flip(instance.mask, instance.mask, 1);
                }
                const float x = static_cast<float>(raw.cols) - instance.box.x - instance.box.width;
                instance.box.x = x;
            }
            result = std::make_shared<const InferenceResult>(std::move(mirrored));
        }

        // --- отрисовка --------------------------------------------------------
        const auto& class_names = pipeline.segmenter().class_names();
        const cv::Mat overlay = renderer.draw(raw, *result, class_names);
        cv::Mat canvas = renderer.compose(raw, overlay, config.layout);

        if (config.show_hud) {
            Renderer::draw_hud(canvas, make_hud_lines(pipeline, capture_rate, infer_rate,
                                                      seconds_between(started, SteadyClock::now()),
                                                      source_info, class_names.size()));
        }

        // --- запись -----------------------------------------------------------
        if (!config.record_path.empty()) {
            if (!recorder_tried) {
                recorder_tried = true;
                const double record_fps = source_info.fps > 0 ? source_info.fps : 30.0;
                if (!open_recorder(recorder, config.record_path, record_fps, canvas.size())) {
                    LOG_WARN << "не удалось открыть для записи " << config.record_path
                             << " — попробуйте расширение .avi (кодек MJPG)";
                } else {
                    LOG_INFO << "запись в " << config.record_path << " (" << canvas.cols << 'x'
                             << canvas.rows << " @ " << record_fps << " fps)";
                }
            }
            if (recorder.isOpened()) {
                recorder.write(canvas);
            }
        }

        // --- снимки -----------------------------------------------------------
        if (!config.snapshot_path.empty() && displayed == 0) {
            if (cv::imwrite(config.snapshot_path, canvas)) {
                LOG_INFO << "снимок сохранён: " << config.snapshot_path;
            } else {
                LOG_WARN << "не удалось сохранить снимок: " << config.snapshot_path;
            }
        }
        if (config.snapshots_every > 0 &&
            displayed % static_cast<std::uint64_t>(config.snapshots_every) == 0) {
            char name[64];
            std::snprintf(name, sizeof(name), "snapshot_%05llu.png",
                          static_cast<unsigned long long>(
                              displayed / static_cast<std::uint64_t>(config.snapshots_every)));
            cv::imwrite(name, canvas);
        }

        ++displayed;
        if (config.max_frames > 0 && displayed >= static_cast<std::uint64_t>(config.max_frames)) {
            LOG_INFO << "достигнут лимит --max-frames=" << config.max_frames;
            break;
        }

        // --- показ и клавиши --------------------------------------------------
        if (show_window) {
            int key = -1;
            try {
                cv::imshow(window, canvas);
                key = cv::waitKey(1) & 0xFF;
            } catch (const cv::Exception& e) {
                // Окно могло быть закрыто или backend отвалился — не падаем.
                LOG_DEBUG << "imshow: " << e.what();
                show_window = false;
            }
            if (key == 'q' || key == 27) {
                break;
            }
            switch (key) {
                case 's': {
                    char name[64];
                    std::snprintf(name, sizeof(name), "snapshot_%05llu.png",
                                  static_cast<unsigned long long>(displayed));
                    cv::imwrite(name, canvas);
                    LOG_INFO << "снимок сохранён: " << name;
                    break;
                }
                case 'm':
                    config.show_masks = !config.show_masks;
                    renderer = Renderer(render_options_from(config));
                    break;
                case 'b':
                    config.show_boxes = !config.show_boxes;
                    renderer = Renderer(render_options_from(config));
                    break;
                case 'c':
                    config.show_contours = !config.show_contours;
                    renderer = Renderer(render_options_from(config));
                    break;
                case 'h':
                    config.show_hud = !config.show_hud;
                    break;
                case '1':
                    config.layout = Layout::SideBySide;
                    break;
                case '2':
                    config.layout = Layout::Stacked;
                    break;
                case '3':
                    config.layout = Layout::OverlayOnly;
                    break;
                default:
                    break;
            }
        }
    }

    // --- корректное завершение ------------------------------------------------
    request_stop();
    pipeline.stop();
    if (recorder.isOpened()) {
        recorder.release();
    }
    if (show_window) {
        cv::destroyAllWindows();
    }

    const PipelineStats& stats = pipeline.stats();
    const double elapsed = seconds_between(started, SteadyClock::now());
    const std::uint64_t captured = stats.frames_captured.load(std::memory_order_relaxed);
    LOG_INFO << "итого: захвачено " << captured << ", обработано моделью "
             << stats.frames_inferred.load(std::memory_order_relaxed) << ", показано "
             << displayed << ", время " << format_seconds(elapsed) << ", средний capture fps ~ "
             << (elapsed > 0.0 ? static_cast<double>(captured) / elapsed : 0.0);
    LOG_INFO << "модель: " << model_info.task << ", классов " << model_info.class_count
             << ", вход " << model_info.input_width << 'x' << model_info.input_height;
    return 0;
}
