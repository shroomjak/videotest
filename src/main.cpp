// videotest: камера -> ONNX instance-segmentation -> трекинг -> два потока рядом.
//
// Вся программа — это:
//   1. явный разбор аргументов (формат камеры и layout модели задаются строкой);
//   2. V4l2Source + Segmenter + Tracker;
//   3. Pipeline (два потока и одна очередь между ними);
//   4. цикл отображения в main-потоке.
// Никакой автоматики: если формат не поддержан — будет ошибка, а не тихий
// подбор другого. Так видно, что именно происходит.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "frame.hpp"
#include "logger.hpp"
#include "model/class_names.hpp"
#include "pipeline.hpp"
#include "render/renderer.hpp"
#include "stop_flag.hpp"
#include "streams/synthetic_source.hpp"
#include "streams/v4l2_source.hpp"

namespace {

using namespace vt;

struct Args {
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    std::string format = "MJPG";  // формат камеры: MJPG | YUYV

    std::string model;            // путь к .onnx; пусто — пустая разметка
    std::string out_layout = "raw";  // формат выхода модели: raw | nms
    std::string classes = "80";   // число классов ИЛИ путь к файлу имён
    float score = 0.35f;
    float nms = 0.45f;
    int threads = 2;

    bool synthetic = false;   // синтетический источник вместо камеры
    bool show = true;         // false — работаем без окна (для записи)
    int max_frames = 0;       // 0 — без ограничения (для смоук-тестов)
    std::string snapshot;     // если задан — сохранить один кадр в png

    // Заполняется после разбора --classes.
    int class_count = 80;
    std::vector<std::string> class_names;

    bool help = false;
};

void print_help(const char* program) {
    std::cout
        << "videotest — камера -> ONNX instance segmentation + трекинг -> визуализация\n\n"
        << "Использование: " << program << " [опции]\n\n"
        << "Камера (формат задаётся явно, автоподбора нет):\n"
        << "  --device /dev/videoX   устройство (по умолчанию /dev/video0)\n"
        << "  --width N --height N   запрашиваемый размер (по умолчанию 1280x720)\n"
        << "  --fps N                запрашиваемая частота (30)\n"
        << "  --format MJPG|YUYV     формат пикселей (MJPG)\n\n"
        << "Модель:\n"
        << "  --model path.onnx      если не задана — разметка пустая, потоки совпадают\n"
        << "  --out-layout raw|nms   формат выхода: raw = [1,4+nc+32,N] (NMS в C++),\n"
        << "                         nms = [1,N,6+32] (NMS уже в графе, YOLO26)\n"
        << "  --classes N|file.txt    число классов (80) ИЛИ файл с именами,\n"
        << "                         по одному в строке; при 80 подставляются\n"
        << "                         стандартные имена COCO\n"
        << "  --score F              порог уверенности (0.35)\n"
        << "  --nms F                порог подавления перекрытий (0.45)\n"
        << "  --threads N            intra-op потоков ONNX Runtime (2)\n\n"
        << "Отладка:\n"
        << "  --synthetic            синтетический источник вместо камеры\n"
        << "  --max-frames N         выйти после N показанных кадров (0 — без предела)\n"
        << "  --snapshot out.png     сохранить один кадр с разметкой\n\n"
        << "Отображение:\n"
        << "  --no-window            без окна (полезно на сервере)\n"
        << "  --help                 эта справка\n\n"
        << "В окне: q/Esc — выход, m — маски, b — боксы.\n"
        << "Формат выхода можно проверить в логе: программа печатает формы выходов.\n";
}

// Разбор аргументов: только явные ключи, никаких эвристик.
bool parse_args(int argc, char** argv, Args& args, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                error = std::string("для ") + name + " не указано значение";
                return {};
            }
            return argv[++i];
        };

        if (key == "--help" || key == "-h") {
            args.help = true;
        } else if (key == "--device") {
            args.device = value("--device");
        } else if (key == "--width") {
            args.width = std::stoi(value("--width"));
        } else if (key == "--height") {
            args.height = std::stoi(value("--height"));
        } else if (key == "--fps") {
            args.fps = std::stoi(value("--fps"));
        } else if (key == "--format") {
            args.format = value("--format");
        } else if (key == "--model") {
            args.model = value("--model");
        } else if (key == "--out-layout") {
            args.out_layout = value("--out-layout");
        } else if (key == "--classes") {
            args.classes = value("--classes");
        } else if (key == "--score") {
            args.score = std::stof(value("--score"));
        } else if (key == "--nms") {
            args.nms = std::stof(value("--nms"));
        } else if (key == "--threads") {
            args.threads = std::stoi(value("--threads"));
        } else if (key == "--synthetic") {
            args.synthetic = true;
        } else if (key == "--max-frames") {
            args.max_frames = std::stoi(value("--max-frames"));
        } else if (key == "--snapshot") {
            args.snapshot = value("--snapshot");
        } else if (key == "--no-window") {
            args.show = false;
        } else {
            error = "неизвестная опция: " + key + " (см. --help)";
            return false;
        }
        if (!error.empty()) {
            return false;
        }
    }

    // Явная валидация перечислимых значений — до создания объектов.
    if (args.format != "MJPG" && args.format != "YUYV") {
        error = "--format: поддерживаются MJPG и YUYV, получено '" + args.format + "'";
        return false;
    }
    if (args.out_layout != "raw" && args.out_layout != "nms") {
        error = "--out-layout: поддерживаются raw и nms, получено '" + args.out_layout + "'";
        return false;
    }
    if (args.width <= 0 || args.height <= 0 || args.fps <= 0) {
        error = "ширина, высота и fps должны быть положительными";
        return false;
    }

    // --classes: либо число, либо путь к файлу имён. Разбираем здесь, до
    // создания объектов, чтобы ошибка была про аргументы, а не про модель.
    const bool all_digits =
        !args.classes.empty() &&
        std::all_of(args.classes.begin(), args.classes.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; });
    if (all_digits) {
        args.class_count = std::stoi(args.classes);
        if (args.class_count <= 0) {
            error = "--classes: число классов должно быть положительным";
            return false;
        }
        // 80 — стандартный набор COCO: подставляем имена, чтобы подписи были
        // читаемыми без отдельного файла.
        if (args.class_count == static_cast<int>(coco_names().size())) {
            args.class_names = coco_names();
        }
    } else {
        if (!load_class_names(args.classes, args.class_names, error)) {
            return false;
        }
        args.class_count = static_cast<int>(args.class_names.size());
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    std::string error;
    if (!parse_args(argc, argv, args, error)) {
        std::cerr << "ошибка аргументов: " << error << "\n\n";
        return 2;
    }
    if (args.help) {
        print_help(argv[0]);
        return 0;
    }

    install_signal_handler();

    // --- 1. источник кадров ---------------------------------------------------
    // Интерфейс у обоих источников одинаковый (open/grab/width/height), поэтому
    // конвейер ниже не знает, откуда приходят кадры.
    V4l2Source camera_source(V4l2Config{args.device, args.width, args.height, args.fps,
                                        args.format});
    SyntheticSource synthetic_source(SyntheticConfig{args.width, args.height, args.fps});
    // Обе реализации — Source; указатель выбирается один раз при старте.
    Source& source = args.synthetic ? static_cast<Source&>(synthetic_source)
                                    : static_cast<Source&>(camera_source);
    if (!source.open(error)) {
        LOG_ERROR << error;
        return 1;
    }

    // --- 2. модель (пустая, если путь не задан) -------------------------------
    ModelConfig model_config;
    model_config.path = args.model;
    model_config.input_size = 640;
    model_config.out_layout = args.out_layout;
    model_config.class_count = args.class_count;
    model_config.score_threshold = args.score;
    model_config.nms_threshold = args.nms;
    model_config.threads = args.threads;
    Segmenter model(model_config);
    if (!model.load(error)) {
        LOG_ERROR << error;
        return 1;
    }

    // --- 3. трекер и конвейер -------------------------------------------------
    Tracker tracker(TrackerConfig{});
    Pipeline pipeline(source, model, tracker);
    pipeline.start();

    // --- 4. окно и цикл отображения ------------------------------------------
    const std::string window = "videotest";
    bool show = args.show;
    if (show) {
        try {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, source.width() * 2, source.height());
        } catch (const cv::Exception& e) {
            // HighGUI без backend'а (сервер без X11) — не повод падать.
            LOG_WARN << "окно недоступно (" << e.what() << "), работаем без окна";
            show = false;
        }
    }

    ViewOptions view;
    view.class_names = args.class_names;  // подписи "#id имя score"
    if (!view.class_names.empty()) {
        LOG_INFO << "классы: " << view.class_names.size() << " имён, например "
                 << view.class_names.front();
    }
    // Метрики: раз в полсекунды пересчитываем скорость по приросту счётчиков
    // конвейера. last_rate сдвигается только при пересчёте, поэтому окно
    // измерения всегда ~0.5 c.
    auto last_rate = std::chrono::steady_clock::now();
    std::uint64_t last_captured = 0;
    std::uint64_t last_processed = 0;
    double capture_fps = 0.0;
    double process_fps = 0.0;
    std::uint64_t shown = 0;
    // Снимок берём после прогрева, но с поправкой на --max-frames: иначе при
    // коротком прогоне кадр номер 25 просто не наступит.
    std::uint64_t snapshot_at = 26;
    if (args.max_frames > 0) {
        snapshot_at = std::min<std::uint64_t>(26, static_cast<std::uint64_t>(args.max_frames));
    }
    bool snapshot_taken = false;
    // Задержка «от драйвера до разметки»: именно она должна оставаться
    // постоянной при перегрузке — это и есть смысл вытеснения в очереди.
    double latency_first = -1.0;
    double latency_max = 0.0;
    double latency_sum = 0.0;

    while (!stop_requested() && pipeline.running()) {
        ResultPtr result;
        if (auto maybe = pipeline.results().pop(std::chrono::milliseconds(10))) {
            result = std::move(*maybe);
        }

        // Метрики обновляем ДО проверки result: иначе при отсутствии нового
        // кадра счётчики не пересчитываются и HUD "замерзает" на нулях.
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_ms = ms_between(last_rate, now);
        if (elapsed_ms >= 500.0) {
            const auto& stats = pipeline.stats();
            const auto captured = stats.captured.load();
            const auto processed = stats.processed.load();
            capture_fps = static_cast<double>(captured - last_captured) * 1000.0 / elapsed_ms;
            process_fps = static_cast<double>(processed - last_processed) * 1000.0 / elapsed_ms;
            last_captured = captured;
            last_processed = processed;
            last_rate = now;
        }

        if (!result) {
            if (show) {
                int key = -1;
                try {
                    key = cv::waitKey(10) & 0xFF;
                } catch (const cv::Exception&) {
                    show = false;
                }
                if (key == 'q' || key == 27) {
                    break;
                }
                if (key == 'm') view.show_masks = !view.show_masks;
                if (key == 'b') view.show_boxes = !view.show_boxes;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            continue;
        }

        // Исходный кадр и размеченный кадр: одна и та же картинка, разметка
        // только на второй половине. Пустая разметка дает идентичные половины.
        const cv::Mat& raw = result->frame->image;
        const cv::Mat overlay = draw_overlay(raw, result->instances, view);
        cv::Mat canvas = compose_side_by_side(raw, overlay);

        if (latency_first < 0.0) {
            latency_first = result->latency_ms;
        }
        latency_max = std::max(latency_max, result->latency_ms);
        latency_sum += result->latency_ms;

        const auto& stats = pipeline.stats();
        char line[160];
        std::vector<std::string> hud;
        std::snprintf(line, sizeof(line), "capture %.1f fps | process %.1f fps | showed %llu",
                      capture_fps, process_fps, static_cast<unsigned long long>(shown));
        hud.emplace_back(line);
        std::snprintf(line, sizeof(line), "infer %.2f ms | latency %.1f ms | queue %zu",
                      result->infer_ms, result->latency_ms, pipeline.results().size());
        hud.emplace_back(line);
        std::snprintf(line, sizeof(line), "dropped %llu | objects %zu | tracks %zu",
                      static_cast<unsigned long long>(stats.dropped.load()),
                      result->instances.size(), tracker.active_tracks());
        hud.emplace_back(line);
        draw_hud(canvas, hud);

        // --snapshot: сохраняем кадр с разметкой, чтобы можно было посмотреть
        // результат без окна (например, на сервере). После прогрева, иначе
        // трекер только успел создать треки.
        if (!args.snapshot.empty() && !snapshot_taken && shown + 1 >= snapshot_at) {
            if (cv::imwrite(args.snapshot, canvas)) {
                LOG_INFO << "снимок сохранён: " << args.snapshot;
                snapshot_taken = true;
            } else {
                LOG_WARN << "не удалось сохранить снимок: " << args.snapshot;
            }
        }

        if (show) {
            int key = -1;
            try {
                cv::imshow(window, canvas);
                key = cv::waitKey(1) & 0xFF;
            } catch (const cv::Exception&) {
                show = false;
            }
            if (key == 'q' || key == 27) {
                break;
            }
            if (key == 'm') view.show_masks = !view.show_masks;
            if (key == 'b') view.show_boxes = !view.show_boxes;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ++shown;
        if (args.max_frames > 0 && shown >= static_cast<std::uint64_t>(args.max_frames)) {
            break;
        }
    }

    // --- 5. корректная остановка ---------------------------------------------
    stop_flag().store(true);
    pipeline.stop();
    // Окно могло так и не появиться (сервер без X11), поэтому вызов защищён.
    try {
        cv::destroyAllWindows();
    } catch (const cv::Exception&) {
        // окна нет (сервер без X11) — закрывать нечего
    }

    const auto& stats = pipeline.stats();
    LOG_INFO << "итого: получено " << stats.captured.load() << ", обработано "
             << stats.processed.load() << ", вытеснено " << stats.dropped.load()
             << ", показано " << shown;
    if (shown > 0) {
        // Главная проверка realtime-схемы: задержка не должна расти со временем.
        // Если она ползёт вверх — очередь копится, и вытеснение не работает.
        LOG_INFO << "задержка, мс: первая " << latency_first << ", средняя "
                 << latency_sum / static_cast<double>(shown) << ", максимальная "
                 << latency_max;
    }
    return 0;
}
