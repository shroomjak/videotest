#include "app/cli.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace vt {
namespace {

bool parse_int(const std::string& text, int& out) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parse_double(const std::string& text, double& out) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == nullptr || *end != '\0') {
        return false;
    }
    out = value;
    return true;
}

bool parse_float(const std::string& text, float& out) {
    double value = 0.0;
    if (!parse_double(text, value)) {
        return false;
    }
    out = static_cast<float>(value);
    return true;
}

bool parse_bool(const std::string& text, bool& out) {
    if (text == "1" || text == "true" || text == "yes" || text == "on") {
        out = true;
        return true;
    }
    if (text == "0" || text == "false" || text == "no" || text == "off") {
        out = false;
        return true;
    }
    return false;
}

// Есть ли у ключа значение. Список должен совпадать с ветками разбора ниже:
// если добавить новый ключ со значением и забыть здесь, парсер сочтёт его
// флагом и не съест следующий аргумент.
bool key_takes_value(const std::string& key) {
    static const char* const kValueKeys[] = {
        "--device",  "--file",      "--width",   "--height",   "--fps",
        "--model",   "--infer-device", "--infer-threads", "--score", "--nms",
        "--mask-threshold", "--layout", "--record", "--snapshot", "--snapshot-every",
        "--max-frames", "--delay-ms", "--log-level"};
    for (const char* candidate : kValueKeys) {
        if (key == candidate) {
            return true;
        }
    }
    return false;
}

}  // namespace

void print_help(std::ostream& out, const char* program) {
    out << "videotest — камера -> ONNX instance segmentation -> отображение в реальном времени\n\n"
        << "Использование:\n  " << program << " [опции]\n\n"
        << "Источник (одно из трёх):\n"
        << "  --device /dev/videoX      V4L2-камера (по умолчанию /dev/video0)\n"
        << "  --file path.mp4           видеофайл вместо камеры (--no-loop, --fast)\n"
        << "  --synthetic               синтетический генератор кадров (без камеры)\n"
        << "  --width N --height N --fps N   запрашиваемый режим камеры\n"
        << "  --no-mjpeg                не просить MJPG, брать сырой YUYV\n\n"
        << "Модель:\n"
        << "  --model path.onnx         YOLOv8-seg / YOLO11-seg / YOLO26-seg в формате ONNX\n"
        << "  --infer-device cpu|cuda   провайдер ONNX Runtime\n"
        << "  --infer-threads N         intra-op потоков модели (0 — по умолчанию)\n"
        << "  --score F                 порог уверенности (по умолчанию 0.35)\n"
        << "  --nms F                   порог NMS (по умолчанию 0.45)\n"
        << "  --mask-threshold F        порог бинаризации маски (0.5)\n\n"
        << "Отображение:\n"
        << "  --layout side|stack|overlay   раскладка окон (side — по умолчанию)\n"
        << "  --no-masks --no-boxes --no-contours --no-hud\n"
        << "  --mirror                  отразить кадр по горизонтали\n"
        << "  --headless                без окон (для записи в файл)\n\n"
        << "Запись и отладка:\n"
        << "  --record out.mp4          записать мозаику в файл\n"
        << "  --snapshot out.png        сохранить один размеченный кадр\n"
        << "  --snapshot-every N        сохранять каждый N-й кадр в каталог snapshot_path\n"
        << "  --max-frames N            остановиться после N кадров\n"
        << "  --delay-ms N              искусственная задержка инференса (учебная)\n"
        << "  --log-level debug|info|warn|error\n"
        << "  --help                    эта справка\n\n"
        << "Горячие клавиши в окне: q/Esc — выход, s — сохранить кадр, m — маски,\n"
        << "b — боксы, h — HUD, 1/2/3 — раскладка.\n";
}

bool parse_args(int argc, char** argv, AppConfig& config, std::string& error) {
    // Первый проход: собираем пары ключ-значение. Это позволяет поддерживать
    // и "--key=value", и "--key value" без дублирования кода на каждый флаг.
    std::vector<std::pair<std::string, std::string>> args;
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token.empty()) {
            continue;
        }
        if (token[0] != '-') {
            error = "неожиданный аргумент: " + token;
            return false;
        }
        const std::size_t eq = token.find('=');
        if (eq != std::string::npos) {
            args.emplace_back(token.substr(0, eq), token.substr(eq + 1));
            continue;
        }
        // Ключи, у которых есть значение, перечислены явно (см. kValueKeys).
        // Неизвестный ключ без "=" считаем флагом и сообщаем о нём позже как о
        // неизвестной опции — это понятнее, чем «не указано значение».
        const std::string key = token.substr(0, eq);
        if (key_takes_value(key)) {
            if (i + 1 >= argc) {
                error = "для " + token + " не указано значение";
                return false;
            }
            args.emplace_back(token, argv[++i]);
        } else {
            args.emplace_back(token, "");
        }
    }

    for (const auto& [raw_key, value] : args) {
        const std::string& key = raw_key;
        if (key == "--help" || key == "-h") {
            config.show_help = true;
        } else if (key == "--device") {
            config.device = value;
        } else if (key == "--file") {
            config.video_file = value;
        } else if (key == "--synthetic") {
            if (value.empty()) {
                config.synthetic = true;
            } else if (bool enabled = false; parse_bool(value, enabled)) {
                config.synthetic = enabled;
            } else {
                error = "--synthetic: ожидалось true/false, получено " + value;
                return false;
            }
        } else if (key == "--width") {
            if (!parse_int(value, config.width) || config.width <= 0) {
                error = "некорректная ширина: " + value;
                return false;
            }
        } else if (key == "--height") {
            if (!parse_int(value, config.height) || config.height <= 0) {
                error = "некорректная высота: " + value;
                return false;
            }
        } else if (key == "--fps") {
            if (!parse_int(value, config.fps) || config.fps <= 0 || config.fps > 240) {
                error = "некорректный fps: " + value;
                return false;
            }
        } else if (key == "--no-loop") {
            config.loop_file = false;
        } else if (key == "--fast") {
            config.pace_file = false;
        } else if (key == "--no-mjpeg") {
            config.prefer_mjpeg = false;
        } else if (key == "--model") {
            config.model_path = value;
        } else if (key == "--infer-device") {
            if (value != "cpu" && value != "cuda") {
                error = "--infer-device: ожидалось cpu или cuda, получено " + value;
                return false;
            }
            config.device_for_infer = value;
        } else if (key == "--infer-threads") {
            if (!parse_int(value, config.infer_threads) || config.infer_threads < 0) {
                error = "некорректное число потоков: " + value;
                return false;
            }
        } else if (key == "--score") {
            if (!parse_float(value, config.score_threshold) || config.score_threshold <= 0.0f ||
                config.score_threshold > 1.0f) {
                error = "--score должно быть в (0, 1]: " + value;
                return false;
            }
        } else if (key == "--nms") {
            if (!parse_float(value, config.nms_threshold) || config.nms_threshold <= 0.0f ||
                config.nms_threshold > 1.0f) {
                error = "--nms должно быть в (0, 1]: " + value;
                return false;
            }
        } else if (key == "--mask-threshold") {
            if (!parse_float(value, config.mask_threshold) || config.mask_threshold <= 0.0f ||
                config.mask_threshold >= 1.0f) {
                error = "--mask-threshold должно быть в (0, 1): " + value;
                return false;
            }
        } else if (key == "--layout") {
            if (value == "side") {
                config.layout = Layout::SideBySide;
            } else if (value == "stack") {
                config.layout = Layout::Stacked;
            } else if (value == "overlay") {
                config.layout = Layout::OverlayOnly;
            } else {
                error = "--layout: ожидалось side|stack|overlay, получено " + value;
                return false;
            }
        } else if (key == "--no-masks") {
            config.show_masks = false;
        } else if (key == "--no-boxes") {
            config.show_boxes = false;
        } else if (key == "--no-contours") {
            config.show_contours = false;
        } else if (key == "--no-hud") {
            config.show_hud = false;
        } else if (key == "--mirror") {
            config.mirror = true;
        } else if (key == "--headless") {
            config.headless = true;
        } else if (key == "--record") {
            config.record_path = value;
        } else if (key == "--snapshot") {
            config.snapshot_path = value;
        } else if (key == "--snapshot-every") {
            if (!parse_int(value, config.snapshots_every) || config.snapshots_every < 0) {
                error = "некорректный --snapshot-every: " + value;
                return false;
            }
        } else if (key == "--max-frames") {
            if (!parse_int(value, config.max_frames) || config.max_frames < 0) {
                error = "некорректный --max-frames: " + value;
                return false;
            }
        } else if (key == "--delay-ms") {
            if (!parse_int(value, config.artificial_delay_ms) || config.artificial_delay_ms < 0) {
                error = "некорректный --delay-ms: " + value;
                return false;
            }
        } else if (key == "--log-level") {
            if (!parse_log_level(value, config.log_level)) {
                error = "--log-level: ожидалось debug|info|warn|error, получено " + value;
                return false;
            }
        } else {
            error = "неизвестная опция: " + key + " (см. --help)";
            return false;
        }
    }

    return true;
}

std::string describe_config(const AppConfig& config) {
    std::ostringstream out;
    out << "источник: "
        << (config.synthetic ? "synthetic"
                             : (config.video_file.empty() ? config.device : config.video_file))
        << " " << config.width << 'x' << config.height << '@' << config.fps
        << " | модель: " << (config.model_path.empty() ? "<нет>" : config.model_path)
        << " | раскладка: "
        << (config.layout == Layout::SideBySide
                ? "side"
                : (config.layout == Layout::Stacked ? "stack" : "overlay"))
        << " | порог: " << config.score_threshold;
    return out.str();
}

}  // namespace vt
