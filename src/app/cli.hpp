#pragma once

// Разбор аргументов командной строки без внешних библиотек.
//
// Своя реализация (~150 строк) здесь уместнее, чем зависимость: аргументов
// немного, а заодно видно, как устроен «взрослый» разбор (--key=value, --key value,
// валидация значений, --help). Ключи собраны в один enum — компилятор ловит
// опечатки в switch.

#include <iosfwd>
#include <string>

#include "core/logger.hpp"
#include "infer/onnx_segmenter.hpp"
#include "render/renderer.hpp"
#include "video/v4l2_source.hpp"

namespace vt {

struct AppConfig {
    // источник
    std::string device = "/dev/video0";
    std::string video_file;  // если задан — используется вместо камеры
    bool synthetic = false;  // синтетический источник (без железа)
    int width = 1280;
    int height = 720;
    int fps = 30;
    bool loop_file = true;
    bool pace_file = true;
    bool prefer_mjpeg = true;

    // модель
    std::string model_path;  // пусто — работаем без модели (только камера)
    std::string device_for_infer = "cpu";
    int infer_threads = 0;
    float score_threshold = 0.35f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.5f;

    // отображение и прочее
    Layout layout = Layout::SideBySide;
    bool show_masks = true;
    bool show_boxes = true;
    bool show_contours = true;
    bool show_hud = true;
    bool mirror = false;  // горизонтальное отражение (привычнее для веб-камеры)
    bool headless = false;
    std::string record_path;  // записать итоговую мозаику в mp4
    std::string snapshot_path;
    int snapshots_every = 0;  // 0 — не сохранять
    int max_frames = 0;       // 0 — без ограничения
    int artificial_delay_ms = 0;
    LogLevel log_level = LogLevel::Info;
    bool show_help = false;
};

// Возвращает true, если разбор успешен. При ошибке текст пишется в error.
bool parse_args(int argc, char** argv, AppConfig& config, std::string& error);
void print_help(std::ostream& out, const char* program);
std::string describe_config(const AppConfig& config);

}  // namespace vt
