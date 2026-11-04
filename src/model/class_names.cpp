#include "model/class_names.hpp"

#include <fstream>
#include <iterator>

namespace vt {
namespace {

// Порядок индексов важен: это порядок, в котором YOLO-модели обучались на COCO.
const char* const kCoco[] = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",
    "bus",           "train",        "truck",         "boat",          "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",         "bird",
    "cat",           "dog",          "horse",         "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",       "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",      "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",
    "baseball glove","skateboard",   "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",         "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",         "donut",
    "cake",          "chair",        "couch",         "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",     "oven",
    "toaster",       "sink",         "refrigerator",  "book",          "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush",
};

}  // namespace

const std::vector<std::string>& coco_names() {
    static const std::vector<std::string> names(std::begin(kCoco), std::end(kCoco));
    return names;
}

bool load_class_names(const std::string& path, std::vector<std::string>& names,
                      std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "не удалось открыть файл имён классов: " + path;
        return false;
    }
    names.clear();
    std::string line;
    while (std::getline(file, line)) {
        // Убираем '\r' из файлов, сохранённых в Windows.
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        names.push_back(line);
    }
    if (names.empty()) {
        error = "в файле имён нет ни одной строки: " + path;
        return false;
    }
    return true;
}

std::string class_label(const std::vector<std::string>& names, int class_id) {
    if (class_id >= 0 && class_id < static_cast<int>(names.size())) {
        return names[static_cast<std::size_t>(class_id)];
    }
    return "c" + std::to_string(class_id);
}

}  // namespace vt
