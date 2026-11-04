#pragma once

// Имена классов модели.
//
// Два способа задать их явно (автоподбора нет — как и с форматами):
//   1. --classes names.txt  — файл, по одному имени в строке; строка i это класс i;
//   2. --classes 80         — число классов моделей COCO (YOLOv8/v11/YOLO26),
//                             стандартный список подставляется автоматически.
//
// Пустые строки в файле пропускаются, '#' в начале строки — комментарий.
// Если имён меньше, чем классов в модели, остальные подписываются как "cN".

#include <string>
#include <vector>

namespace vt {

// Стандартные 80 имён COCO — те, что видят YOLO*-модели из коробки.
const std::vector<std::string>& coco_names();

// Читает файл имён. При ошибке возвращает false и текст в error.
bool load_class_names(const std::string& path, std::vector<std::string>& names,
                      std::string& error);

// Подпись класса: name(class_id), если имя есть, иначе "c<class_id>".
std::string class_label(const std::vector<std::string>& names, int class_id);

}  // namespace vt
