# videotest — камера → ONNX instance segmentation → визуализация в реальном времени

Учебный проект на C++: захват видео с USB-камеры (`/dev/videoX` через V4L2), инференс ONNX-модели instance-сегментации (YOLOv8-seg / YOLO11-seg / YOLO26-seg) и показ двух потоков рядом — сырого кадра с камеры и кадра с масками объектов. Трекинга пока нет; архитектура оставляет для него место.

Цель — разобраться и со средствами C++ (RAII, `unique_ptr`/`shared_ptr`, `std::atomic`, `std::mutex` + `condition_variable`, `std::optional`, `std::chrono`, шаблоны, pimpl), и с самой задачей: V4L2, memory-mapped буферы, препроцессинг/постпроцессинг YOLO, конвейер реального времени.

* Подробное описание технической стороны — **[docs/architecture.md](docs/architecture.md)**
* Справочник по средствам языка — **[docs/cpp-notes.md](docs/cpp-notes.md)**

![пример работы](docs/example.png)

*(кадр получен на синтетическом источнике с моделью-заглушкой: боксы и маски — от заглушки, а не от настоящей YOLO)*

```
┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌────────────────────┐
│ /dev/vid │──▶│ V4L2 capture │──▶│ очередь   │──▶│ ONNX Runtime │──▶│ очередь результатов│
│ камера   │   │ poll+DQBUF   │   │ (2, drop) │   │ YOLO-seg     │   │ (2, drop)          │
└──────────┘   └──────────────┘   └───────────┘   └──────────────┘   └─────────┬──────────┘
                                                                               ▼
                                                       ┌──────────────────────────────────┐
                                                       │ UI: [ камера | сегментация ] +HUD│
                                                       └──────────────────────────────────┘
```

## Быстрый старт

### Зависимости

```bash
sudo apt install build-essential cmake libopencv-dev v4l-utils
```

ONNX Runtime — официальный SDK, в дистрибутивах его нет:

```bash
./scripts/fetch_onnxruntime.sh 1.20.1     # разложит в third_party/onnxruntime
```

### Сборка и проверка

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

# сквозная проверка без камеры и без настоящих весов:
./scripts/smoke_test.sh
```

Собрать без инференса (проверить камеру, конвейер и UI): `-DVIDEOTEST_WITH_ONNX=OFF`.

### Модель

```bash
pip install ultralytics
python scripts/export_yolo26_seg.py --model yolo26n-seg.pt --imgsz 640 --output models/yolo26n-seg.onnx
```

Можно взять готовый ONNX из Ultralytics (`yolo11n-seg.onnx`, `yolov8n-seg.onnx`) — формат выходов определяется автоматически:
«сырой» `[1, 4+nc+32, 8400]` (YOLOv8/11) или end-to-end `[1, 300, 6+32]` (YOLO26, NMS уже в графе).

### Запуск

```bash
# камера + модель, два потока рядом
./build/videotest --model models/yolo26n-seg.onnx

# только камера, без модели
./build/videotest --device /dev/video0 --width 1280 --height 720 --fps 30

# без камеры: синтетическая картинка (проверка конвейера и отрисовки)
./build/videotest --synthetic --model models/yolo26n-seg.onnx

# файл вместо камеры
./build/videotest --file data/test.mp4 --model models/yolo26n-seg.onnx

# запись результата и снимок
./build/videotest --model models/yolo26n-seg.onnx --record out.avi --snapshot frame.png

# посмотреть, что будет, если модель медленнее камеры
./build/videotest --model models/yolo26n-seg.onnx --delay-ms 100
```

Полный список опций: `./build/videotest --help`.

`--record` подбирает кодек под расширение (для `.avi` — MJPG, он есть в любой сборке OpenCV; для `.mp4` нужен OpenCV с FFmpeg/H.264). Если кодек недоступен, программа предупредит, а не упадёт.

### Горячие клавиши

| Клавиша | Действие |
|---|---|
| `q` / `Esc` | выход |
| `s` | сохранить текущий кадр в PNG |
| `m` | маски вкл/выкл |
| `b` | боксы вкл/выкл |
| `c` | контуры масок вкл/выкл |
| `h` | HUD вкл/выкл |
| `1` / `2` / `3` | раскладка: рядом / друг под другом / только разметка |

## Что показывает HUD

```
capture  30.0 fps | infer  18.4 fps | captured 512
source 1280x720 @30 MJPG | uptime 00:28
infer 24.31 ms | latency 41.7 ms | dropped 3/1
queue 1/2 | objects 87 | classes 80
```

* `capture fps` и `infer fps` **расходятся** при медленной модели — это нормально: камера идёт с постоянной скоростью, модель обрабатывает столько, сколько успевает.
* `latency` — от получения буфера камеры до готовой разметки. Должна быть стабильной (≈ время инференса + 1–2 кадра). Если растёт — где-то копится очередь.
* `dropped a/b` — сколько кадров вытеснено из очереди кадров и из очереди результатов. Растущий счётчик значит, что модель не успевает; это осознанный выбор: показываем самый свежий кадр, а не отстаём всё сильнее.

## Требования

| Компонент | Версия |
|---|---|
| Linux с V4L2 | любое современное ядро |
| CMake | ≥ 3.16 |
| Компилятор | GCC 9+ / Clang 10+ (C++20; собирается и как C++17) |
| OpenCV | ≥ 4.5 (`core imgproc imgcodecs highgui videoio dnn`) |
| ONNX Runtime | 1.16+ |

## Структура

```
src/core/      frame, bounded_queue, latest_value, logger, stats, signal_flag
src/video/     IVideoSource, V4l2Source (V4L2), FileSource / SyntheticSource
src/infer/     ISegmenter, OnnxSegmenter (letterbox, декод, NMS, маски), nms
src/pipeline/  потоки захвата и инференса, жизненный цикл
src/render/    маски, контуры, боксы, HUD, склейка потоков
src/app/       CLI и точка входа
tests/         очереди (в т.ч. стресс на потоках), NMS/IoU, геометрия letterbox
scripts/       fetch_onnxruntime.sh, export_yolo26_seg.py, make_stub_model.py, smoke_test.sh
```

## Поведение в нестандартных условиях

Проект рассчитан на учебное использование, поэтому деградирует предсказуемо, а не падает:

| Ситуация | Поведение |
|---|---|
| камера не открылась | сообщение с errno и подсказкой, код возврата 1 |
| камера отключилась в процессе | 30 ошибок подряд → мягкая остановка конвейера с логом |
| модель стабильно падает | 20 ошибок инференса подряд → остановка (без «тихого» вечного цикла) |
| OpenCV без HighGUI-backend (сервер без X11) | предупреждение, работа продолжается в режиме headless (запись/снимки) |
| нет кодека для `--record` | предупреждение и подсказка про `.avi`/MJPG |
| нет ONNX Runtime | сборка с `-DVIDEOTEST_WITH_ONNX=OFF`: камера и UI работают, `--model` даёт понятную ошибку |
| неизвестное число классов (end-to-end модель без метаданных) | объекты подписываются как `class_N` |

## Ограничения текущей версии

* нет автоматического переподключения камеры (после серии ошибок конвейер останавливается);
* одна модель, один экземпляр, без батчинга;
* отображение через `cv::imshow` (для 60+ fps нужен собственный рендер);
* число коэффициентов масок зашито как 32 (стандарт YOLO-seg).

## Лицензия

См. [LICENSE](LICENSE).
