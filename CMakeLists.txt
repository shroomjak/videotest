cmake_minimum_required(VERSION 3.16)
project(videotest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Тип сборки" FORCE)
endif()

# --- зависимости -------------------------------------------------------------
# Нужны core/imgproc (cv::Mat, конвертации), highgui (окно), imgcodecs (JPEG),
# dnn (blobFromImage для входа сети) и video (cv::KalmanFilter для трекера).
# videoio не нужен: видео берём напрямую из V4L2.

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn video)
find_package(Threads REQUIRED)

# ONNX Runtime ставится как обычный SDK: include/ + lib/ (см. scripts/fetch_onnxruntime.sh)
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
        CACHE PATH "Корень SDK ONNX Runtime")
endif()
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "Не найден libonnxruntime.so в ${ONNXRUNTIME_ROOT}/lib.\n"
        "Запустите ./scripts/fetch_onnxruntime.sh или передайте -DONNXRUNTIME_ROOT=/путь/к/sdk")
endif()

add_library(videotest_core STATIC
    src/logger.cpp
    src/stop_flag.cpp
    src/streams/v4l2_source.cpp
    src/streams/synthetic_source.cpp
    src/model/segmenter.cpp
    src/track/tracker.cpp
    src/pipeline.cpp
    src/render/renderer.cpp
)
target_include_directories(videotest_core PUBLIC src "${ONNXRUNTIME_ROOT}/include")
target_link_libraries(videotest_core PUBLIC
    ${OpenCV_LIBS} Threads::Threads ${ONNXRUNTIME_LIBRARY})
set_target_properties(videotest_core PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${ONNXRUNTIME_ROOT}/lib64")

add_executable(videotest src/main.cpp)
target_link_libraries(videotest PRIVATE videotest_core)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -Wno-shadow для ONNX Runtime: его заголовки сами имеют shadow-предупреждения
    target_compile_options(videotest_core PRIVATE -Wall -Wextra)
endif()

# --- тесты -------------------------------------------------------------------
enable_testing()
add_executable(tracker_tests tests/tracker_test.cpp)
target_link_libraries(tracker_tests PRIVATE videotest_core)
add_test(NAME tracker_tests COMMAND tracker_tests)

add_executable(queue_tests tests/queue_test.cpp)
target_link_libraries(queue_tests PRIVATE videotest_core)
add_test(NAME queue_tests COMMAND queue_tests)
