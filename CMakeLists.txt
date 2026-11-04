cmake_minimum_required(VERSION 3.16)
project(videotest LANGUAGES CXX)

# --- стандарт языка ----------------------------------------------------------
# C++20 нужен из-за std::jthread (если включён), std::span и designated
# initializers. Проект собирается и как C++17 — это оставлено для старых GCC.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Тип сборки" FORCE)
endif()

option(VIDEOTEST_WITH_ONNX "Собирать с ONNX Runtime (инференс)" ON)
option(VIDEOTEST_WARNINGS_AS_ERRORS "Считать предупреждения ошибками" OFF)

# --- поиск зависимостей ------------------------------------------------------
# OpenCV: core/imgproc/highgui/videoio/dnn нужны обязательно.
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)

message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")

# --- исходники ---------------------------------------------------------------
set(VIDEOTEST_SOURCES
    src/app/main.cpp
    src/app/cli.cpp
    src/core/logger.cpp
    src/core/signal_flag.cpp
    src/video/v4l2_source.cpp
    src/video/file_source.cpp
    src/pipeline/pipeline.cpp
    src/render/renderer.cpp
)

if(VIDEOTEST_WITH_ONNX)
    list(APPEND VIDEOTEST_SOURCES src/infer/onnx_segmenter.cpp)
endif()

add_executable(videotest ${VIDEOTEST_SOURCES})
target_include_directories(videotest PRIVATE src)
target_link_libraries(videotest PRIVATE ${OpenCV_LIBS})
target_compile_features(videotest PRIVATE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(videotest PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
    if(VIDEOTEST_WARNINGS_AS_ERRORS)
        target_compile_options(videotest PRIVATE -Werror)
    endif()
endif()

# --- ONNX Runtime ------------------------------------------------------------
# Вариант 1 (по умолчанию): SDK лежит в third_party/onnxruntime или задан
#   cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ...
# Вариант 2: системная установка, видимая через find_package(onnxruntime).
if(VIDEOTEST_WITH_ONNX)
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
            CACHE PATH "Корень распакованного ONNX Runtime SDK")
    endif()

    if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
        target_include_directories(videotest PRIVATE "${ONNXRUNTIME_ROOT}/include")
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
            NO_DEFAULT_PATH)
        if(NOT ONNXRUNTIME_LIBRARY)
            message(FATAL_ERROR
                "В ${ONNXRUNTIME_ROOT}/lib не найден libonnxruntime.so. "
                "Скачайте официальный релиз onnxruntime-linux-x64-*.tgz и распакуйте в "
                "third_party/onnxruntime, либо соберите с -DVIDEOTEST_WITH_ONNX=OFF")
        endif()
        target_link_libraries(videotest PRIVATE "${ONNXRUNTIME_LIBRARY}")
        # RPATH, чтобы бинарник запускался без LD_LIBRARY_PATH.
        set_target_properties(videotest PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${ONNXRUNTIME_ROOT}/lib64")
    else()
        find_package(onnxruntime QUIET)
        if(onnxruntime_FOUND)
            target_link_libraries(videotest PRIVATE onnxruntime::onnxruntime)
        else()
            message(FATAL_ERROR
                "ONNX Runtime не найден. Варианты:\n"
                "  1) распакуйте SDK в third_party/onnxruntime (см. scripts/fetch_onnxruntime.sh)\n"
                "  2) передайте -DONNXRUNTIME_ROOT=/путь/к/sdk\n"
                "  3) соберите без инференса: -DVIDEOTEST_WITH_ONNX=OFF")
        endif()
    endif()
    target_compile_definitions(videotest PRIVATE VIDEOTEST_WITH_ONNX=1)
endif()

# --- предупреждение про платформу --------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "V4L2 доступен только на Linux; на этой платформе используйте "
                    "--synthetic или --file")
endif()

# --- юнит-тесты (без внешних фреймворков) ------------------------------------
# queue_tests зависят только от заголовков core/, поэтому собираются даже без
# OpenCV и ONNX Runtime — это позволяет проверять многопоточные структуры
# на любой машине.
enable_testing()

add_executable(queue_tests tests/queue_test.cpp)
target_include_directories(queue_tests PRIVATE src)
target_compile_features(queue_tests PRIVATE cxx_std_20)
add_test(NAME queue_tests COMMAND queue_tests)

# nms_tests не требуют ни OpenCV-видео, ни ONNX Runtime: только заголовки
# infer/nms.hpp и infer/detection.hpp (+ cv::Rect2f из OpenCV core).
add_executable(nms_tests tests/postprocess_test.cpp)
target_include_directories(nms_tests PRIVATE src)
target_link_libraries(nms_tests PRIVATE ${OpenCV_LIBS})
target_compile_features(nms_tests PRIVATE cxx_std_20)
add_test(NAME nms_tests COMMAND nms_tests)

if(VIDEOTEST_WITH_ONNX)
    add_executable(letterbox_tests tests/letterbox_test.cpp src/infer/onnx_segmenter.cpp
        src/core/logger.cpp)
    target_include_directories(letterbox_tests PRIVATE src "${ONNXRUNTIME_ROOT}/include")
    target_compile_definitions(letterbox_tests PRIVATE VIDEOTEST_WITH_ONNX=1)
    target_link_libraries(letterbox_tests PRIVATE
        ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})
    set_target_properties(letterbox_tests PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${ONNXRUNTIME_ROOT}/lib64")
    target_compile_features(letterbox_tests PRIVATE cxx_std_20)
    add_test(NAME letterbox_tests COMMAND letterbox_tests)
endif()
