#pragma once

// Простейший трекер: сопоставление детекций с существующими треками по IoU
// и предсказание положения фильтром Калмана (тот же принцип, что в SORT).
//
// Механика на один кадр:
//   1. predict():  каждый живой трек сдвигает своё состояние Калманом вперёд;
//   2. associate(): для каждой детекции ищем трек с максимальным IoU
//                   (жадно, по убыванию IoU, без венгерского алгоритма);
//   3. update():   найденные треки корректируются наблюдением, счётчик
//                  «сколько кадров не видели» сбрасывается;
//   4. новые детекции становятся новыми треками; треки, которых давно не
//      видели, удаляются.
//
// Состояние трека — 8 чисел: cx, cy, w, h, vx, vy, vw, vh (измерения + скорости).
// Матрица перехода F для модели «равномерное движение»:
//     [1 0 0 0 1 0 0 0]
//     [0 1 0 0 0 1 0 0]
//     [0 0 1 0 0 0 1 0]
//     [0 0 0 1 0 0 0 1]
//     [0 0 0 0 1 0 0 0] ...
// Проще говоря: позиция += скорость.

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>  // cv::KalmanFilter

#include "frame.hpp"

namespace vt {

struct TrackerConfig {
    float iou_threshold = 0.3f;  // минимальное перекрытие для сопоставления
    int max_age = 15;            // сколько кадров трек живёт без детекции
};

class Tracker {
public:
    explicit Tracker(TrackerConfig config);

    // Обновляет треки по детекциям текущего кадра и проставляет track_id в
    // каждом экземпляре. Порядок instances не меняется.
    void update(std::vector<Instance>& instances);

    std::size_t active_tracks() const { return tracks_.size(); }

private:
    struct Track {
        int id = 0;
        int missed = 0;      // кадров подряд без обновления
        cv::Rect box;        // последнее сглаженное положение
        cv::Mat state;       // 8x1 CV_32F: cx, cy, w, h, vx, vy, vw, vh
        cv::Mat transition;  // 8x8 CV_32F: модель движения
        cv::Mat measurement; // 4x8 CV_32F: из состояния берём только позицию
        cv::KalmanFilter kalman;
    };

    void predict(Track& track) const;
    void correct(Track& track, const cv::Rect& box) const;
    static cv::Rect to_box(const cv::Mat& state);

    TrackerConfig config_;
    std::vector<Track> tracks_;
    int next_id_ = 0;
};

}  // namespace vt
