#include "track/tracker.hpp"

#include <algorithm>
#include <numeric>

namespace vt {
namespace {

float iou(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int w = std::max(0, x2 - x1);
    const int h = std::max(0, y2 - y1);
    const float intersection = static_cast<float>(w * h);
    const float uni = static_cast<float>(a.width * a.height + b.width * b.height) - intersection;
    return uni > 0.0f ? intersection / uni : 0.0f;
}

// box(x,y,w,h) -> состояние(cx,cy,w,h)
cv::Mat box_to_state(const cv::Rect& box) {
    cv::Mat state(8, 1, CV_32F, cv::Scalar(0.0f));
    state.at<float>(0) = static_cast<float>(box.x) + static_cast<float>(box.width) * 0.5f;
    state.at<float>(1) = static_cast<float>(box.y) + static_cast<float>(box.height) * 0.5f;
    state.at<float>(2) = static_cast<float>(box.width);
    state.at<float>(3) = static_cast<float>(box.height);
    return state;
}

}  // namespace

Tracker::Tracker(TrackerConfig config) : config_(config) {}

cv::Rect Tracker::to_box(const cv::Mat& state) {
    const float cx = state.at<float>(0);
    const float cy = state.at<float>(1);
    const float w = std::max(1.0f, state.at<float>(2));
    const float h = std::max(1.0f, state.at<float>(3));
    return cv::Rect(cvRound(cx - w * 0.5f), cvRound(cy - h * 0.5f), cvRound(w), cvRound(h));
}

void Tracker::predict(Track& track) const {
    // Важно: арифметику predict() делает сам Калман, а НЕ умножение матриц
    // руками. Без вызова kalman.predict() матрица ковариации ошибки не растёт
    // (errorCovPre остаётся единичной), gain выходит крошечным, и трек почти не
    // двигается — объект «убегает» от своего трека, IoU падает ниже порога,
    // и id пересоздаётся.
    //
    // cv::KalmanFilter::predict() использует СВОИ копии матриц (transitionMatrix
    // и другие), которые он берёт из публичных полей на момент вызова. Поэтому
    // сначала синхронизируем поля, а результат забираем из statePre.
    track.kalman.transitionMatrix = track.transition;
    track.kalman.measurementMatrix = track.measurement;
    track.kalman.predict();

    track.state = track.kalman.statePre.clone();
    track.box = to_box(track.state);
}

void Tracker::correct(Track& track, const cv::Rect& box) const {
    // Детекция в измерении Калмана: только позиция и размер (4 значения).
    cv::Mat measurement = cv::Mat(4, 1, CV_32F, cv::Scalar(0.0f));
    measurement.at<float>(0) = static_cast<float>(box.x) + static_cast<float>(box.width) * 0.5f;
    measurement.at<float>(1) = static_cast<float>(box.y) + static_cast<float>(box.height) * 0.5f;
    measurement.at<float>(2) = static_cast<float>(box.width);
    measurement.at<float>(3) = static_cast<float>(box.height);

    track.kalman.transitionMatrix = track.transition;
    track.kalman.measurementMatrix = track.measurement;
    track.state = track.kalman.correct(measurement).clone();
    track.box = to_box(track.state);
    track.missed = 0;
}

void Tracker::update(std::vector<Instance>& instances) {
    // --- 1. предсказание: где каждый трек должен быть на этом кадре ---
    for (Track& track : tracks_) {
        predict(track);
    }

    // --- 2. сопоставление: жадно по убыванию IoU ---
    // Полная матрица «детекции x треки» и её сортировка по убыванию IoU.
    struct Pair {
        float iou;
        std::size_t det;
        std::size_t track;
    };
    std::vector<Pair> pairs;
    for (std::size_t d = 0; d < instances.size(); ++d) {
        for (std::size_t t = 0; t < tracks_.size(); ++t) {
            const float value = iou(instances[d].box, tracks_[t].box);
            if (value >= config_.iou_threshold) {
                pairs.push_back({value, d, t});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.iou > b.iou; });

    std::vector<bool> det_used(instances.size(), false);
    std::vector<bool> track_used(tracks_.size(), false);
    for (const Pair& pair : pairs) {
        if (det_used[pair.det] || track_used[pair.track]) {
            continue;
        }
        det_used[pair.det] = true;
        track_used[pair.track] = true;
        correct(tracks_[pair.track], instances[pair.det].box);
        instances[pair.det].track_id = tracks_[pair.track].id;
    }

    // --- 3. новые треки для несопоставленных детекций ---
    for (std::size_t d = 0; d < instances.size(); ++d) {
        if (det_used[d]) {
            continue;
        }
        Track track;
        track.id = next_id_++;
        track.box = instances[d].box;
        track.kalman.init(8, 4, 0, CV_32F);

        // Матрица перехода «равномерное движение»: позиция += скорость.
        track.transition = cv::Mat::eye(8, 8, CV_32F);
        for (int i = 0; i < 4; ++i) {
            track.transition.at<float>(i, i + 4) = 1.0f;
        }
        // Измеряем только позицию/размер (4 из 8 компонент состояния).
        track.measurement = cv::Mat::zeros(4, 8, CV_32F);
        for (int i = 0; i < 4; ++i) {
            track.measurement.at<float>(i, i) = 1.0f;
        }
        track.kalman.transitionMatrix = track.transition;
        track.kalman.measurementMatrix = track.measurement;
        // Небольшой шум процесса и заметный шум измерений: детекции шумят
        // сильнее, чем физика движения.
        cv::setIdentity(track.kalman.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(track.kalman.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(track.kalman.errorCovPost, cv::Scalar::all(1.0));
        track.state = box_to_state(instances[d].box);
        track.kalman.statePost = track.state.clone();
        track.kalman.statePre = track.state.clone();

        instances[d].track_id = track.id;
        tracks_.push_back(std::move(track));
    }

    // --- 4. треки без детекции стареют и удаляются ---
    // track_used описывает треки, существовавшие ДО этого кадра: новые треки
    // добавлены шагом выше и «не виделись» быть не могут — иначе индексация
    // выйдет за границы track_used.
    for (std::size_t t = 0; t < track_used.size(); ++t) {
        if (!track_used[t]) {
            ++tracks_[t].missed;
        }
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const Track& track) {
                                     return track.missed > config_.max_age;
                                 }),
                  tracks_.end());
}

}  // namespace vt
