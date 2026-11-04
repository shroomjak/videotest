// Тесты трекера: id сохраняется между кадрами, новый объект получает новый id,
// трек исчезает после max_age кадров без детекции.

#include <cstdio>
#include <vector>

#include "track/tracker.hpp"

namespace {
int g_failed = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failed;                                                   \
        }                                                                 \
    } while (0)

vt::Instance make(float x, float y, float w = 100.0f, float h = 100.0f) {
    vt::Instance instance;
    instance.box = cv::Rect(cvRound(x), cvRound(y), cvRound(w), cvRound(h));
    instance.score = 0.9f;
    instance.class_id = 0;
    return instance;
}
}  // namespace

int main() {
    using vt::Tracker;
    using vt::TrackerConfig;

    // 1. Один и тот же объект, немного сдвинутый: track_id не меняется.
    {
        Tracker tracker(TrackerConfig{});
        std::vector<vt::Instance> first{make(100, 100)};
        tracker.update(first);
        CHECK(first[0].track_id == 0);

        std::vector<vt::Instance> second{make(110, 105)};  // тот же объект
        tracker.update(second);
        CHECK(second[0].track_id == 0);
        CHECK(tracker.active_tracks() == 1);
    }

    // 2. Второй, далёкий объект получает новый id.
    {
        Tracker tracker(TrackerConfig{});
        std::vector<vt::Instance> frame{make(100, 100), make(600, 400)};
        tracker.update(frame);
        CHECK(frame[0].track_id == 0);
        CHECK(frame[1].track_id == 1);
        CHECK(tracker.active_tracks() == 2);
    }

    // 3. Объект пропал: после max_age кадров трек удаляется.
    {
        TrackerConfig config;
        config.max_age = 3;
        Tracker tracker(config);
        std::vector<vt::Instance> seen{make(200, 200)};
        tracker.update(seen);
        CHECK(tracker.active_tracks() == 1);

        for (int i = 0; i < 4; ++i) {
            std::vector<vt::Instance> nothing;
            tracker.update(nothing);
        }
        CHECK(tracker.active_tracks() == 0);

        // Если объект вернулся — это уже новый трек с новым id.
        std::vector<vt::Instance> again{make(200, 200)};
        tracker.update(again);
        CHECK(again[0].track_id == 1);
    }

    // 4. Пустая разметка (модели нет) не должна падать.
    {
        Tracker tracker(TrackerConfig{});
        std::vector<vt::Instance> nothing;
        tracker.update(nothing);
        CHECK(nothing.empty());
        CHECK(tracker.active_tracks() == 0);
    }

    if (g_failed != 0) {
        std::printf("ошибок: %d\n", g_failed);
        return 1;
    }
    std::printf("tracker_tests: ok\n");
    return 0;
}
