// Тесты на структуры данных конвейера — без GoogleTest, чтобы не тянуть
// зависимость: минимальный самодельный assert-макрос достаточно нагляден и
// показывает, как проверять многопоточный код детерминированно.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "core/bounded_queue.hpp"
#include "core/latest_value.hpp"

namespace {

int g_failures = 0;

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #condition);    \
            ++g_failures;                                                         \
        }                                                                         \
    } while (false)

using namespace std::chrono_literals;
using vt::BoundedQueue;
using vt::LatestValue;
using vt::OverflowPolicy;

void test_fifo_order() {
    BoundedQueue<int> queue(4, OverflowPolicy::Block);
    for (int i = 0; i < 4; ++i) {
        CHECK(queue.push(i));
    }
    for (int i = 0; i < 4; ++i) {
        const auto value = queue.try_pop();
        CHECK(value.has_value() && *value == i);
    }
    CHECK(!queue.try_pop().has_value());
}

void test_drop_oldest() {
    BoundedQueue<int> queue(2, OverflowPolicy::DropOldest);
    queue.push(1);
    queue.push(2);
    queue.push(3);  // вытесняет 1
    CHECK(queue.dropped() == 1);
    CHECK(*queue.try_pop() == 2);
    CHECK(*queue.try_pop() == 3);
}

void test_drop_newest() {
    BoundedQueue<int> queue(1, OverflowPolicy::DropNewest);
    CHECK(queue.push(1));
    CHECK(!queue.push(2));
    CHECK(queue.dropped() == 1);
    CHECK(*queue.try_pop() == 1);
}

void test_block_until_space() {
    BoundedQueue<int> queue(1, OverflowPolicy::Block);
    CHECK(queue.push(1));
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        queue.push(2);  // должен уснуть до освобождения места
        pushed.store(true);
    });
    std::this_thread::sleep_for(50ms);
    CHECK(!pushed.load());
    CHECK(*queue.try_pop() == 1);
    producer.join();
    CHECK(pushed.load());
    CHECK(*queue.try_pop() == 2);
}

void test_close_wakes_consumer() {
    BoundedQueue<int> queue(2, OverflowPolicy::Block);
    std::atomic<bool> woke{false};
    std::thread consumer([&] {
        const auto value = queue.pop_wait_for(10s);  // проснётся по close()
        woke.store(!value.has_value());
    });
    std::this_thread::sleep_for(20ms);
    queue.close();
    consumer.join();
    CHECK(woke.load());
    CHECK(!queue.push(1));  // после close push запрещён
}

void test_pop_timeout() {
    BoundedQueue<int> queue(2, OverflowPolicy::Block);
    const auto start = std::chrono::steady_clock::now();
    const auto value = queue.pop_wait_for(30ms);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(!value.has_value());
    CHECK(elapsed >= 20ms);
}

void test_latest_value() {
    LatestValue<std::string> latest;
    CHECK(!latest.get().has_value());
    latest.set("a");
    latest.set("b");
    CHECK(latest.get().has_value() && *latest.get() == "b");
}

// Стресс-тест: производитель и потребитель на разных скоростях.
// Проверяем отсутствие потерь при Block и корректность счётчиков при DropOldest.
void test_stress_no_loss() {
    BoundedQueue<int> queue(8, OverflowPolicy::Block);
    constexpr int kTotal = 20000;

    std::thread producer([&] {
        for (int i = 0; i < kTotal; ++i) {
            queue.push(i);
        }
        queue.close();
    });

    int expected = 0;
    while (auto value = queue.pop_wait_for(5s)) {
        if (*value != expected) {
            CHECK(false);
            break;
        }
        ++expected;
    }
    producer.join();
    CHECK(expected == kTotal);
    CHECK(queue.dropped() == 0);
}

void test_stress_drop_oldest() {
    BoundedQueue<int> queue(4, OverflowPolicy::DropOldest);
    constexpr int kTotal = 50000;
    std::atomic<int> received{0};

    std::thread producer([&] {
        for (int i = 0; i < kTotal; ++i) {
            queue.push(i);
        }
        queue.close();
    });
    std::thread consumer([&] {
        while (queue.pop_wait_for(200ms)) {
            received.fetch_add(1);
        }
    });

    producer.join();
    consumer.join();
    // Всё, что не доставлено, должно быть учтено как вытесненное:
    // received + dropped == kTotal (плюс элементы, оставшиеся в очереди).
    const std::size_t left = queue.size();
    CHECK(static_cast<long>(received.load()) + static_cast<long>(queue.dropped()) +
              static_cast<long>(left) ==
          kTotal);
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests = {
        {"fifo_order", test_fifo_order},
        {"drop_oldest", test_drop_oldest},
        {"drop_newest", test_drop_newest},
        {"block_until_space", test_block_until_space},
        {"close_wakes_consumer", test_close_wakes_consumer},
        {"pop_timeout", test_pop_timeout},
        {"latest_value", test_latest_value},
        {"stress_no_loss", test_stress_no_loss},
        {"stress_drop_oldest", test_stress_drop_oldest},
    };

    for (const auto& [name, fn] : tests) {
        const int before = g_failures;
        fn();
        std::printf("%-24s %s\n", name, (g_failures == before) ? "OK" : "FAILED");
    }
    if (g_failures != 0) {
        std::printf("\nвсего ошибок: %d\n", g_failures);
        return 1;
    }
    std::printf("\nвсе тесты прошли\n");
    return 0;
}
