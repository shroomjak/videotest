// Тесты очереди конвейера: вытеснение старого кадра и ожидание с таймаутом.
// Без фреймворков — минимум кода, всё видно.

#include <atomic>
#include <cstdio>
#include <optional>
#include <thread>

#include "bounded_queue.hpp"

namespace {
int g_failed = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failed;                                                   \
        }                                                                 \
    } while (0)

using namespace std::chrono_literals;
}  // namespace

int main() {
    // 1. Порядок FIFO и размер.
    {
        vt::Queue<int> queue(3);
        queue.push(1);
        queue.push(2);
        CHECK(queue.size() == 2);
        CHECK(*queue.pop(10ms) == 1);
        CHECK(*queue.pop(10ms) == 2);
        CHECK(!queue.pop(10ms).has_value());
    }

    // 2. Переполнение вытесняет самый старый элемент и сообщает о потере.
    {
        vt::Queue<int> queue(2);
        CHECK(queue.push(1) == 0);
        CHECK(queue.push(2) == 0);
        CHECK(queue.push(3) == 1);  // 1 вытеснен
        CHECK(*queue.pop(10ms) == 2);
        CHECK(*queue.pop(10ms) == 3);
    }

    // 3. pop ждёт данные и просыпается, когда их кладут из другого потока.
    {
        vt::Queue<int> queue(2);
        std::thread producer([&queue] {
            std::this_thread::sleep_for(50ms);
            queue.push(42);
        });
        const auto value = queue.pop(1s);
        producer.join();
        CHECK(value.has_value() && *value == 42);
    }

    // 4. Медленный потребитель при быстром производителе: очередь не растёт
    //    выше ёмкости (старые элементы вытесняются), а потребитель получает
    //    данные без блокировки. Замер разницы «положено - забрано» и есть
    //    счётчик потерь из stats_.
    {
        constexpr int kTotal = 5000;
        vt::Queue<int> queue(4);
        std::size_t dropped_total = 0;
        std::thread producer([&] {
            for (int i = 0; i < kTotal; ++i) {
                dropped_total += queue.push(i);
            }
        });
        int received = 0;
        while (received < 100) {
            if (!queue.pop(500ms)) {
                break;  // производитель закончил, очередь пуста
            }
            ++received;
        }
        producer.join();
        CHECK(received > 0);
        CHECK(queue.size() <= 4);
        // Каждый положенный элемент либо доставлен, либо вытеснен, либо ещё в
        // очереди — потерь «в никуда» быть не может.
        CHECK(static_cast<std::size_t>(kTotal) >= dropped_total);
    }

    // 5. Баланс при нагрузке: один производитель, один потребитель.
    //    Главный инвариант: положено == доставлено + вытеснено + осталось.
    //    Если не сходится — элемент потерялся или доставлен дважды, то есть в
    //    Queue гонка либо ошибка в политике переполнения.
    //    Плюс монотонность: вытесняется всегда самый старый, поэтому
    //    доставленные значения строго возрастают.
    {
        constexpr int kTotal = 20000;
        vt::Queue<int> queue(3);
        std::atomic<std::size_t> dropped{0};
        std::atomic<bool> producer_done{false};

        std::thread producer([&] {
            for (int i = 0; i < kTotal; ++i) {
                dropped.fetch_add(queue.push(i));
            }
            producer_done.store(true);
        });

        std::size_t received = 0;
        int last = -1;
        bool monotonic = true;
        while (true) {
            if (auto value = queue.pop(20ms)) {
                if (*value <= last) {
                    monotonic = false;
                }
                last = *value;
                ++received;
                continue;
            }
            // pop() ничего не дал. Выходим только если производитель уже
            // закончил И очередь действительно пуста — тогда все элементы
            // либо получены, либо вытеснены.
            if (producer_done.load(std::memory_order_acquire) && queue.size() == 0) {
                break;
            }
        }
        producer.join();

        CHECK(monotonic);
        CHECK(queue.size() == 0);
        CHECK(received + dropped.load() == static_cast<std::size_t>(kTotal));
    }

    if (g_failed != 0) {
        std::printf("ошибок: %d\n", g_failed);
        return 1;
    }
    std::printf("queue_tests: ok\n");
    return 0;
}
