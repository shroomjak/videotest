#pragma once

// Кооперативная остановка по Ctrl+C.
//
// Обработчик сигнала — особая среда: в нём нельзя брать мьютексы, логировать,
// выделять память. Поэтому он делает ровно одно — выставляет флаг, а вся
// тяжёлая работа по остановке делается в обычном коде, который флаг опрашивает.

namespace vt {

void install_signal_handlers();  // SIGINT/SIGTERM -> request_stop()
void request_stop() noexcept;
bool stop_requested() noexcept;

}  // namespace vt
