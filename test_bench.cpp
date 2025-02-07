#include <chrono>
#include <eve/module/algo.hpp>
#include <functional>
#include <iostream>
#include <nanobench.h>
#include <vector>

constexpr float dot(const std::vector<float> &l, const std::vector<float> &r) {
  float cum = 0.0;
  for (size_t i = 0; i < l.size(); ++i) {
    cum += l[i] * r[i];
  }
  return cum;
}

float dot_eve(const std::vector<float> &l, const std::vector<float> &r) {
  std::vector<float> out(l.size());
  eve::algo::transform_to(eve::views::zip(l, r), out,
                          [](auto xy) { return get<0>(xy) * get<1>(xy); });
  return eve::algo::reduce(out, 0.0);
}

float dot_eve2(const std::vector<float> &l, const std::vector<float> &r) {
  return eve::algo::transform_reduce(
      eve::views::zip(l, r), [](auto xy) { return get<0>(xy) * get<1>(xy); },
      0.0);
}

float dot_eve3(const std::vector<float> &l, const std::vector<float> &r) {
  return eve::algo::transform_reduce[eve::algo::fuse_operations](
      eve::views::zip(l, r),
      [](auto xy, auto sum) { return eve::fma(get<0>(xy), get<1>(xy), sum); },
      0.0f);
}

void bench(
    const std::string &name,
    std::function<float(const std::vector<float> &, const std::vector<float> &)>
        f) {
  ankerl::nanobench::Bench benchmark;
  benchmark.title(name).performanceCounters(true).minEpochTime(
      std::chrono::milliseconds(100));

  for (size_t length = 10; length <= 100'000; length *= 10) {
    std::vector<float> in1(length, 2.4);
    std::vector<float> in2(length, 4.2);
    benchmark.run(std::to_string(length), [&] {
      float res = f(in1, in2);
      ankerl::nanobench::doNotOptimizeAway(res);
      return res;
    });
  }
}

int main() {
  bench("dot", dot);
  bench("dot_eve", dot_eve);
  bench("dot_eve2", dot_eve2);
  bench("dot_eve3", dot_eve3);
}
