#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <chrono>
#include <eve/module/algo.hpp>
#include <functional>
#include <nanobench.h>
#include <span>
#include <vector>

constexpr float dot(const std::span<float> &l, const std::span<float> &r) {
  float cum = 0.0;
  for (size_t i = 0; i < l.size(); ++i) {
    cum += l[i] * r[i];
  }
  return cum;
}

float dot_eve(const std::span<float> l, const std::span<float> r) {
  std::vector<float> out(l.size());
  eve::algo::transform_to(eve::views::zip(l, r), out,
                          [](auto xy) { return get<0>(xy) * get<1>(xy); });
  return eve::algo::reduce(out, 0.0);
}

float dot_eve2(const std::span<float> l, const std::span<float> r) {
  return eve::algo::transform_reduce(
      eve::views::zip(l, r), [](auto xy) { return get<0>(xy) * get<1>(xy); },
      0.0);
}

float dot_eve3(const std::span<float> l, const std::span<float> r) {
  return eve::algo::transform_reduce[eve::algo::fuse_operations](
      eve::views::zip(l, r),
      [](auto xy, auto sum) { return eve::fma(get<0>(xy), get<1>(xy), sum); },
      0.0f);
}

float dot_eve_aligned(const std::span<float> l, const std::span<float> r) {
  return eve::algo::transform_reduce[eve::algo::fuse_operations](
      eve::views::zip(l, r),
      [](auto xy, auto sum) { return eve::fma(get<0>(xy), get<1>(xy), sum); },
      0.0f);
}

void dot_bench(
    const std::string &name,
    std::function<float(const std::span<float>, const std::span<float>)>
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

TEST_CASE("dot", "[correctness]") {
  const int length = 16;
  std::vector<float> in1(length, 2.4);
  std::vector<float> in2(length, 4.2);
  float ref = dot(in1, in2);
  float ans = dot_eve(in1, in2);
  REQUIRE(ans == Catch::Approx(ref));
  ans = dot_eve2(in1, in2);
  REQUIRE(ans == Catch::Approx(ref));
  ans = dot_eve3(in1, in2);
  REQUIRE(ans == Catch::Approx(ref));
}

TEST_CASE("dot_bench", "[benchmark]") {
  dot_bench("dot", dot);
  dot_bench("dot_eve", dot_eve);
  dot_bench("dot_eve2", dot_eve2);
  dot_bench("dot_eve3", dot_eve3);

  ankerl::nanobench::Bench benchmark;
  benchmark.title("dot_eve_aligned")
      .performanceCounters(true)
      .minEpochTime(std::chrono::milliseconds(100));

  for (size_t length = 10; length <= 100'000; length *= 10) {
    alignas(eve::wide<float>::alignment()) float in1[length];
    alignas(eve::wide<float>::alignment()) float in2[length];
    for (size_t i = 0; i < length; ++i) {
      in1[i] = 2.4;
      in2[i] = 4.2;
    }
    benchmark.run(std::to_string(length), [&] {
      float res = dot_eve_aligned({in1, length}, {in2, length});
      ankerl::nanobench::doNotOptimizeAway(res);
      return res;
    });
  }
}
