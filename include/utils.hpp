#pragma once
#include "spdlog/spdlog.h"
#include <cstdint>

inline unsigned int random_u32(uint64_t *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

inline float random_f32(uint64_t *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

#define WARN_IF(cond, ...)                                                     \
  if ((cond))                                                                  \
  SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__)

#define CHECK_IF(cond, ...)                                                    \
  if ((cond))                                                                  \
  SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__)

struct scoped_debug {
  const std::string msg;
  ~scoped_debug() { spdlog::debug(msg); }
};
