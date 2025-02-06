cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(dagger)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(SAFETENSORS_CPP_IMPLEMENTATION)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
include_directories("third_party")
include_directories("include")

# add dependencies
include(cmake/CPM.cmake)
cpmaddpackage(
  NAME
  eve
  URL
  https://github.com/jfalcou/eve/archive/refs/tags/v2023.02.15.zip
  VERSION
  2023.02.15
  OPTIONS
  "EVE_BUILD_TEST OFF"
  "EVE_BUILD_BENCHMARKS OFF"
  "EVE_BUILD_DOCUMENTATION OFF")

cpmaddpackage("gh:martinus/nanobench@4.3.11")
cpmaddpackage("gh:catchorg/Catch2@3.6.0")
cpmaddpackage("gh:gabime/spdlog@1.x")

project(dagger)
add_definitions("-march=native")

# add_executable(example example.cpp) target_link_libraries(example PUBLIC
# Threads::Threads OpenMP::OpenMP_CXX spdlog::spdlog) add_executable(gpt2
# gpt.cpp) target_link_libraries(gpt2 PUBLIC Threads::Threads OpenMP::OpenMP_CXX
# spdlog::spdlog)
add_executable(test_bench test_bench.cpp)
target_link_libraries(test_bench PUBLIC Threads::Threads nanobench
                                        OpenMP::OpenMP_CXX spdlog::spdlog)
add_executable(test_safetensor test_safetensor.cpp)
target_link_libraries(test_safetensor PUBLIC Threads::Threads
                                             OpenMP::OpenMP_CXX spdlog::spdlog)
# add_executable(kvcache kvcache.cpp) target_link_libraries(kvcache PUBLIC
# Threads::Threads OpenMP::OpenMP_CXX spdlog::spdlog) add_executable(linear_vx
# linear_vx.cpp) target_link_libraries(linear_vx PUBLIC Threads::Threads
# OpenMP::OpenMP_CXX cnpy) add_executable(hw hw.cpp) target_link_libraries(hw
# PUBLIC Threads::Threads OpenMP::OpenMP_CXX cnpy nanobench) add_executable(mult
# mult.cpp) target_link_libraries(mult PUBLIC Threads::Threads
# OpenMP::OpenMP_CXX nanobench) add_executable(test test.cpp)
# target_link_libraries( test PRIVATE Catch2::Catch2WithMain

# PUBLIC Threads::Threads OpenMP::OpenMP_CXX spdlog::spdlog)
