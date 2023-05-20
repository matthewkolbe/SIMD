cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_COMPILER clang++)

project ("minfinder")

add_compile_options(
  -std=c++20
  -Qunused-arguments
  -O3
  -march=native
)

set(SOURCES
    src/main.cpp
    ../../src/simd_unroller.hh
    ../../src/intrinsic_generator.hh
)

add_executable(minfinder ${SOURCES})