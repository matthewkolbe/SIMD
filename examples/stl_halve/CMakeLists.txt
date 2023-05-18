cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_COMPILER clang++)

project ("stlhalve")

add_compile_options(
  -std=c++2b
  -Qunused-arguments
  -O3
  -march=native
)

set(SOURCES
    src/main.cpp
    ../../src/simd_unroller.hh
)

add_executable(stlhalve ${SOURCES})