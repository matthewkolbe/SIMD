cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_COMPILER clang++)

project ("agnerexp")

add_compile_options(
  -std=c++20
  -Qunused-arguments
  -O3
  -march=native
)

set(SOURCES
    src/main.cpp
    ../../src/simd_unroller.hh
)


include(FetchContent)

FetchContent_Declare(vectorclass
  GIT_REPOSITORY https://github.com/vectorclass/version2.git
)

FetchContent_MakeAvailable(vectorclass)

include_directories(${vectorclass_SOURCE_DIR})


add_executable(agnerexp ${SOURCES})