cmake_minimum_required (VERSION 3.14...3.24)

project(WaveRNN)

include(FetchContent)

# CMake Requirements
find_package (Python3 COMPONENTS Interpreter Development)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.2.3
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -ffast-math -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ffast-math -march=native")

add_executable (vocoder vocoder.cpp)
add_library(wavernn wavernn.cpp net_impl.cpp)
target_link_libraries(vocoder wavernn Eigen3::Eigen)

pybind11_add_module(WaveRNNVocoder WaveRNNVocoder.cpp wavernn.cpp net_impl.cpp)

