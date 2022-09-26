cmake_minimum_required(VERSION 3.14...3.24)

project(WaveRNN)

include(FetchContent)

# CMake Requirements
SET(CMAKE_OBJECT_PATH_MAX 512)

find_package(Python3 COMPONENTS Interpreter Development)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.2.3
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
        cnpy
        GIT_REPOSITORY https://github.com/rogersce/cnpy
        GIT_TAG master
)
FetchContent_MakeAvailable(cnpy)
set(CNPY_INCLUDE_DIR ${cnpy_SOURCE_DIR})

FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen3)
set(EIGEN3_INCLUDE_DIR ${Eigen3_SOURCE_DIR})

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${CNPY_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -ffast-math -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ffast-math -march=native")

add_executable(vocoder vocoder.cpp)
add_library(wavernn wavernn.cpp net_impl.cpp)
target_link_libraries(vocoder wavernn Eigen3::Eigen cnpy)

pybind11_add_module(WaveRNNVocoder WaveRNNVocoder.cpp wavernn.cpp net_impl.cpp)

