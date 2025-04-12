cmake_minimum_required(VERSION 3.10)
project(MotionEstimation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(motion_estimation src/main.cpp)
target_link_libraries(motion_estimation ${OpenCV_LIBS}) 