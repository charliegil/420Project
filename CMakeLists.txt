cmake_minimum_required(VERSION 3.30)
project(420Project)

set(CMAKE_CXX_STANDARD 17)  # C++26 is too new; CMake may not support it yet

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(420Project main.cpp)
target_link_libraries(420Project ${OpenCV_LIBS})
