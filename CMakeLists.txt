cmake_minimum_required(VERSION 3.16)

project(vjo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM 4.0 REQUIRED)
find_package(OpenCV 4.0 REQUIRED)

add_library(vjo
  src/vjo.cpp
)
target_include_directories(vjo
  PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vjo
  gtsam
  ${OpenCV_LIBS}
)

add_executable(main
  src/main.cpp
)
target_link_libraries(main
  vjo
)
