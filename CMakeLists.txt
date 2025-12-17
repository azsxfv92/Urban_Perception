cmake_minimum_required(VERSION 3.10)
project(Urban_Perception)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED) 

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

add_executable(Urban_Perception source/main.cpp source/Yolov5Detector.cpp)

target_link_libraries(Urban_Perception
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    cudart
)