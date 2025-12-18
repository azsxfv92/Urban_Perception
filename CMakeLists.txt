cmake_minimum_required(VERSION 3.10)
project(Urban_Perception)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

find_package(CUDAToolkit REQUIRED)

link_directories(/usr/lib/aarch64-linux-gnu)

include_directories(
    source
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu # TensorRT 헤더 경로
)

add_executable(Urban_Perception source/main.cpp source/Yolov5Detector.cpp)

target_link_libraries(Urban_Perception
    ${OpenCV_LIBS}
    CUDA::cudart          # CUDA Runtime (최신 방식)
    CUDA::cuda_driver     # CUDA Driver (이게 있어야 nsys 오류 해결됨!)
    nvinfer               # TensorRT Core
    nvinfer_plugin        # TensorRT Plugin
    dl                    # 동적 로딩 (필수)
)