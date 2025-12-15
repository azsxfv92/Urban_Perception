# Urban Perception - YOLOv5 TensorRT Object Detection

Real-time object detection using YOLOv5 with TensorRT acceleration.

## Docker Setup (Recommended)

**Environment:** Same as yolo-benchmark (dustynv/l4t-pytorch:r36.4.0)
- CUDA 12.6
- cuDNN 9.4
- TensorRT (built-in)
- OpenCV 4.5.4

### Build Docker Image

```bash
docker build -t urban-perception .
```

### Run Container

```bash
# For Jetson devices (Orin Nano, Xavier, etc.)
docker run -d --runtime nvidia --gpus all \
  --name urban-perception \
  -v $(pwd):/workspace/Urban_perception \
  -v /path/to/your/engine:/workspace/engine \
  -v /path/to/your/video:/workspace/video \
  urban-perception

# Enter the container
docker exec -it urban-perception bash
```

### Run Detection Inside Container

```bash
# Already in /workspace/Urban_perception/build
./Urban_Perception /workspace/engine/yolov5s.engine /workspace/video/input.mp4

# Results will be saved in the current directory
ls result_*.jpg
```

## Manual Setup

### Prerequisites

- CUDA Toolkit
- TensorRT
- OpenCV
- CMake 3.10+

### Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./Urban_Perception <engine_file> <video_file>
```

Example:
```bash
./Urban_Perception yolov5s.engine input.mp4
```

## Notes

- You need a `.engine` file (TensorRT engine) to run the detector
- The program saves detection results as images every ~1 second
- Output images: `result_0.jpg`, `result_31.jpg`, etc.
- Runs for approximately 10 seconds by default

## Detected Classes

Person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
