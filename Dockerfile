# Use the same base image as yolo-benchmark
# This image includes: PyTorch, CUDA 12.6, cuDNN 9.4, TensorRT, OpenCV 4.5.4
FROM dustynv/l4t-pytorch:r36.4.0

# Set working directory (same as yolo-benchmark)
WORKDIR /workspace

# Install additional build tools if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create project directory
WORKDIR /workspace/Urban_perception

# Copy source files
COPY source/ ./source/
COPY CMakeLists.txt .

# Build the project
RUN mkdir -p build && cd build && \
    cmake .. && \
    make

# Set working directory to build folder for easy execution
WORKDIR /workspace/Urban_perception/build

# Keep container running
CMD ["sleep", "infinity"]
