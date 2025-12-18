#!/bin/bash

# [중요] 기존 환경변수($LD_LIBRARY_PATH)를 가져오지 않습니다!
# 오직 Jetson Tegra 드라이버와 CUDA 핵심 경로만 딱 설정합니다.
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64

echo ">>> Cleaned LD_LIBRARY_PATH applied."

# 출력 버퍼링 끄기 (로그가 씹히지 않게)
# stdbuf -o0 -e0 명령어로 실행
stdbuf -o0 -e0 ./build/Urban_Perception ../yolov5s.engine ./train/train_video.mp4
