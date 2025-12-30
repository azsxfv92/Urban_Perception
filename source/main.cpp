#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Yolov5Detector.h"
#include <chrono>
#include <thread>
#include <nvtx3/nvToolsExt.h>
#include "SafeQueue.h"
#include <vector>
#include <cuda_runtime.h>

void cuda_preprocess(void* src_ptr, void* dst_ptr, int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream);

// FrameData structure for zero-copy
struct FrameData {
    cv::Mat frame;
    void* device_ptr;  // GPU가 직접 접근할 수 있는 device pointer
};

void* alloc_pinned(size_t size){
    void* ptr = nullptr;
    cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
    return ptr;
}

void free_pinned(void* ptr){
    cudaFreeHost(ptr);
}

// [BEFORE] void InputTask(SafeQueue<cv::Mat>& q_in, cv::VideoCapture& cap, const int width, const int height){
void InputTask(SafeQueue<FrameData>& q_in, cv::VideoCapture& cap, const int width, const int height){
    // naming thread
    nvtxNameOsThreadA(pthread_self(), "Input Thread");
    
    // width * height * 3chennel(B,G,R) * 1 byte
    size_t sizeofFrame = width*height*3*sizeof(float);
    const int PINNED_MEM_SIZE = 15;

    std::vector<void*> real_mem;
    // add device pointer list for zero copy
    std::vector<void*> device_ptrs;
    std::vector<cv::Mat> pinned_mem_buf;

    for(int i =0; i<PINNED_MEM_SIZE;i++){
        void* alloc_mem = alloc_pinned(sizeofFrame);
        real_mem.push_back(alloc_mem);

        // declare device pointer
        void* dev_ptr;
        cudaHostGetDevicePointer(&dev_ptr, alloc_mem, 0);
        device_ptrs.push_back(dev_ptr);

        pinned_mem_buf.push_back(cv::Mat(height, width, CV_8UC3, alloc_mem));
    }

    int cusor = 0;
    while(true){
        // start period
        nvtxRangePushA("Camera Load & Push");

        // cv::Mat frame;
        // cap >> frame;
        cv::Mat& frame = pinned_mem_buf[cusor];
        if(!(cap.read(frame))){
            // if the video is finished, pass the empty frame to InferenceTask
            // [BEFORE] q_in.push(cv::Mat());
            FrameData empty_data = {cv::Mat(), nullptr};
            q_in.push(empty_data);
            break;
        }

        // [BEFORE] q_in.push(frame);
        // [AFTER] send the frame and device pointer 
        FrameData frame_data = {frame, device_ptrs[cusor]};
        q_in.push(frame_data);
        cusor = (cusor+1)%PINNED_MEM_SIZE;
        //end period
        nvtxRangePop();
    }
    for(void* addr : real_mem){
        free_pinned(addr);
    }
}

// [BEFORE] void InferenceTask(Yolov5Detector& detector, SafeQueue<cv::Mat>& q_in, SafeQueue<cv::Mat>& q_out){
void InferenceTask(Yolov5Detector& detector, SafeQueue<FrameData>& q_in, SafeQueue<cv::Mat>& q_out){
    nvtxNameOsThreadA(pthread_self(), "Inference Thread");
    nvtxNameOsThreadA(pthread_self(), "Post-process Thread");

    void* d_input[2] = {nullptr, nullptr};
    // allocate memmory for TensorRT
    cudaMalloc(&d_input[0], 640*640*3*sizeof(float));
    cudaMalloc(&d_input[1], 640*640*3*sizeof(float));

    size_t input_size = 640 * 640 *3 *sizeof(float);
    //store the original frame
    std::queue<cv::Mat> frame_orig;

    int frame_count = 0;

    while(true){
        // pop the frame, which is added in InputTask
        // [BEFORE] cv::Mat frame = q_in.pop();
        // [AFTER] FrameData를 받아서 frame과 device_ptr 분리
        FrameData frame_data = q_in.pop();

        // [BEFORE] if(frame.empty()) {
        if(frame_data.frame.empty()) {
            q_out.push(cv::Mat());
            break;
        }

        cv::Mat& frame = frame_data.frame;
        int curr_slot = frame_count % 2;

        nvtxRangePushA("Perform Pre-process & inference");
        // std::vector<Detection> result = detector.detector(frame);
        // perform pre-process using GPU
        // [BEFORE] cuda_preprocess(frame.data, d_input[curr_slot], frame.cols, frame.rows, 640,640,streams[curr_slot]);
        cuda_preprocess(frame_data.device_ptr, d_input[curr_slot], frame.cols, frame.rows, 640, 640, detector.getStream(curr_slot));
        // perform inference
        auto start = std::chrono::high_resolution_clock::now();
        detector.enqueue_gpu(curr_slot, d_input[curr_slot], frame.cols, frame.rows);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        nvtxRangePop();

        frame_orig.push(frame);

        if(frame_count > 0){
            nvtxRangePushA("Post-process Previous Frame");
            int prev_slot = (frame_count - 1) % 2;
            std::vector<Detection> results = detector.postprocess_cpu(prev_slot, frame.cols, frame.rows);
            // load original image
            cv::Mat prev_frame = frame_orig.front();
            frame_orig.pop();

            for (const auto& det : results){
                cv::rectangle(prev_frame, det.box, cv::Scalar(0,0,255), 2);
                std::string label = std::to_string(det.classID) + ": " + 
                                std::to_string((int)(det.confi * 100)) + "%";
                cv::putText(prev_frame, label, cv::Point(det.box.x, det.box.y - 5), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            if(q_out.size() < 5){
                q_out.push(prev_frame);
            }
            
            nvtxRangePop();

            if (frame_count % 30 == 0) {
                std::cout << "Processing frame : " << frame_count
                            << "| Time : " << duration << "ms"
                            << "| FPS " << (duration > 0 ? 1000.0 / duration : 0.0) << std::endl;
            }
        }
        frame_count++;    
    }

    cudaFree(d_input[0]);
    cudaFree(d_input[1]);
}


void SaveTask(SafeQueue<cv::Mat>& q_out, cv::VideoWriter& writer){
    nvtxNameOsThreadA(pthread_self(), "SaveTask Thread");
    while(true){
        cv::Mat frame = q_out.pop();

        if(frame.empty()){
            break;
        }

        nvtxRangePushA("Save Video");
        writer.write(frame);
        nvtxRangePop();
    }
}



int main(int argc, char** argv) {

    if (argc < 3)
    {
        std::cerr << "How to use :./Ur" << std::endl;
        return -1;
    }

    std::string engine_path = argv[1];
    std::string video_path = argv[2];
    std::cout << " >>> load Engine (" << engine_path << ")" << std::endl;
    Yolov5Detector detector; 
    
    if(!detector.loadEngine(engine_path)){
        std::cerr << " Fail engine load" << std::endl;
        return -1;
    }
    std::cout << ">>> success engine load" << std::endl;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << " Video error!" << std::endl;
        return -1;
    }

    // prepare for saving result file
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // get FPS iformation (default value is 30)
    double fps = cap.get(cv::CAP_PROP_FPS);
    if(fps <= 0) fps = 30.0;

    cv::VideoWriter writer("result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if(!writer.isOpened()){
        std::cerr << " Can not create result.mp4 file" << std::endl;
        return -1;
    }

    // [BEFORE] SafeQueue<cv::Mat> input_q(5);
    // [AFTER] use FramData to send the device pointer
    SafeQueue<FrameData> input_q(5);
    SafeQueue<cv::Mat> output_q(5);

    ///// std::ref() is used to pass the parameter as reference(&)
    // creat thread
    std::thread input_thread(InputTask, std::ref(input_q), std::ref(cap), width, height);
    std::thread infer_thread(InferenceTask, std::ref(detector), std::ref(input_q), std::ref(output_q));
    std::thread save_thread(SaveTask, std::ref(output_q), std::ref(writer));
    //start thread
    input_thread.join();
    infer_thread.join();
    save_thread.join();

    cap.release();
    cv::destroyAllWindows();
    std::cout << ">>> Complete: result.mp4" << std::endl;
    return 0;
}