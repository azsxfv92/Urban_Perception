#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <iostream>
#include <memory>

struct TensorRTDeleter {
    template <typename T>
    void operator()(T* obj) const{
        if (obj) delete obj;
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, TensorRTDeleter>;


struct Detection {
        int classID;
        float confi;
        cv::Rect box;       
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

class Yolov5Detector {
public:

    static const int INPUT_W = 640;
    static const int INPUT_H = 640;
    static const int NUM_CLASSES = 80;
    static const int OUTPUT_SIZE = 25200 * (5+80);

    Yolov5Detector();
    ~Yolov5Detector();

    bool loadEngine(const std::string& enginePath);
    std::vector<Detection> detector(cv::Mat& img);
    

    // bool loadEngine(const std::string& enginePath) 
    // {
    //     std::ifstream file(enginePath, std::ios::binary);
    //     if(!file.good()){
    //         std::cerr << "Can not find Engine file" << std::endl;
    //         return false;
    //     }
    //     file.seekg(0, file.end);
    //     size_t size = file.tellg();
    //     file.seekg(0, file.beg);

    //     std::vector<char> engineData(size);
    //     file.read(engineData.data(), size);
    //     file.close();

    //     runtime = nvinfer1::createInferRuntime(gLogger);
    //     engine = runtime->deserializeCudaEngine(engineData.data(), size);
    //     if(!engine) return false;

    //     context = engine->createExecutionContext();

    //     size_t inputByteSize = 3 * INPUT_W * INPUT_H * sizeof(float);
    //     size_t outputByteSize = OUTPUT_SIZE * sizeof(float);

    //     // cudaMalloc asign memory
    //     // &buffers[0] is an address to be asigned
    //     // inputByteSize is a size of memory to be asigned
    //     cudaMalloc(&buffer[0], inputByteSize);
    //     cudaMalloc(&buffer[1], outputByteSize);

    //     return true;
    // }

    // //perform inference function
    // std::vector<Detection> detect(cv::Mat& img){
    //     std::vector<Detection> result;

    //     std::vector<float> hostInputBuffer(3*INPUT_W*INPUT_H);
    //     preprocess(img, hostInputBuffer.data());

    //     // Copy data from CPU to GPU
    //     // Since kernels run on device, they cannot directly access host memory, so data must be copied to device memory
    //     // cudaMemcpy(dst, src, size, kind)
    //     // dst: destination memory address, src: source memory, size: bytes to copy, kind: HtD or DtH
    //     cudaMemcpy(buffer[0], hostInputBuffer.data(), 3*INPUT_W*INPUT_H*sizeof(float), cudaMemcpyHostToDevice);
        
    //     // check what the input and output name is
    //     const char* inputName = engine->getIOTensorName(0);
    //     const char* outputName = engine->getIOTensorName(1);
    //     // connect buffer and tensor name
    //     context->setTensorAddress(inputName, buffer[0]);
    //     context->setTensorAddress(outputName, buffer[1]);
    //     // perform inference
    //     context->enqueueV3(0);

    //     // copy data from GPU to CPU
    //     std::vector<float> hostOutputBuffer(OUTPUT_SIZE);
    //     cudaMemcpy(hostOutputBuffer.data(), buffer[1], OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //     postprocess(hostOutputBuffer, result, img.cols, img.rows);

    //     return result;
    // }


private:
    void preprocess(cv::Mat& img, float* hostDataBuffer);
    void postprocess(std::vector<float>& output, std::vector<Detection>& result, int orgwW, int orgH);
    float computeIoU(const cv::Rect& box1, const cv::Rect& box2);
    UniquePtr<nvinfer1::IRuntime> runtime = nullptr;
    UniquePtr<nvinfer1::ICudaEngine> engine = nullptr;
    UniquePtr<nvinfer1::IExecutionContext> context = nullptr;    
    nvinfer1::ILogger* gLogger = nullptr;

    void* buffer[2];
    // void preprocess(cv::Mat& img, float* hostDataBuffer) {
    //     cv::Mat resized;
    //     cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    //     cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    //     // Scan image pixels
    //     int index = 0;
    //     // TensorRT(PyTorch) order: Channel -> Height -> Width
    //     for (int c = 0; c < 3; c++) {
    //         for (int h = 0; h < INPUT_H; h++) {
    //             for (int w = 0; w < INPUT_W; w++) {
    //                 // OpenCV order: Height, Width, Channel
    //                 // Normalize to 0~1 range by dividing by 255.0
    //                 hostDataBuffer[index++] = resized.at<cv::Vec3b>(h, w)[c] / 255.0f;
    //             }
    //         }
    //     }
    // }
    
    // float computeIoU(const cv::Rect& box1, const cv::Rect& box2){
        // int x1 = std::max(box1.x, box2.x);
        // int y1 = std::max(box1.y, box2.y);
        // int x2 = std::min(box1.x+box1.width, box2.x+box2.width);
        // int y2 = std::min(box1.y+box1.height, box2.y+box2.height);

    //     if (x1 >= x2 || y1>= y2) return 0.0f;
        
    //     // calculate box area
    //     float intersection = (float)(x2-x1) *(y2-y1);
    //     float area1 = box1.width * box1.height;
    //     float area2 = box2.width * box2.height;
        
    //     // calculate uinon area
    //     float union_area = area1 + area2 - intersection;
    //     return intersection/union_area;
    // }


    // // Simple post-processing (coordinate restoration)
    // void postprocess(std::vector<float>& output, std::vector<Detection>& result, int orgW, int orgH) {
    //     int num_proposals = 25200;
    //     int num_info = 85; // 5(box+conf) + 80(classes)

    //     // Calculate width/height scale ratios (for restoring to original image size)
    //     float scaleX = (float)orgW / INPUT_W;
    //     float scaleY = (float)orgH / INPUT_H;
    //     std::set<int> allowed_classes = {0, 1, 2, 3, 5, 7, 9, 11};
    //     std::vector<Detection> proposals;
    //     for (int i = 0; i < num_proposals; i++){
    //         float* data = output.data() + (i*num_info);
    //         float obj_conf = data[4];
    //         if (obj_conf < 0.5f) continue;

    //         float max_class_conf = 0;
    //         int max_class_id = -1;
            
    //         for (int j=5; j < num_info; j++){
    //             if (max_class_conf < data[j]){
    //                 max_class_conf = data[j];
    //                 max_class_id = j;
    //             }
    //         }
    //         // except uncategozied calsses
    //         if (allowed_classes.find(max_class_id) == allowed_classes.end()){
    //             continue;
    //         }

    //         float final_score = obj_conf * max_class_conf;
    //         if(final_score > 0.5f) {
    //             Detection det;
    //             det.classID = max_class_id;
    //             det.confi = final_score;
                
    //             float cx = data[0];
    //             float cy = data[1];
    //             float w = data[2];
    //             float h = data[3];

    //             det.box.x = (int)((cx-w/2)*scaleX);
    //             det.box.y = (int)((cy-h/2)*scaleY);
    //             det.box.width = (int)(w*scaleX);
    //             det.box.height = (int)(h*scaleY);

    //             proposals.push_back(det);
    //         }
    //     }

    //     // NMS algorithm
    //     std::sort(proposals.begin(), proposals.end(), [](const Detection& a, const Detection& b){
    //         return a.confi> b.confi;
    //     });

    //     std::vector<int> indices;

    //     for(size_t i=0; i<proposals.size(); ++i){
    //         bool keep = true;

    //         for (int k : indices) {
    //             if(computeIoU(proposals[i].box, proposals[k].box) > 0.45f){
    //                 keep = false;
    //                 break;
    //             }
    //         }
    //         if (keep) indices.push_back(i);
    //     }
    //     for (int idx : indices){
    //         result.push_back(proposals[idx]);
    //     }
    // }
};
