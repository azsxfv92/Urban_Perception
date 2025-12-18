#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <iostream>
#include <memory>

// define TensorRT object delete rule
struct TensorRTDeleter {
    template <typename T>
    void operator()(T* obj) const{
        if (obj) delete obj;
    }
};

//define alias name for smart pointer
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

private:
    void preprocess(cv::Mat& img, float* hostDataBuffer);
    void postprocess(std::vector<float>& output, std::vector<Detection>& result, int orgwW, int orgH);
    float computeIoU(const cv::Rect& box1, const cv::Rect& box2);
    
    //use smart pointer
    /////  Member variables of a C++ class are destroyed in the reverse order of their declaration, 
    ///// though they are constructed in the order of their declaration.
    UniquePtr<nvinfer1::IRuntime> runtime = nullptr;
    UniquePtr<nvinfer1::ICudaEngine> engine = nullptr;
    UniquePtr<nvinfer1::IExecutionContext> context = nullptr; 
    
    nvinfer1::ILogger* gLogger = nullptr;
    void* buffer[2];
};