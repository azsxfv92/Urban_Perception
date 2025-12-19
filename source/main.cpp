#include <iostream>
#include <opencv2/opencv.hpp>
#include "Yolov5Detector.h"
#include <chrono>
#include <thread>
#include <nvtx3/nvToolsExt.h>

void InputTask(Yolov5Detector* detector, cv::VideoCapture& cap){
    // naming thread
    nvtxNameOsThreadA(pthread_self(), "Input Thread");
    while(true){
        // start period
        nvtxRangePushA("Camera Load & Push");
        cv::Mat frame;
        cap >> frame;
        if(frame.empty()){
            // if the video is finished, pass the empty frame to InferenceTask
            detector->push(cv::Mat());
            break;
        } 
        detector->push(frame);
        //end period
        nvtxRangePop();
    }
}

void InferenceTask(Yolov5Detector* detector, cv::VideoWriter& writer){
    nvtxNameOsThreadA(pthread_self(), "Inference Thread");
    int frame_count = 0;
    while(true){
        cv::Mat frame;
        frame = detector->pop();
        if(frame.empty()) break;
     
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> result = detector->detector(frame);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        for (const auto& det : result)
        {
            //rectangle(Imaege, Rect, Scalar(b,g,r), thickness, linetype, shift)
            // Image : input images, Rect : range of rectangles, Scalar :color, thicknees: line thicknees 
            cv::rectangle(frame, det.box, cv::Scalar(0,0,255), 2);
            std::string label = std::to_string(det.classID) + ": " + 
                                std::to_string((int)(det.confi * 100)) + "%";
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }        

        nvtxRangePushA("Video Save");
        writer.write(frame);
        nvtxRangePop();

        frame_count++;
        if (frame_count % 30 == 0) {
            std::cout << "Processing frame : " << frame_count
                      << "| Time : " << duration << "ms"
                      << "| FPS " << (duration > 0 ? 1000.0 / duration : 0.0) << std::endl;
        }
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
    

    ///// std::ref() is used to pass the parameter as reference(&)
    // creat thread
    std::thread input_thread(InputTask, &detector, std::ref(cap));
    std::thread infer_thread(InferenceTask, &detector, std::ref(writer));
    //start thread
    input_thread.join();
    infer_thread.join();

    cap.release();
    cv::destroyAllWindows();
    std::cout << ">>> Complete: result.mp4" << std::endl;
    return 0;
}