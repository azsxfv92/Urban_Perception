#include <iostream>
#include <opencv2/opencv.hpp>
#include "Yolov5Detector.h"
#include <chrono>
#include <thread>
#include <nvtx3/nvToolsExt.h>
#include "SafeQueue.h"

void InputTask(SafeQueue<cv::Mat>& q_in, cv::VideoCapture& cap){
    // naming thread
    nvtxNameOsThreadA(pthread_self(), "Input Thread");
    while(true){
        // start period
        nvtxRangePushA("Camera Load & Push");
        cv::Mat frame;
        cap >> frame;
        if(frame.empty()){
            // if the video is finished, pass the empty frame to InferenceTask
            q_in.push(cv::Mat());
            break;
        } 
        q_in.push(frame);
        //end period
        nvtxRangePop();
    }
}

void InferenceTask(Yolov5Detector& detector, SafeQueue<cv::Mat>& q_in, SafeQueue<cv::Mat>& q_out){
    nvtxNameOsThreadA(pthread_self(), "Inference Thread");
    int frame_count = 0;
    while(true){
        // pop the frame, which is added in InputTask 
        cv::Mat frame = q_in.pop();

        if(frame.empty()) {
            q_out.push(cv::Mat());
            break;
        }

        nvtxRangePushA("Perform inference");
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> result = detector.detector(frame);
        auto end = std::chrono::high_resolution_clock::now();
        nvtxRangePop();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
       
        for (const auto& det : result){
            //rectangle(Imaege, Rect, Scalar(b,g,r), thickness, linetype, shift)
            // Image : input images, Rect : range of rectangles, Scalar :color, thicknees: line thicknees 
            cv::rectangle(frame, det.box, cv::Scalar(0,0,255), 2);
            std::string label = std::to_string(det.classID) + ": " + 
                                std::to_string((int)(det.confi * 100)) + "%";
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        if (frame_count % 30 == 0) {
        std::cout << "Processing frame : " << frame_count
                    << "| Time : " << duration << "ms"
                    << "| FPS " << (duration > 0 ? 1000.0 / duration : 0.0) << std::endl;
        }
        frame_count++;

        //push frame to output q to share it to SaveTask
        q_out.push(frame);

    }

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

    SafeQueue<cv::Mat> input_q(5);
    SafeQueue<cv::Mat> output_q(5);

    ///// std::ref() is used to pass the parameter as reference(&)
    // creat thread
    std::thread input_thread(InputTask, std::ref(input_q), std::ref(cap));
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