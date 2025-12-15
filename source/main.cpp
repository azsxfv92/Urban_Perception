#include <iostream>
#include <opencv2/opencv.hpp>
#include "Yolov5Detector.h"


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
    
    cv::Mat frame;
    int frame_count = 0;

    while (true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cout << "\nDone!" << std::endl;
            break;
        }

        frame_count++;
        // float progress = (float)frame_count / total_frames * 100.0f;
        // std::cout << "\rProgress: " << (int)progress << "%" << std::flush;

        std::vector<Detection> result = detector.detect(frame);

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

        writer.write(frame);
        frame_count++;
        if (frame_count % 30 == 0) {
            std::cout << "Processing frame: " << frame_count << std::endl;
        }

        if (cv::waitKey(1) == 'q')
        {
            break;
        }
        
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << ">>> Complete: result.mp4" << std::endl;
    return 0;
}