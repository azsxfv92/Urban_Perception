#include "Yolov5Detector.h"
#include <fstream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <set>

Yolov5Detector::Yolov5Detector(){
    gLogger = new Logger();
    buffer[0] = nullptr;
    buffer[1] = nullptr;
}

Yolov5Detector::~Yolov5Detector() {
    if (gLogger) delete gLogger;
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
}

bool Yolov5Detector::loadEngine(const std::string& enginePath){
    //read engine file as binary format
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()){
        std::cerr<<"Engine File Error!"<<std::endl;
        return false;
    }
    
    //get file size
    file.seekg(0, file.end); // move cursor to last byte
    size_t size = file.tellg(); // store the current cursor position
    file.seekg(0, file.beg); // restore the position

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime =nvinfer1::createInferRuntime(*gLogger);
    if(!runtime) return false;

    // deserializeCudaEngine convert an encrypted file into an excutable model
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine) return false;

    context = engine->createExecutionContext();
    if(!context) return false;

    size_t inputByteSize = 3 * INPUT_W * INPUT_H * sizeof(float);
    size_t outputByteSize = OUTPUT_SIZE * sizeof(float);
    // cudaMalloc asign memory and return the asigned address through the first parameter
    // &buffers[0] is an address to be asigned
    // inputByteSize is a size of memory to be asigned
    ///// GPU를 사용하기위해 GPU 메모리를 선점
    cudaMalloc(&buffer[0], inputByteSize);
    cudaMalloc(&buffer[1], outputByteSize);
    
    return true;
}


std::vector<Detection> Yolov5Detector::detector(cv::Mat& img){
    std::vector<Detection> result;
    std::vector<float> hostInputBuffer(3*INPUT_W*INPUT_H);
    preprocess(img, hostInputBuffer.data());
    cudaMemcpy(buffer[0], hostInputBuffer.data(), 3*INPUT_W*INPUT_H, cudaMemcpyHostToDevice);

    // check what the input and output name is
    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    // connect buffer and tensor name
    context->setTensorAddress(inputName, buffer[0]);
    context->setTensorAddress(outputName, buffer[1]);
    //inference
    // "0" means using default stream
    context->enqueueV3(0);

    std::vector<float> hostOutputBuffer(OUTPUT_SIZE);
    cudaMemcpy(hostOutputBuffer.data(),buffer[1], OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    postprocess(hostOutputBuffer, result, img.cols, img.rows);
    
    return result;
}

void Yolov5Detector::preprocess(cv::Mat& img, float* hostDataBuffer) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
          // Scan image pixels
        int index = 0;
    // TensorRT(PyTorch) order: Channel -> Height -> Width
    for (int c= 0; c<3; c++){
        for (int h = 0; h<INPUT_H; h++){
            for (int w=0; w<INPUT_W; w++){
                hostDataBuffer[index++] = resized.at<cv::Vec3b>(h,w)[c] / 255.0f;
            }
        }
    }    
}

float Yolov5Detector::computeIoU(const cv::Rect& box1, const cv::Rect& box2){
    int x1 = std::max(box1.x, box2.x);
    // 이미지에서는 y값이 큰것이 밑에 있는 좌표다
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x+box1.width, box2.x+box2.width);
    int y2 = std::min(box1.y+box1.height, box2.y+box2.height);

    if (x1>=x2 || y1>=y2) return 0.0f;
    float intersection = (float)(x2-x1)*(y2-y1);
    float area1 = box1.width * box2.height;
    float area2 = box2.width * box2.height;

    float union_area = area1 + area2 - intersection;

    return intersection/union_area;
}

void Yolov5Detector::postprocess(std::vector<float>& output, std::vector<Detection>& result, int orgW, int orgH){
    int num_proposals = 25200;
    int num_info = 85; // 5(box+conf) + 80(classes)
    
    // Calculate width/height scale ratios (for restoring to original image size)
    float scaleX = (float)orgW / INPUT_W;
    float scaleY = (float)orgH / INPUT_H;
    std::set<int> allowed_classes = {0, 1, 2, 3, 5, 7, 9, 11};

    std::vector<Detection> proposals;
    for (int i = 0; i < num_proposals; i++){
        float* data = output.data() + (i*num_info);
        float obj_conf = data[4];
        if (obj_conf < 0.3f) continue;

        float max_class_conf = 0;
        int max_class_id = -1;
        
        for (int j=5; j < num_info; j++){
            if (max_class_conf < data[j]){
                max_class_conf = data[j];
                max_class_id = j-5;
            }
        }
        // except uncategozied calsses
        if (allowed_classes.find(max_class_id) == allowed_classes.end()){
            continue;
        }

        float final_score = obj_conf * max_class_conf;
        if(final_score > 0.3f) {
            Detection det;
            det.classID = max_class_id;
            det.confi = final_score;
            
            
            if (det.classID==9){
                std::cout << "Detected class ID: " << max_class_id << " with score: " << final_score << std::endl;
            }
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            det.box.x = (int)((cx-w/2)*scaleX);
            det.box.y = (int)((cy-h/2)*scaleY);
            det.box.width = (int)(w*scaleX);
            det.box.height = (int)(h*scaleY);

            proposals.push_back(det);
        }
    }

    // NMS algorithm
    std::sort(proposals.begin(), proposals.end(), [](const Detection& a, const Detection& b){
        return a.confi> b.confi;
    });

    std::vector<int> indices;

    for(size_t i=0; i<proposals.size(); ++i){
        bool keep = true;

        for (int k : indices) {
            if(computeIoU(proposals[i].box, proposals[k].box) > 0.45f){
                keep = false;
                break;
            }
        }
        if (keep) indices.push_back(i);
    }
    for (int idx : indices){
        result.push_back(proposals[idx]);
    }
}