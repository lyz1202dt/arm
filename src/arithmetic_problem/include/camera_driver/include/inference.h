#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);
    std::vector<Detection> runInference(const cv::Mat &input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};

    // 类别顺序必须与训练时的输出一致，供识别结果和绘制标签直接索引。
    std::vector<std::string> classes{
        "0","1","2","3","4","5","6","7","8","9",
        "+","-","×","÷","(",")"};
    cv::Size2f modelShape{};

    // confidence 用于初筛，score 用于类别分数过滤，NMS 用于去除重叠框。
    float modelConfidenceThreshold {0.25};
    float modelScoreThreshold      {0.25};
    float modelNMSThreshold        {0.45};

    // 仅在输入为正方形且模型要求固定尺寸时启用，避免直接拉伸导致字符变形。
    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

#endif // INFERENCE_H
