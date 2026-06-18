#ifndef ARM_INFERENCE_HPP_
#define ARM_INFERENCE_HPP_

// 声明 ONNX 目标检测推理封装，对外提供模型加载、单帧推理和类别信息访问接口。

#include <string>
#include <vector>

#include <arm/detection.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace arm
{

// 封装 ONNX 目标检测模型的加载与单帧推理流程。
class Inference
{
public:
  Inference() = default;
  // 直接构造并加载模型；支持指定输入尺寸、类别文件和是否启用 CUDA。
  explicit Inference(
    const std::string & model_path,
    const cv::Size & input_shape = {640, 640},
    const std::string & classes_path = "",
    bool run_with_cuda = true);

  // 重新加载模型及其推理配置。
  void load(
    const std::string & model_path,
    const cv::Size & input_shape = {640, 640},
    const std::string & classes_path = "",
    bool run_with_cuda = true);

  // 对单帧图像执行推理，返回按置信度降序排列的检测结果。
  std::vector<Detection> run(const cv::Mat & frame);
  // 返回当前模型对应的类别名称列表。
  const std::vector<std::string> & classes() const;

private:
  // 从文本文件加载类别名称。
  void loadClassesFromFile();
  // 加载 ONNX 网络并配置推理后端。
  void loadOnnxNetwork();
  // 将非方形图像补边到方形，便于输入固定尺寸模型。
  static cv::Mat formatToSquare(const cv::Mat & source);

  // ONNX 模型文件路径。
  std::string model_path_{};
  // 类别名称文件路径；为空时使用默认类别表。
  std::string classes_path_{};
  // 是否启用 CUDA 推理。
  bool cuda_enabled_{};
  // 默认类别名称，仅在未提供外部类别文件时使用。
  std::vector<std::string> classes_{"instrument", "pill", "tool", "food"};
  // 模型输入尺寸。
  cv::Size2f model_shape_{};
  // 置信度阈值，低于该值的候选框会被丢弃。
  float model_score_threshold_{0.70F};
  // 非极大值抑制阈值，用于去除重复检测框。
  float model_nms_threshold_{0.50F};
  // 是否将输入图像补边到方形。
  bool letter_box_for_square_{true};
  // OpenCV DNN 网络实例。
  cv::dnn::Net net_;
};

}  // namespace arm

#endif  // ARM_INFERENCE_HPP_
