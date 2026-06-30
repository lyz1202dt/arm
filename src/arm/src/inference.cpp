// 实现 ONNX 目标检测模型的加载、预处理、推理与后处理，作为各视觉模块共享的基础能力。
#include <arm/inference.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace arm
{
namespace
{

// 为不同类别生成稳定的伪随机颜色，便于调试时区分检测结果。
cv::Scalar makeColor(const int class_id)
{
  cv::RNG rng(static_cast<uint64>(class_id + 1) * 12345U);
  return {
    static_cast<double>(rng.uniform(0, 255)),
    static_cast<double>(rng.uniform(0, 255)),
    static_cast<double>(rng.uniform(0, 255))};
}

}  // namespace

Inference::Inference(
  const std::string & model_path,
  const cv::Size & input_shape,
  const std::string & classes_path,
  const bool run_with_cuda)
{
  load(model_path, input_shape, classes_path, run_with_cuda);
}

void Inference::load(
  const std::string & model_path,
  const cv::Size & input_shape,
  const std::string & classes_path,
  const bool run_with_cuda)
{
  model_path_ = model_path;
  model_shape_ = input_shape;
  classes_path_ = classes_path;
  cuda_enabled_ = run_with_cuda;

  // 若提供了类别文件，则使用外部类别名覆盖默认类别表。
  if (!classes_path_.empty()) {
    loadClassesFromFile();
  }
  loadOnnxNetwork();
}

std::vector<Detection> Inference::run(const cv::Mat & frame)
{
  if (frame.empty()) {
    return {};
  }

  if (net_.empty()) {
    throw std::runtime_error("推理网络尚未初始化");
  }

  cv::Mat model_input = frame;
  // 当前模型默认使用方形输入，非方形图像会先补边再送入网络。
  if (letter_box_for_square_ && model_shape_.width == model_shape_.height) {
    model_input = formatToSquare(model_input);
  }

  cv::Mat blob;
  // 归一化到 [0,1]、缩放到模型输入尺寸并交换 R/B 通道，得到网络输入张量。
  cv::dnn::blobFromImage(model_input, blob, 1.0 / 255.0, model_shape_, cv::Scalar(), true, false);
  net_.setInput(blob);

  std::vector<cv::Mat> outputs;
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());
  if (outputs.empty()) {
    return {};
  }

  cv::Mat output = outputs[0];
  int rows = output.size[1];
  int dimensions = output.size[2];

  // 兼容不同 YOLO 导出格式：有的输出是 [1, N, C]，有的是 [1, C, N]。
  if (dimensions > rows) {
    rows = output.size[2];
    dimensions = output.size[1];
    output = output.reshape(1, dimensions);
    cv::transpose(output, output);
  } else {
    output = output.reshape(1, rows);
  }

  float * data = reinterpret_cast<float *>(output.data);
  // 网络输入经过补边缩放，需用缩放因子把框坐标还原回原图尺度。
  const float x_factor = static_cast<float>(model_input.cols) / model_shape_.width;
  const float y_factor = static_cast<float>(model_input.rows) / model_shape_.height;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows; ++i) {
    const float * class_scores = data + 4;
    cv::Mat scores(1, static_cast<int>(classes_.size()), CV_32FC1, const_cast<float *>(class_scores));
    cv::Point class_id_point;
    double max_class_score = 0.0;
    cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

    // 每一行前 4 个值是框信息，后面才是各类别分数；这里取最大类别分数作为该候选的类别置信度。
    if (max_class_score > model_score_threshold_) {
      // 模型输出的是中心点坐标加宽高，转换为左上角坐标的矩形并映射回原图。
      const float x = data[0];
      const float y = data[1];
      const float w = data[2];
      const float h = data[3];

      const int left = static_cast<int>((x - 0.5F * w) * x_factor);
      const int top = static_cast<int>((y - 0.5F * h) * y_factor);
      const int width = static_cast<int>(w * x_factor);
      const int height = static_cast<int>(h * y_factor);

      cv::Rect box(left, top, width, height);
      // 与图像边界求交，裁掉越界部分，丢弃无效空框。
      box &= cv::Rect(0, 0, frame.cols, frame.rows);
      if (box.width > 0 && box.height > 0) {
        confidences.push_back(static_cast<float>(max_class_score));
        class_ids.push_back(class_id_point.x);
        boxes.push_back(box);
      }
    }

    // 步进到下一候选行。
    data += dimensions;
  }

  std::vector<int> nms_indices;
  // 用非极大值抑制去掉同一目标上的重叠候选框。
  cv::dnn::NMSBoxes(boxes, confidences, model_score_threshold_, model_nms_threshold_, nms_indices);

  std::vector<Detection> detections;
  detections.reserve(nms_indices.size());
  for (const int index : nms_indices) {
    Detection detection;
    detection.class_id = class_ids[index];
    detection.confidence = confidences[index];
    detection.color = makeColor(detection.class_id);
    if (detection.class_id >= 0 && detection.class_id < static_cast<int>(classes_.size())) {
      detection.class_name = classes_[detection.class_id];
    }
    detection.box = boxes[index];
    detections.push_back(detection);
  }

  // 统一按置信度降序输出，方便上层优先处理更可靠的结果。
  std::sort(detections.begin(), detections.end(), [](const Detection & lhs, const Detection & rhs) {
    return lhs.confidence > rhs.confidence;
  });

  return detections;
}

const std::vector<std::string> & Inference::classes() const
{
  return classes_;
}

void Inference::loadClassesFromFile()
{
  std::ifstream input_file(classes_path_);
  if (!input_file.is_open()) {
    throw std::runtime_error("无法打开类别文件: " + classes_path_);
  }

  classes_.clear();
  for (std::string class_line; std::getline(input_file, class_line); ) {
    if (!class_line.empty()) {
      classes_.push_back(class_line);
    }
  }

  if (classes_.empty()) {
    throw std::runtime_error("类别文件为空: " + classes_path_);
  }
}

void Inference::loadOnnxNetwork()
{
  net_ = cv::dnn::readNetFromONNX(model_path_);
  if (net_.empty()) {
    throw std::runtime_error("无法加载 ONNX 模型: " + model_path_);
  }

  // 根据参数选择 CUDA 或 CPU 推理后端，便于在不同机器环境下切换。
  /*if (cuda_enabled_) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  } else*/ {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
}

cv::Mat Inference::formatToSquare(const cv::Mat & source)
{
  // 取长边作为边长生成黑底方图，把原图贴在左上角，保持纵横比不被拉伸。
  const int col = source.cols;
  const int row = source.rows;
  const int max_dim = std::max(col, row);
  cv::Mat result = cv::Mat::zeros(max_dim, max_dim, CV_8UC3);
  source.copyTo(result(cv::Rect(0, 0, col, row)));
  return result;
}

}  // namespace arm
