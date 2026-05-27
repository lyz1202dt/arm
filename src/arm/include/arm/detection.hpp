#ifndef ARM_DETECTION_HPP_
#define ARM_DETECTION_HPP_

#include <string>

#include <opencv2/core.hpp>

namespace arm
{

// 表示一次目标检测的基础结果，供矩阵识别与 PnP 模块复用。
struct Detection
{
  // 模型输出的类别索引。
  int class_id{0};
  // 类别索引对应的名称；若未加载类别文件则可能为空。
  std::string class_name{};
  // 检测置信度。
  float confidence{0.0F};
  // 为可视化或调试预留的显示颜色。
  cv::Scalar color{};
  // 检测框，坐标基于原始输入图像。
  cv::Rect box{};
};

}  // namespace arm

#endif  // ARM_DETECTION_HPP_
