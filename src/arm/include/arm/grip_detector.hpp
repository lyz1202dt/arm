#ifndef ARM_GRIP_DETECTOR_HPP_
#define ARM_GRIP_DETECTOR_HPP_

#include <opencv2/opencv.hpp>

namespace arm
{

// 使用图像整体亮度的简单阈值法判断是否抓取成功。
class GripDetector
{
public:
  explicit GripDetector(double brightness_threshold = 60.0);

  // 对单帧图像执行抓取结果判断；当前约定 1.0 表示抓到，-1.0 表示未抓到或输入无效。
  float detectOnce(const cv::Mat & frame) const;
  // 返回当前使用的亮度阈值。
  double brightnessThreshold() const;

private:
  // 判定抓取成功与否的平均亮度阈值。
  double brightness_threshold_;
};

}  // namespace arm

#endif  // ARM_GRIP_DETECTOR_HPP_
