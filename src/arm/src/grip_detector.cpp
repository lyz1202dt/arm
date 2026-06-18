// 实现基于整帧平均亮度阈值的轻量抓取结果判断逻辑。
#include <arm/grip_detector.hpp>
#include <opencv2/highgui.hpp>

namespace arm
{

GripDetector::GripDetector(const double brightness_threshold)
: brightness_threshold_(brightness_threshold)
{
}

float GripDetector::detectOnce(const cv::Mat & frame) const
{
  if (frame.empty()) {
    return -1.0F;
  }

  cv::imshow("...",frame);
  cv::waitKey(1);

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  const double brightness = cv::mean(gray)[0];
  // 当前实现用整幅图像平均亮度近似判断抓取状态：亮度低于阈值视为抓到目标。
  return brightness < brightness_threshold_ ? 1.0F : -1.0F;
}

double GripDetector::brightnessThreshold() const
{
  return brightness_threshold_;
}

}  // namespace arm
