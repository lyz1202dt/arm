#ifndef ARM_COLOR_PNP_DETECTOR_HPP_
#define ARM_COLOR_PNP_DETECTOR_HPP_

// 声明基于 LAB 色块的正方形角点提取与 solvePnP 中心点估计接口。

#include <array>
#include <optional>
#include <vector>

#include <arm/pnp_detector.hpp>
#include <opencv2/opencv.hpp>

namespace arm
{

// 通过四个已知颜色的色块定位正方形四角，再用 solvePnP 估计正方形中心的空间位置。
class ColorPnpDetector
{
public:
  ColorPnpDetector();

  // 对单帧图像执行一次识别，成功提取到四个色块角点并解算后返回中心三维坐标。
  std::optional<PnpResult> detectOnce(const cv::Mat & frame);

private:
  // 以 OpenCV 8 位 LAB 尺度表示的单个色块阈值范围。
  struct LabThreshold
  {
    cv::Scalar lower;
    cv::Scalar upper;
  };

  // 初始化相机内参、畸变参数与正方形四角的三维坐标。
  void initPnpParameters();
  // 初始化四个色块的 LAB 阈值（由标准 CIELAB 数值换算到 OpenCV 8 位尺度）。
  void initColorThresholds();
  // 在 LAB 图像中按单个阈值提取最大色块的质心。
  std::optional<cv::Point2f> findColorCentroid(
    const cv::Mat & lab, const LabThreshold & threshold) const;
  // 将四个角点排序为与 object_points_ 对应的稳定顺序（左上、右上、右下、左下）。
  static std::vector<cv::Point2f> sortSquarePoints(std::vector<cv::Point2f> points);

  // 相机内参矩阵。
  cv::Mat camera_matrix_;
  // 畸变参数。
  cv::Mat dist_coeffs_;
  // 正方形四角在物体坐标系下的三维坐标（Z=0 平面）。
  std::vector<cv::Point3f> object_points_;
  // 四个色块的 LAB 阈值。
  std::array<LabThreshold, 4> color_thresholds_;
};

}  // namespace arm

#endif  // ARM_COLOR_PNP_DETECTOR_HPP_
