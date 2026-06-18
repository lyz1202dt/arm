#ifndef ARM_PNP_DETECTOR_HPP_
#define ARM_PNP_DETECTOR_HPP_

// 声明基于检测框与几何约束的 PnP 位姿识别接口及其结果结构。

#include <memory>
#include <optional>
#include <vector>

#include <arm/inference.hpp>
#include <opencv2/opencv.hpp>

namespace arm
{

// 表示一次 PnP 位姿识别得到的平移向量结果。
struct PnpResult
{
  double x{};
  double y{};
  double z{};
};

// 在目标检测框约束下搜索近似矩形目标，并通过 solvePnP 估计空间位置。
class PnpDetector
{
public:
  explicit PnpDetector(std::shared_ptr<Inference> inference);

  // 对单帧图像执行一次 PnP 识别，成功时返回三维平移结果。
  std::optional<PnpResult> detectOnce(const cv::Mat & frame);

private:
  // 初始化相机内参、畸变参数和目标物的三维角点。
  void initPnpParameters();
  // 在边缘图中筛选最符合要求的矩形轮廓。
  std::vector<cv::Point2f> checkRect(
    cv::Mat & edge,
    const cv::Mat & mask,
    std::vector<cv::Point2f> & best_rect_points) const;
  // 将矩形四点排序为 solvePnP 可稳定使用的顺序。
  static std::vector<cv::Point2f> sortRectanglePoints(std::vector<cv::Point2f> points);

  // 共享的目标检测推理器。
  std::shared_ptr<Inference> inference_;
  // 相机内参矩阵。
  cv::Mat camera_matrix_;
  // 畸变参数。
  cv::Mat dist_coeffs_;
  // 目标物四个角点在物体坐标系下的三维坐标。
  std::vector<cv::Point3f> object_points_;
  // 用于局部对比度增强的 CLAHE 算子。
  cv::Ptr<cv::CLAHE> clahe_;
};

}  // namespace arm

#endif  // ARM_PNP_DETECTOR_HPP_
