#ifndef ARM_BOX_GRID_DETECTOR_HPP_
#define ARM_BOX_GRID_DETECTOR_HPP_

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <arm/detection.hpp>
#include <arm/inference.hpp>
#include <opencv2/opencv.hpp>

namespace arm
{

// 基于目标检测结果识别 2x4 箱子矩阵，并输出从左上到右下的一维编号数组。
class BoxGridDetector
{
public:
  explicit BoxGridDetector(std::shared_ptr<Inference> inference);

  // 连续采样摄像头画面，只有当多次结果一致时才返回稳定矩阵。
  std::optional<std::array<int32_t, 8>> detectStableGrid(cv::VideoCapture & camera);

private:
  // 对单帧图像执行一次矩阵识别；若不是完整的 2x4 结果则返回空。
  std::optional<std::array<int32_t, 8>> detectGridOnce(const cv::Mat & frame) const;
  // 按检测框中心点的垂直位置分成两排，并在每排内部按水平位置排序。
  std::vector<std::vector<Detection>> groupDetectionsByRows(
    const std::vector<Detection> & detections) const;

  // 共享的目标检测推理器。
  std::shared_ptr<Inference> inference_;
  // 判定结果稳定所需的连续一致次数。
  int stable_count_{3};
  // 为获得稳定结果允许尝试的最大帧数。
  int max_attempts_{100};
};

}  // namespace arm

#endif  // ARM_BOX_GRID_DETECTOR_HPP_
