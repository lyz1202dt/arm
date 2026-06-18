#ifndef ARM_BOX_GRID_DETECTOR_HPP_
#define ARM_BOX_GRID_DETECTOR_HPP_

// 声明箱子编号排序与多帧稳定判定接口。

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <arm/detection.hpp>
#include <arm/inference.hpp>
#include <opencv2/opencv.hpp>

namespace arm
{

// 基于目标检测结果提取排序后的箱子编号序列，并输出稳定的 8 位结果。
class BoxGridDetector
{
public:
  explicit BoxGridDetector(std::shared_ptr<Inference> inference);

  // 连续采样摄像头画面，只有当连续多帧都得到相同的 8 个编号结果时才返回。
  std::optional<std::array<int32_t, 8>> detectStableGrid(cv::VideoCapture & camera);

private:
  // 对单帧图像执行一次排序，提取当前帧从左上到右下的编号序列。
  std::vector<int32_t> collectOrderedIds(const cv::Mat & frame) const;
  // 对单帧图像执行一次识别；只有当前帧恰好得到 8 个编号时才返回固定长度结果。
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
