#ifndef ARM_BOX_GRID_DETECTOR_HPP_
#define ARM_BOX_GRID_DETECTOR_HPP_

// 声明箱子编号排序与多帧稳定判定接口。

#include <atomic>
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

  // 持续采样摄像头画面，使用最近 7 帧有效识别结果逐位置多数投票后返回。
  std::optional<std::array<int32_t, 8>> detectStableGrid(
    cv::VideoCapture & camera, const std::atomic_bool & keep_running);
  // 销毁识别预览窗口；可由外部取消路径调用，保证窗口及时关闭。
  static void destroyPreviewWindows();

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
};

}  // namespace arm

#endif  // ARM_BOX_GRID_DETECTOR_HPP_
