#ifndef ARM_PNP_DETECTOR_HPP_
#define ARM_PNP_DETECTOR_HPP_

// 声明基于检测框与几何约束的 PnP 位姿识别接口及其结果结构。

#include <array>
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

struct PnpDetectorConfig
{
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  std::array<double, 5> dist_coeffs{};
  std::array<double, 3> plane_normal{0.0, 0.0, 1.0};
  double plane_yaw{};
  double plane_distance{1.0};
  bool enable_rectification{false};
};

// 在目标检测框约束下搜索近似矩形目标，并通过 solvePnP 估计空间位置。
class PnpDetector
{
public:
  PnpDetector(std::shared_ptr<Inference> inference, const PnpDetectorConfig & config);

  // 对单帧图像执行一次 PnP 识别，成功时返回三维平移结果。
  std::optional<PnpResult> detectOnce(const cv::Mat & frame);

  // 对单帧图像执行一次 YOLO 推理，返回置信度最高的箱子类别索引。
  std::optional<int> detectBoxIndex(const cv::Mat & frame);

private:
  // 初始化相机内参、畸变参数和目标物的三维角点。
  void initPnpParameters(const PnpDetectorConfig & config);
  // 在边缘图中筛选最符合要求的矩形轮廓。
  std::vector<cv::Point2f> checkRect(
    cv::Mat & edge,
    const cv::Mat & mask,
    std::vector<cv::Point2f> & best_rect_points) const;
  // 将矩形四点排序为 solvePnP 可稳定使用的顺序。
  static std::vector<cv::Point2f> sortRectanglePoints(std::vector<cv::Point2f> points);
  // 对输入图像执行去畸变与平面矫正。
  cv::Mat preprocessFrame(const cv::Mat & frame);
  // 根据当前图像尺寸初始化去畸变映射与平面矫正矩阵。
  bool ensurePreprocessCache(const cv::Size & frame_size);

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
  // checkRect 中边缘膨胀所用的固定结构元素，构造时建好后每帧复用。
  cv::Mat dilate_kernel_;
  // 是否启用输入图像平面矫正。
  bool enable_rectification_{false};
  // 虚拟俯视相机与原始相机共享输出分辨率时使用的内参。
  cv::Mat rectified_camera_matrix_;
  // 俯视图对应的零畸变参数。
  cv::Mat rectified_dist_coeffs_;
  // 平面法向（相机坐标系）。
  cv::Vec3d plane_normal_{0.0, 0.0, 1.0};
  // 绕平面法向旋转角（弧度）。
  double plane_yaw_{};
  // 相机到目标平面的法向距离。
  double plane_distance_{1.0};
  // 当前缓存对应的输入图尺寸。
  cv::Size preprocess_frame_size_;
  // 去畸变 remap 映射。
  cv::Mat undistort_map1_;
  cv::Mat undistort_map2_;
  // 从去畸变图到俯视图的单应矩阵。
  cv::Mat plane_rectification_homography_;
  // 预处理缓存是否已就绪。
  bool preprocess_cache_ready_{false};
};

}  // namespace arm

#endif  // ARM_PNP_DETECTOR_HPP_
