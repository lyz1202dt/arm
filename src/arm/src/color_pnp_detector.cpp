// 实现基于四个 LAB 色块的正方形角点提取与 solvePnP 中心点位姿估计流程。
#include <arm/color_pnp_detector.hpp>

#include <algorithm>
#include <cmath>
#include <opencv2/highgui.hpp>

namespace arm
{
namespace
{
// 正方形边长占位符，单位米；实际测量后请直接修改此处。
constexpr float kSquareSide = 0.10F;

// 单个色块在标准 CIELAB 尺度下的阈值（L:0~100，a/b:-128~127）。
struct CieLabThreshold
{
  double l_min;
  double l_max;
  double a_min;
  double a_max;
  double b_min;
  double b_max;
};

// 四个色块的 LAB 阈值，顺序与正方形四角的颜色一一对应（具体角位由几何排序决定）。
constexpr std::array<CieLabThreshold, 4> kCieLabThresholds{{
  {43, 68, 7, 70, 0, 86},
  {0, 52, -37, 36, -80, -15},
  {57, 79, -30, 20, -17, 7},
  {47, 74, -88, -25, -1, 52},
}};

// 把数值限制在 OpenCV 8 位通道的有效区间 [0, 255]。
double clampToByte(double value)
{
  return std::max(0.0, std::min(255.0, value));
}

}  // namespace

ColorPnpDetector::ColorPnpDetector()
{
  initPnpParameters();
  initColorThresholds();
}

void ColorPnpDetector::initPnpParameters()
{
  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    330.732920, 0.000000, 323.284477,
    0.000000, 330.604624, 235.624095,
    0.000000, 0.000000, 1.000000);

  dist_coeffs_ = (cv::Mat_<double>(1, 5) <<
    -0.015437, -0.017894, -0.000542, 0.001233, 0.000000);

  const float half = kSquareSide / 2.0F;
  object_points_.clear();
  // 正方形以中心为原点平放于 Z=0 平面，四角顺序为左上、右上、右下、左下，
  // 因而 solvePnP 解出的平移向量即为正方形中心的空间坐标。
  object_points_.push_back(cv::Point3f(-half, -half, 0));
  object_points_.push_back(cv::Point3f(half, -half, 0));
  object_points_.push_back(cv::Point3f(half, half, 0));
  object_points_.push_back(cv::Point3f(-half, half, 0));
}

void ColorPnpDetector::initColorThresholds()
{
  for (std::size_t i = 0; i < color_thresholds_.size(); ++i) {
    const CieLabThreshold & src = kCieLabThresholds[i];
    // 标准 CIELAB 到 OpenCV 8 位 LAB 的换算：L*255/100，a/b 各加 128。
    color_thresholds_[i].lower = cv::Scalar(
      clampToByte(src.l_min * 255.0 / 100.0),
      clampToByte(src.a_min + 128.0),
      clampToByte(src.b_min + 128.0));
    color_thresholds_[i].upper = cv::Scalar(
      clampToByte(src.l_max * 255.0 / 100.0),
      clampToByte(src.a_max + 128.0),
      clampToByte(src.b_max + 128.0));
  }
}

std::optional<PnpResult> ColorPnpDetector::detectOnce(const cv::Mat & frame)
{
  if (frame.empty()) {
    return std::nullopt;
  }

  cv::Mat preview = frame.clone();

  cv::Mat lab;
  cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);

  std::vector<cv::Point2f> corners;
  corners.reserve(color_thresholds_.size());
  for (const auto & threshold : color_thresholds_) {
    const auto centroid = findColorCentroid(lab, threshold);
    if (!centroid) {
      // 任一色块缺失都无法构成正方形，放弃本帧。
      cv::imshow("...", preview);
      cv::waitKey(1);
      return std::nullopt;
    }
    corners.push_back(*centroid);
  }

  corners = sortSquarePoints(corners);

  cv::Mat rvec;
  cv::Mat tvec;
  const bool success = cv::solvePnP(
    object_points_, corners, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_ITERATIVE);

  if (!success || tvec.empty()) {
    cv::imshow("...", preview);
    cv::waitKey(1);
    return std::nullopt;
  }

  for (int i = 0; i < 4; ++i) {
    const cv::Point start = corners[i];
    const cv::Point end = corners[(i + 1) % 4];
    cv::line(preview, start, end, cv::Scalar(0, 255, 0), 2);
    cv::circle(preview, start, 4, cv::Scalar(0, 0, 255), -1);
  }
  cv::imshow("...", preview);
  cv::waitKey(1);

  return PnpResult{tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)};
}

std::optional<cv::Point2f> ColorPnpDetector::findColorCentroid(
  const cv::Mat & lab, const LabThreshold & threshold) const
{
  cv::Mat mask;
  cv::inRange(lab, threshold.lower, threshold.upper, mask);

  // 形态学开运算去除零散噪点，闭运算填补色块内部空洞。
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double best_area = 0.0;
  const std::vector<cv::Point> * best_contour = nullptr;
  for (const auto & contour : contours) {
    const double area = cv::contourArea(contour);
    // 过滤过小连通域，避免噪声被误判为色块。
    if (area < 100.0) {
      continue;
    }
    if (area > best_area) {
      best_area = area;
      best_contour = &contour;
    }
  }

  if (best_contour == nullptr) {
    return std::nullopt;
  }

  const cv::Moments moments = cv::moments(*best_contour);
  if (std::abs(moments.m00) < 1e-6) {
    return std::nullopt;
  }

  return cv::Point2f(
    static_cast<float>(moments.m10 / moments.m00),
    static_cast<float>(moments.m01 / moments.m00));
}

std::vector<cv::Point2f> ColorPnpDetector::sortSquarePoints(std::vector<cv::Point2f> points)
{
  if (points.size() != 4) {
    return points;
  }

  cv::Point2f center(0, 0);
  for (const auto & point : points) {
    center += point;
  }
  center *= 0.25F;

  // 先按相对中心点的极角排序，确保四个角按统一方向排列。
  std::sort(points.begin(), points.end(), [center](const cv::Point2f & lhs, const cv::Point2f & rhs) {
    return std::atan2(lhs.y - center.y, lhs.x - center.x) <
           std::atan2(rhs.y - center.y, rhs.x - center.x);
  });

  int top_left_index = 0;
  float min_sum = points[0].x + points[0].y;
  for (int i = 1; i < 4; ++i) {
    const float sum = points[i].x + points[i].y;
    if (sum < min_sum) {
      min_sum = sum;
      top_left_index = i;
    }
  }

  // 把左上角旋转到首位，得到与 object_points_ 对应的稳定点序。
  std::vector<cv::Point2f> sorted(4);
  for (int i = 0; i < 4; ++i) {
    sorted[i] = points[(top_left_index + i) % 4];
  }
  return sorted;
}

}  // namespace arm
