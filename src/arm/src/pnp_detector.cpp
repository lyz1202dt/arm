// 实现检测框约束下的矩形筛选与 solvePnP 位姿估计流程。
#include <arm/pnp_detector.hpp>

#include <algorithm>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <stdexcept>

namespace arm
{
namespace
{
constexpr bool kEnableGammaCorrection = true;
constexpr double kGamma = 0.7;
constexpr int kMaskExpandPixels = 3;

// 伽马校正查找表只依赖编译期常量 kGamma，首次调用时构建一次后复用，避免每帧重建。
const cv::Mat & gammaLut()
{
  static const cv::Mat lut = [] {
    cv::Mat table(1, 256, CV_8UC1);
    unsigned char * table_data = table.ptr<unsigned char>();
    for (int i = 0; i < 256; ++i) {
      const double normalized = static_cast<double>(i) / 255.0;
      table_data[i] = cv::saturate_cast<unsigned char>(std::pow(normalized, kGamma) * 255.0);
    }
    return table;
  }();
  return lut;
}

cv::Mat applyGammaCorrection(const cv::Mat & frame)
{
  if (!kEnableGammaCorrection) {
    return frame;
  }

  cv::Mat corrected;
  cv::LUT(frame, gammaLut(), corrected);
  return corrected;
}
}  // namespace

PnpDetector::PnpDetector(std::shared_ptr<Inference> inference)
: inference_(std::move(inference)),
  clahe_(cv::createCLAHE(2.0)),
  dilate_kernel_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)))
{
  if (!inference_) {
    throw std::invalid_argument("PnpDetector 需要有效的 Inference 实例");
  }
  initPnpParameters();
}

std::optional<PnpResult> PnpDetector::detectOnce(const cv::Mat & frame)
{
  if (frame.empty()) {
    return std::nullopt;
  }

  const cv::Mat gamma_frame = applyGammaCorrection(frame);
  cv::Mat undistorted_frame;
  cv::undistort(gamma_frame, undistorted_frame, camera_matrix_, dist_coeffs_);
  cv::Mat preview = undistorted_frame.clone();

  cv::Mat gray;
  cv::cvtColor(undistorted_frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 10, 20);
  // 使用 CLAHE 提升局部对比度，增强边缘可见性，便于后续矩形轮廓提取。
  clahe_->apply(gray, gray);

  cv::Mat edges;
  cv::Canny(gray, edges, 50, 150);

  const auto detections = inference_->run(undistorted_frame);
  if (detections.empty()) {
    cv::imshow("...", preview);
    cv::waitKey(1);
    return std::nullopt;
  }

  cv::Mat mask = cv::Mat::zeros(cv::Size(undistorted_frame.cols, undistorted_frame.rows), CV_8UC1);
  for (const auto & detection : detections) {
    cv::Rect box = detection.box;
    // 参考项目会对检测框做固定像素外扩，避免目标边缘恰好落在框外导致轮廓被截断。
    box.x -= kMaskExpandPixels;
    box.y -= kMaskExpandPixels;
    box.width += 2 * kMaskExpandPixels;
    box.height += 2 * kMaskExpandPixels;
    box &= cv::Rect(0, 0, undistorted_frame.cols, undistorted_frame.rows);
    if (box.width > 0 && box.height > 0) {
      cv::rectangle(mask, box, cv::Scalar(255), -1);
    }
  }

  std::vector<cv::Point2f> best_rect_points;
  // 只在目标检测框覆盖的区域内寻找矩形，减少背景干扰。
  checkRect(edges, mask, best_rect_points);
  if (best_rect_points.size() != 4) {
    cv::imshow("...", preview);
    cv::waitKey(1);
    return std::nullopt;
  }

  best_rect_points = sortRectanglePoints(best_rect_points);

  std::vector<cv::Point2f> normalized_rect_points;
  cv::undistortPoints(
    best_rect_points, normalized_rect_points, camera_matrix_, cv::noArray());

  cv::Mat rvec;
  cv::Mat tvec;
  const bool success = cv::solvePnP(
    object_points_, normalized_rect_points, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), rvec, tvec,
    false, cv::SOLVEPNP_IPPE);

  if (!success || tvec.empty()) {
    cv::imshow("...", preview);
    cv::waitKey(1);
    return std::nullopt;
  }

  for (int i = 0; i < 4; ++i) {
    const cv::Point start = best_rect_points[i];
    const cv::Point end = best_rect_points[(i + 1) % 4];
    cv::line(preview, start, end, cv::Scalar(0, 255, 0), 2);
    cv::circle(preview, start, 4, cv::Scalar(0, 0, 255), -1);
  }
  cv::imshow("...", preview);
  cv::waitKey(1);

  return PnpResult{tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)};
}

std::optional<int> PnpDetector::detectBoxIndex(const cv::Mat & frame)
{
  if (frame.empty()) {
    return std::nullopt;
  }

  const cv::Mat gamma_frame = applyGammaCorrection(frame);
  cv::Mat undistorted_frame;
  cv::undistort(gamma_frame, undistorted_frame, camera_matrix_, dist_coeffs_);
  const auto detections = inference_->run(undistorted_frame);
  if (detections.empty()) {
    return std::nullopt;
  }

  // detections 已按置信度降序排列，取首个即为置信度最高的箱子。
  return detections.front().class_id;
}

void PnpDetector::initPnpParameters()
{
  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    331.108119, 0.000000, 329.710143,
    0.000000, 331.215455, 243.548501,
    0.000000, 0.000000, 1.000000);


  dist_coeffs_ = (cv::Mat_<double>(1, 5) <<
    -0.017538,-0.016338,-0.001018,-0.000804,0.000000);

  constexpr float width = 0.25F;
  constexpr float height = 0.25F;
  object_points_.clear();
  // 目标物被建模为位于 Z=0 平面上的矩形，四点顺序需与图像点顺序一致。
  object_points_.push_back(cv::Point3f(-width / 2.0F, -height / 2.0F, 0.0F));
  object_points_.push_back(cv::Point3f(width / 2.0F, -height / 2.0F, 0.0F));
  object_points_.push_back(cv::Point3f(width / 2.0F, height / 2.0F, 0.0F));
  object_points_.push_back(cv::Point3f(-width / 2.0F, height / 2.0F, 0.0F));
}

std::vector<cv::Point2f> PnpDetector::checkRect(
  cv::Mat & edge,
  const cv::Mat & mask,
  std::vector<cv::Point2f> & best_rect_points) const
{
  cv::bitwise_and(edge, mask, edge);

  cv::dilate(edge, edge, dilate_kernel_);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edge, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  double best_area = 0.0;
  for (const auto & contour : contours) {
    const double area = cv::contourArea(contour);
    // 先过滤掉过小轮廓，避免噪声边缘参与后续拟合。
    if (area < 1000.0) {
      continue;
    }

    std::vector<cv::Point2f> approx;
    const double epsilon = 0.02 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    // 候选目标必须是凸四边形。
    if (approx.size() != 4 || !cv::isContourConvex(approx)) {
      continue;
    }

    const double approx_area = cv::contourArea(approx);
    if (approx_area < 1000.0) {
      continue;
    }

  // 轮廓面积与拟合四边形面积之比优先对齐参考项目，再叠加当前项目的额外约束。
  const double area_ratio = area / approx_area;
  if (area_ratio < 0.8 || area_ratio > 1.2) {
    continue;
  }

    const cv::RotatedRect min_rect = cv::minAreaRect(approx);
    const float rect_w = min_rect.size.width;
    const float rect_h = min_rect.size.height;
    if (rect_w < 1e-3F || rect_h < 1e-3F) {
      continue;
    }

    // 保留当前项目的宽高比约束，抑制明显偏离箱面形状的候选。
    const float wh_ratio = std::max(rect_w, rect_h) / std::min(rect_w, rect_h);
    if (wh_ratio > 1.8F) {
      continue;
    }

    // 保留当前项目的填充率过滤，继续作为误检保护条件。
    const double fill_ratio = approx_area / (rect_w * rect_h);
    if (fill_ratio < 0.5 || fill_ratio > 1.15) {
      continue;
    }

    std::vector<float> angles;
    for (int j = 0; j < 4; ++j) {
      const cv::Point2f a = approx[(j + 3) % 4];
      const cv::Point2f b = approx[j];
      const cv::Point2f c = approx[(j + 1) % 4];
      const cv::Point2f v1 = a - b;
      const cv::Point2f v2 = c - b;

      const float len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
      const float len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
      if (len1 < 1e-6F || len2 < 1e-6F) {
        continue;
      }

      float cos_value = (v1.x * v2.x + v1.y * v2.y) / (len1 * len2);
      cos_value = std::max(-1.0F, std::min(1.0F, cos_value));
      angles.push_back(std::acos(cos_value) * 180.0F / CV_PI);
    }

    if (angles.size() != 4) {
      continue;
    }

    // 角度范围对齐参考项目，允许一定透视但排除明显非矩形候选。
    const float min_angle = *std::min_element(angles.begin(), angles.end());
    const float max_angle = *std::max_element(angles.begin(), angles.end());
    if (min_angle < 30.0F || max_angle > 150.0F) {
      continue;
    }

    // 在多个合法候选中优先保留面积最大的一个。
    if (area > best_area) {
      best_area = area;
      best_rect_points = approx;
    }
  }

  return best_rect_points;
}

std::vector<cv::Point2f> PnpDetector::sortRectanglePoints(std::vector<cv::Point2f> points)
{
  if (points.size() != 4) {
    return points;
  }

  cv::Point2f center(0, 0);
  for (const auto & point : points) {
    center += point;
  }
  center *= 0.25F;

  // 先按相对中心点的极角排序，确保四个角按顺时针或逆时针排列。
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

  // 再把左上角旋转到首位，得到与 object_points_ 对应的稳定点序。
  std::vector<cv::Point2f> sorted(4);
  for (int i = 0; i < 4; ++i) {
    sorted[i] = points[(top_left_index + i) % 4];
  }
  return sorted;
}

}  // namespace arm
