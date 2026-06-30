// 实现检测框约束下的矩形筛选与 solvePnP 位姿估计流程。
#include <arm/pnp_detector.hpp>

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>

namespace arm
{
namespace
{
constexpr bool kEnableGammaCorrection = true;
constexpr double kGamma = 0.7;
constexpr double kPlaneNormalNormEpsilon = 1e-6;
constexpr double kPlaneDistanceEpsilon = 1e-6;

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

cv::Vec3d normalizeVector(const cv::Vec3d & vector)
{
  const double norm = cv::norm(vector);
  if (norm < kPlaneNormalNormEpsilon) {
    throw std::invalid_argument("平面法向长度过小，无法构造平面矫正坐标系");
  }
  return vector / norm;
}

cv::Vec3d rotateAroundAxis(const cv::Vec3d & vector, const cv::Vec3d & axis, double angle)
{
  return vector * std::cos(angle) +
         axis.cross(vector) * std::sin(angle) +
         axis * axis.dot(vector) * (1.0 - std::cos(angle));
}
}  // namespace

PnpDetector::PnpDetector(std::shared_ptr<Inference> inference, const PnpDetectorConfig & config)
: inference_(std::move(inference)),
  clahe_(cv::createCLAHE(2.0)),
  dilate_kernel_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3))),
  enable_rectification_(config.enable_rectification)
{
  if (!inference_) {
    throw std::invalid_argument("PnpDetector 需要有效的 Inference 实例");
  }
  initPnpParameters(config);
}

std::optional<PnpResult> PnpDetector::detectOnce(const cv::Mat & frame)
{
  if (frame.empty()) {
    return std::nullopt;
  }

  const cv::Mat processed_frame = preprocessFrame(frame);
  if (processed_frame.empty()) {
    return std::nullopt;
  }

  cv::imshow("pnp_rectified", processed_frame);
  cv::waitKey(1);

  const cv::Mat gamma_frame = applyGammaCorrection(processed_frame);
  cv::Mat preview = gamma_frame.clone();

  // 先做推理：没有检测框时，矩形约束区域为空，后续边缘处理必然无结果，
  // 因此可在此早返回，省去无目标帧上昂贵的灰度/模糊/CLAHE/Canny 运算。
  const auto detections = inference_->run(gamma_frame);
  if (detections.empty()) {
    cv::imshow("...", preview);
    cv::waitKey(1);
    return std::nullopt;
  }

  cv::Mat gray;
  cv::cvtColor(gamma_frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 10, 20);
  // 使用 CLAHE 提升局部对比度，增强边缘可见性，便于后续矩形轮廓提取。
  clahe_->apply(gray, gray);

  cv::Mat edges;
  cv::Canny(gray, edges, 50, 150);

  cv::Mat mask = cv::Mat::zeros(cv::Size(gamma_frame.cols, gamma_frame.rows), CV_8UC1);
  for (const auto & detection : detections) {
    cv::Rect box = detection.box;
    // 在检测框基础上适度外扩，避免边缘刚好落在框外导致目标轮廓被截断。
    box.x -= static_cast<int>(box.height * 0.1);
    box.y -= static_cast<int>(box.height * 0.1);
    box.height = static_cast<int>(box.height * 1.2);
    box.width = static_cast<int>(box.width * 1.2);
    box &= cv::Rect(0, 0, gamma_frame.cols, gamma_frame.rows);
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

  cv::Mat rvec;
  cv::Mat tvec;
  const bool success = cv::solvePnP(
    object_points_, best_rect_points,
    enable_rectification_ ? rectified_camera_matrix_ : camera_matrix_,
    enable_rectification_ ? rectified_dist_coeffs_ : dist_coeffs_,
    rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

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

  const cv::Mat processed_frame = preprocessFrame(frame);
  if (processed_frame.empty()) {
    return std::nullopt;
  }

  const cv::Mat gamma_frame = applyGammaCorrection(processed_frame);
  const auto detections = inference_->run(gamma_frame);
  if (detections.empty()) {
    return std::nullopt;
  }

  // detections 已按置信度降序排列，取首个即为置信度最高的箱子。
  return detections.front().class_id;
}

void PnpDetector::initPnpParameters(const PnpDetectorConfig & config)
{
  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    config.fx, 0.0, config.cx,
    0.0, config.fy, config.cy,
    0.0, 0.0, 1.0);

  dist_coeffs_ = (cv::Mat_<double>(1, 5) <<
    config.dist_coeffs[0], config.dist_coeffs[1], config.dist_coeffs[2],
    config.dist_coeffs[3], config.dist_coeffs[4]);

  plane_normal_ = normalizeVector(cv::Vec3d(
    config.plane_normal[0], config.plane_normal[1], config.plane_normal[2]));
  plane_yaw_ = config.plane_yaw;
  plane_distance_ = config.plane_distance;
  if (plane_distance_ <= kPlaneDistanceEpsilon) {
    throw std::invalid_argument("平面距离必须大于 0");
  }

  constexpr float width = 0.25F;
  constexpr float height = 0.25F;
  object_points_.clear();
  // 目标物被建模为位于 Z=0 平面上的矩形，四点顺序需与图像点顺序一致。
  object_points_.push_back(cv::Point3f(-width / 2.0F, -height / 2.0F, 0.125F));
  object_points_.push_back(cv::Point3f(width / 2.0F, -height / 2.0F, 0.125F));
  object_points_.push_back(cv::Point3f(width / 2.0F, height / 2.0F, 0.125F));
  object_points_.push_back(cv::Point3f(-width / 2.0F, height / 2.0F, 0.125F));
}

cv::Mat PnpDetector::preprocessFrame(const cv::Mat & frame)
{
  if (!enable_rectification_) {
    return frame;
  }

  if (!ensurePreprocessCache(frame.size())) {
    return cv::Mat();
  }

  cv::Mat undistorted_frame;
  cv::remap(frame, undistorted_frame, undistort_map1_, undistort_map2_, cv::INTER_LINEAR);

  cv::Mat rectified_frame;
  cv::warpPerspective(
    undistorted_frame, rectified_frame, plane_rectification_homography_, frame.size(),
    cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  return rectified_frame;
}

bool PnpDetector::ensurePreprocessCache(const cv::Size & frame_size)
{
  if (preprocess_cache_ready_ && preprocess_frame_size_ == frame_size) {
    return true;
  }

  if (frame_size.width <= 0 || frame_size.height <= 0) {
    return false;
  }

  rectified_camera_matrix_ = camera_matrix_.clone();
  rectified_camera_matrix_.at<double>(0, 2) = static_cast<double>(frame_size.width) * 0.5;
  rectified_camera_matrix_.at<double>(1, 2) = static_cast<double>(frame_size.height) * 0.5;
  rectified_dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

  cv::initUndistortRectifyMap(
    camera_matrix_, dist_coeffs_, cv::Mat(), camera_matrix_, frame_size,
    CV_32FC1, undistort_map1_, undistort_map2_);

  const cv::Vec3d normal = plane_normal_;
  const cv::Vec3d reference_axis =
    std::abs(normal[2]) < 0.9 ? cv::Vec3d(0.0, 0.0, 1.0) : cv::Vec3d(0.0, 1.0, 0.0);
  const cv::Vec3d base_x = normalizeVector(reference_axis.cross(normal));
  const cv::Vec3d plane_x = normalizeVector(rotateAroundAxis(base_x, normal, plane_yaw_));
  const cv::Vec3d plane_y = normalizeVector(normal.cross(plane_x));
  const cv::Vec3d plane_origin = normal * plane_distance_;

  cv::Mat plane_to_camera = (cv::Mat_<double>(3, 3) <<
    plane_x[0], plane_y[0], plane_origin[0],
    plane_x[1], plane_y[1], plane_origin[1],
    plane_x[2], plane_y[2], plane_origin[2]);
  const cv::Mat source_homography = camera_matrix_ * plane_to_camera;

  cv::Mat virtual_plane_to_camera = (cv::Mat_<double>(3, 3) <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, plane_distance_);
  const cv::Mat target_homography = rectified_camera_matrix_ * virtual_plane_to_camera;

  plane_rectification_homography_ = target_homography * source_homography.inv();
  preprocess_frame_size_ = frame_size;
  preprocess_cache_ready_ = true;
  return true;
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

    // 轮廓面积与拟合四边形面积应接近，说明拟合结果可靠。
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

    // 目标物近似正方形，宽高比过大则认为不是目标矩形。
    const float wh_ratio = std::max(rect_w, rect_h) / std::min(rect_w, rect_h);
    if (wh_ratio > 1.35F) {
      continue;
    }

    // 填充率用于排除过于瘦长或凹陷明显的轮廓。
    const double fill_ratio = approx_area / (rect_w * rect_h);
    if (fill_ratio < 0.65 || fill_ratio > 1.1) {
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

    // 四个角应接近直角，用于排除透视过大或形状明显不符的候选。
    const float min_angle = *std::min_element(angles.begin(), angles.end());
    const float max_angle = *std::max_element(angles.begin(), angles.end());
    if (min_angle < 60.0F || max_angle > 120.0F) {
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
