// 实现 PnP 三维坐标的卡尔曼平滑与离群值剔除逻辑。
#include <arm/coordinate_filter.hpp>

#include <cmath>

namespace arm
{
namespace
{
// 过程噪声 Q：值越大越信任新测量，跟随更快但更抖。
constexpr double kProcessNoise = 1e-4;
// 测量噪声 R：值越大越信任历史估计，平滑更强但响应更慢。
constexpr double kMeasurementNoise = 4e-3;
// 离群判定阈值，单位米：任一轴测量值与当前估计偏差超过该值即视为极端值并抛弃。
constexpr double kOutlierThreshold = 0.05;
}  // namespace

ScalarKalmanFilter::ScalarKalmanFilter(double process_noise, double measurement_noise)
: process_noise_(process_noise), measurement_noise_(measurement_noise)
{
}

void ScalarKalmanFilter::reset()
{
  estimate_ = 0.0;
  covariance_ = 0.0;
  initialized_ = false;
}

bool ScalarKalmanFilter::initialized() const
{
  return initialized_;
}

double ScalarKalmanFilter::estimate() const
{
  return estimate_;
}

double ScalarKalmanFilter::update(double measurement)
{
  // 首个测量直接作为初始状态，避免从 0 开始的收敛抖动。
  if (!initialized_) {
    estimate_ = measurement;
    covariance_ = measurement_noise_;
    initialized_ = true;
    return estimate_;
  }

  // 预测：随机游走模型下状态不变，仅协方差随过程噪声增大。
  const double predicted_covariance = covariance_ + process_noise_;

  // 更新：按卡尔曼增益融合预测与测量。
  const double kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise_);
  estimate_ += kalman_gain * (measurement - estimate_);
  covariance_ = (1.0 - kalman_gain) * predicted_covariance;
  return estimate_;
}

CoordinateFilter::CoordinateFilter()
: filter_x_(kProcessNoise, kMeasurementNoise),
  filter_y_(kProcessNoise, kMeasurementNoise),
  filter_z_(kProcessNoise, kMeasurementNoise)
{
}

void CoordinateFilter::reset()
{
  filter_x_.reset();
  filter_y_.reset();
  filter_z_.reset();
}

bool CoordinateFilter::isOutlier(const PnpResult & measurement) const
{
  // 尚未初始化时无可比较的估计，不判定为极端值。
  if (!filter_x_.initialized()) {
    return false;
  }

  return std::abs(measurement.x - filter_x_.estimate()) > kOutlierThreshold ||
         std::abs(measurement.y - filter_y_.estimate()) > kOutlierThreshold ||
         std::abs(measurement.z - filter_z_.estimate()) > kOutlierThreshold;
}

std::optional<PnpResult> CoordinateFilter::update(const PnpResult & measurement)
{
  // 极端值不参与平滑，直接抛弃以免污染滤波状态。
  if (isOutlier(measurement)) {
    return std::nullopt;
  }

  PnpResult smoothed;
  smoothed.x = filter_x_.update(measurement.x);
  smoothed.y = filter_y_.update(measurement.y);
  smoothed.z = filter_z_.update(measurement.z);
  return smoothed;
}

}  // namespace arm
