#ifndef ARM_COORDINATE_FILTER_HPP_
#define ARM_COORDINATE_FILTER_HPP_

// 声明对 PnP 三维坐标进行卡尔曼平滑与离群值剔除的滤波器。

#include <optional>

#include <arm/pnp_detector.hpp>

namespace arm
{

// 单轴一维卡尔曼滤波器，采用随机游走（常量位置）模型对标量序列做平滑。
class ScalarKalmanFilter
{
public:
  // process_noise 为过程噪声 Q，measurement_noise 为测量噪声 R。
  ScalarKalmanFilter(double process_noise, double measurement_noise);

  // 清除内部状态，下一次 update 会以该测量值重新初始化。
  void reset();
  // 是否已由首个测量完成初始化。
  bool initialized() const;
  // 当前滤波估计值；未初始化时返回 0。
  double estimate() const;
  // 融合一个新测量并返回更新后的估计值。
  double update(double measurement);

private:
  double process_noise_;
  double measurement_noise_;
  double estimate_{0.0};
  double covariance_{0.0};
  bool initialized_{false};
};

// 对三维坐标的 X、Y、Z 三轴分别做卡尔曼平滑，并在融合前剔除极端值。
class CoordinateFilter
{
public:
  CoordinateFilter();

  // 重置三轴滤波器，用于每次识别任务开始时清空历史状态。
  void reset();
  // 融合一帧坐标测量；若判定为极端值则返回 std::nullopt 表示应当抛弃，
  // 否则返回平滑后的坐标。
  std::optional<PnpResult> update(const PnpResult & measurement);

private:
  // 判断本帧测量相对当前估计是否为极端值（任一轴偏差超阈值即抛弃）。
  bool isOutlier(const PnpResult & measurement) const;

  ScalarKalmanFilter filter_x_;
  ScalarKalmanFilter filter_y_;
  ScalarKalmanFilter filter_z_;
};

}  // namespace arm

#endif  // ARM_COORDINATE_FILTER_HPP_
