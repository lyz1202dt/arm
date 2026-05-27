#include <arm/box_grid_detector.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace arm
{
namespace
{

// 计算检测框中心点的 X 坐标，用于左右排序。
float centerX(const Detection & detection)
{
  return static_cast<float>(detection.box.x) + static_cast<float>(detection.box.width) * 0.5F;
}

// 计算检测框中心点的 Y 坐标，用于上下分排。
float centerY(const Detection & detection)
{
  return static_cast<float>(detection.box.y) + static_cast<float>(detection.box.height) * 0.5F;
}

}  // namespace

BoxGridDetector::BoxGridDetector(std::shared_ptr<Inference> inference)
: inference_(std::move(inference))
{
  if (!inference_) {
    throw std::invalid_argument("BoxGridDetector 需要有效的 Inference 实例");
  }
}

std::optional<std::array<int32_t, 8>> BoxGridDetector::detectStableGrid(cv::VideoCapture & camera)
{
  std::optional<std::array<int32_t, 8>> last_grid;
  int same_count = 0;

  // 连续采样多帧，只有多次识别结果一致才认为矩阵稳定，降低抖动与误检影响。
  for (int attempt = 0; attempt < max_attempts_; ++attempt) {
    cv::Mat frame;
    if (!camera.read(frame) || frame.empty()) {
      continue;
    }

    const auto grid = detectGridOnce(frame);
    if (!grid) {
      last_grid.reset();
      same_count = 0;
      continue;
    }

    if (last_grid && *last_grid == *grid) {
      ++same_count;
    } else {
      last_grid = grid;
      same_count = 1;
    }

    if (same_count >= stable_count_) {
      return grid;
    }
  }

  return std::nullopt;
}

std::optional<std::array<int32_t, 8>> BoxGridDetector::detectGridOnce(const cv::Mat & frame) const
{
  if (frame.empty()) {
    return std::nullopt;
  }

  const auto rows = groupDetectionsByRows(inference_->run(frame));
  // 只接受完整的 2x4 识别结果；不足 8 个或分排异常时直接丢弃。
  if (rows.size() != 2 || rows[0].size() != 4 || rows[1].size() != 4) {
    return std::nullopt;
  }

  std::array<int32_t, 8> grid{};
  int index = 0;
  for (const auto & row : rows) {
    for (const auto & detection : row) {
      // 直接使用模型类别索引作为箱子编号，顺序保持左上到右下。
      grid[index++] = detection.class_id;
    }
  }

  return grid;
}

std::vector<std::vector<Detection>> BoxGridDetector::groupDetectionsByRows(
  const std::vector<Detection> & detections) const
{
  if (detections.empty()) {
    return {};
  }

  std::vector<Detection> sorted = detections;
  // 先按中心点 Y 排序，Y 非常接近时再按 X 排序，便于后续逐个归入行。
  std::sort(sorted.begin(), sorted.end(), [](const Detection & lhs, const Detection & rhs) {
    if (std::abs(centerY(lhs) - centerY(rhs)) < 1.0F) {
      return centerX(lhs) < centerX(rhs);
    }
    return centerY(lhs) < centerY(rhs);
  });

  std::vector<std::vector<Detection>> rows;
  for (const Detection & detection : sorted) {
    const float current_y = centerY(detection);
    bool assigned = false;

    for (auto & row : rows) {
      float average_y = 0.0F;
      float average_height = 0.0F;
      for (const auto & item : row) {
        average_y += centerY(item);
        average_height += static_cast<float>(item.box.height);
      }

      average_y /= static_cast<float>(row.size());
      average_height /= static_cast<float>(row.size());
      // 用该行平均框高的一半作为动态阈值，兼顾不同距离下的尺度变化。
      const float threshold = std::max(average_height * 0.5F, 25.0F);

      if (std::abs(current_y - average_y) <= threshold) {
        row.push_back(detection);
        assigned = true;
        break;
      }
    }

    if (!assigned) {
      rows.push_back({detection});
    }
  }

  // 行与行之间从上到下排序。
  std::sort(rows.begin(), rows.end(), [](const auto & lhs, const auto & rhs) {
    return centerY(lhs.front()) < centerY(rhs.front());
  });

  // 每一行内部从左到右排序，最终得到左上到右下的稳定遍历顺序。
  for (auto & row : rows) {
    std::sort(row.begin(), row.end(), [](const Detection & lhs, const Detection & rhs) {
      return centerX(lhs) < centerX(rhs);
    });
  }

  return rows;
}

}  // namespace arm
