// 实现基于目标检测结果的箱子编号排序与多帧稳定判定逻辑。
#include <arm/box_grid_detector.hpp>

#include <algorithm>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace arm
{
namespace
{
constexpr std::size_t kGridSize = 8;
constexpr std::size_t kVoteBufferSize = 7;
constexpr std::chrono::milliseconds kFrameReadRetryDelay{50};

using GridResult = std::array<int32_t, kGridSize>;
using GridBuffer = std::array<std::optional<GridResult>, kVoteBufferSize>;

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

std::string joinIds(const std::vector<int32_t> & ids)
{
  std::ostringstream stream;
  for (const int32_t id : ids) {
    stream << id;
  }
  return stream.str();
}

std::string joinIds(const std::array<int32_t, 8> & ids)
{
  std::ostringstream stream;
  for (const int32_t id : ids) {
    stream << id;
  }
  return stream.str();
}

bool allSlotsValid(const GridBuffer & buffer)
{
  return std::all_of(buffer.begin(), buffer.end(), [](const auto & result) {
    return result.has_value();
  });
}

GridResult selectMajorityGrid(const GridBuffer & buffer, const std::size_t latest_slot)
{
  GridResult result{};

  for (std::size_t grid_index = 0; grid_index < result.size(); ++grid_index) {
    int32_t best_value = (*buffer[latest_slot])[grid_index];
    int best_count = 0;

    for (std::size_t age = 0; age < buffer.size(); ++age) {
      const std::size_t candidate_slot =
        (latest_slot + buffer.size() - age) % buffer.size();
      const int32_t candidate_value = (*buffer[candidate_slot])[grid_index];
      int candidate_count = 0;

      for (const auto & frame_result : buffer) {
        if ((*frame_result)[grid_index] == candidate_value) {
          ++candidate_count;
        }
      }

      if (candidate_count > best_count) {
        best_value = candidate_value;
        best_count = candidate_count;
      }
    }

    result[grid_index] = best_value;
  }

  return result;
}

}  // namespace

BoxGridDetector::BoxGridDetector(std::shared_ptr<Inference> inference)
: inference_(std::move(inference))
{
  if (!inference_) {
    throw std::invalid_argument("BoxGridDetector 需要有效的 Inference 实例");
  }
}

std::optional<std::array<int32_t, 8>> BoxGridDetector::detectStableGrid(
  cv::VideoCapture & camera, const std::atomic_bool & keep_running)
{
  GridBuffer recent_grids{};
  std::size_t next_slot = 0;

  // 持续采样最新图像，最近 7 帧都得到 8 个编号后，对每个位置做多数投票。
  while (keep_running.load()) {
    cv::Mat frame;
    if (!camera.read(frame) || frame.empty()) {
      std::this_thread::sleep_for(kFrameReadRetryDelay);
      continue;
    }

    cv::imshow("...", frame);
    cv::waitKey(1);

    const auto current_ids = collectOrderedIds(frame);
    const std::size_t current_slot = next_slot;
    recent_grids[current_slot].reset();
    next_slot = (next_slot + 1) % recent_grids.size();

    if (current_ids.empty()) {
      continue;
    }

    std::cout << "本次识别结果: " << joinIds(current_ids) << std::endl;

    if (current_ids.size() == kGridSize) {
      GridResult grid{};
      std::copy(current_ids.begin(), current_ids.end(), grid.begin());
      recent_grids[current_slot] = grid;
    }

    if (allSlotsValid(recent_grids)) {
      const auto majority_grid = selectMajorityGrid(recent_grids, current_slot);
      std::cout << "多数投票识别结果: " << joinIds(majority_grid) << std::endl;
      return majority_grid;
    }
  }

  return std::nullopt;
}

std::vector<int32_t> BoxGridDetector::collectOrderedIds(const cv::Mat & frame) const
{
  if (frame.empty()) {
    return {};
  }

  const auto rows = groupDetectionsByRows(inference_->run(frame));
  std::vector<int32_t> ids;
  ids.reserve(8);

  for (const auto & row : rows) {
    for (const auto & detection : row) {
      // 直接使用模型类别索引作为箱子编号，顺序保持左上到右下。
      ids.push_back(detection.class_id);
    }
  }

  return ids;
}

std::optional<std::array<int32_t, 8>> BoxGridDetector::detectGridOnce(const cv::Mat & frame) const
{
  const auto ids = collectOrderedIds(frame);
  if (ids.size() != 8) {
    return std::nullopt;
  }

  std::array<int32_t, 8> grid{};
  std::copy(ids.begin(), ids.end(), grid.begin());
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
      // 当前检测框与已有行都不匹配时，认为它是新的一排。
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
