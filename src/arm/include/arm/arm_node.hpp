#ifndef ARM_ARM_NODE_HPP_
#define ARM_ARM_NODE_HPP_

// 声明机械臂视觉 ROS2 主节点及其后台识别任务同步工具。

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <arm/box_grid_detector.hpp>
#include <arm/pnp_detector.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <semaphore.h>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

namespace arm
{

class CommandSemaphore
{
public:
  CommandSemaphore();
  ~CommandSemaphore();

  CommandSemaphore(const CommandSemaphore &) = delete;
  CommandSemaphore & operator=(const CommandSemaphore &) = delete;

  bool post() noexcept;
  bool wait() noexcept;

private:
  sem_t semaphore_{};
};

// ROS2 主节点：订阅控制命令并按需触发箱子矩阵识别或箱子位置识别。
class ArmNode : public rclcpp::Node
{
public:
  ArmNode();
  ~ArmNode() override;

private:
  enum class RecognitionTask
  {
    BoxGrid,
    Pnp,
  };

  struct PnpWindowStats
  {
    double mean_x{};
    double mean_y{};
    double mean_z{};
    double var_x{};
    double var_y{};
    double variance_sum{};
  };

  void onCommand(const std_msgs::msg::Int32::SharedPtr msg);
  rcl_interfaces::msg::SetParametersResult onParametersChanged(
    const std::vector<rclcpp::Parameter> & parameters);
  bool requestRecognition(RecognitionTask task, const char * source);
  bool cancelRecognition(const char * source);
  void visionWorker();
  void handleCommand();

  std::string cameraSourceLabel() const;
  bool openCamera();
  void releaseCamera();
  void destroyRecognitionUi();
  void cleanupRecognitionResources();
  bool readCameraFrame(cv::Mat & frame, const char * task_name);

  void runBoxGrid();
  bool publishGridIfRecognitionActive(const std::array<int32_t, 8> & grid);

  void runPnp();
  void resetPnpWindow();
  void appendPnpSample(const PnpResult & result);
  bool hasFullPnpWindow() const;
  PnpWindowStats computePnpWindowStats() const;
  bool publishPnpPoint(double x, double y, double z);
  bool publishPnpFailure();
  void updateVisionVarianceParameter(double variance_sum);
  void resetCancelRecognitionParameter();

  std::string camera_index_{0};
  std::string camera_device_;
  cv::VideoCapture camera_;
  std::unique_ptr<BoxGridDetector> box_grid_detector_;
  std::unique_ptr<PnpDetector> pnp_detector_;

  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr box_grid_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pnp_pub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr command_sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  CommandSemaphore command_signal_;
  std::atomic_bool worker_running_{true};
  std::atomic_bool recognition_keep_running_{false};
  std::atomic_bool vision_task_busy_{false};
  std::atomic_bool pnp_stop_requested_{false};
  std::mutex recognition_state_mutex_;
  std::thread vision_worker_;
  RecognitionTask pending_task_{RecognitionTask::BoxGrid};
  std::optional<RecognitionTask> active_task_;
  std::deque<PnpResult> pnp_window_;
  std::chrono::steady_clock::time_point last_variance_update_{};
};

}  // namespace arm

#endif  // ARM_ARM_NODE_HPP_
