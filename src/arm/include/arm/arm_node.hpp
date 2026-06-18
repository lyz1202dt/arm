#ifndef ARM_ARM_NODE_HPP_
#define ARM_ARM_NODE_HPP_

// 声明机械臂视觉 ROS2 主节点及其后台识别任务同步工具。

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <arm/box_grid_detector.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <semaphore.h>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

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

// ROS2 主节点：订阅控制命令并按需触发箱子矩阵识别。
class ArmNode : public rclcpp::Node
{
public:
  ArmNode();
  ~ArmNode() override;

private:
  void onCommand(const std_msgs::msg::Int32::SharedPtr msg);
  void visionWorker();
  void handleCommand();

  std::string cameraSourceLabel() const;
  bool openCamera();
  void releaseCamera();
  bool readCameraFrame(cv::Mat & frame, const char * task_name);
  void runBoxGrid();

  int camera_index_{0};
  std::string camera_device_;
  cv::VideoCapture camera_;
  std::unique_ptr<BoxGridDetector> box_grid_detector_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr box_grid_pub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr command_sub_;
  
  CommandSemaphore command_signal_;
  std::atomic_bool worker_running_{true};
  std::atomic_bool vision_task_busy_{false};
  std::thread vision_worker_;
};

}  // namespace arm

#endif  // ARM_ARM_NODE_HPP_
