// 机械臂视觉 ROS2 主节点，负责相机初始化、命令分发和箱子矩阵识别结果发布。
#include <atomic>
#include <cerrno>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#include <arm/box_grid_detector.hpp>
#include <arm/inference.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <semaphore.h>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

namespace
{
constexpr int kQueueSize = 50000;
constexpr int kDefaultCameraIndex = 0;
constexpr int kCameraReadMaxRetries = 10;
constexpr std::chrono::milliseconds kCameraReadRetryDelay{100};

class CommandSemaphore
{
public:
  CommandSemaphore()
  {
    if (sem_init(&semaphore_, 0, 0) != 0) {
      throw std::system_error(errno, std::generic_category(), "sem_init");
    }
  }

  ~CommandSemaphore()
  {
    sem_destroy(&semaphore_);
  }

  CommandSemaphore(const CommandSemaphore &) = delete;
  CommandSemaphore & operator=(const CommandSemaphore &) = delete;

  bool post() noexcept
  {
    return sem_post(&semaphore_) == 0;
  }

  bool wait() noexcept
  {
    while (sem_wait(&semaphore_) != 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    return true;
  }

private:
  sem_t semaphore_{};
};
}  // namespace

// ROS2 主节点：订阅控制命令并按需触发箱子矩阵识别。
class ArmNode : public rclcpp::Node
{
public:
  ArmNode()
  : Node("arm_node")
  {
    camera_index_ = declare_parameter<int>("camera_index", kDefaultCameraIndex);
    camera_device_ = declare_parameter<std::string>("camera_device", "");
    // 这里的模型路径只是当前环境下的默认值，部署到其他机器时应优先通过参数覆写。
    const std::string name_model_path = declare_parameter<std::string>(
      "name_model_path", "/home/pc2/Desktop/arm/src/arm/best.onnx");
    const bool name_cuda = declare_parameter<bool>("name_run_with_cuda", false);

    openCamera();
    if (!camera_.isOpened()) {
      throw std::runtime_error("无法打开 USB 摄像头，索引: " + cameraSourceLabel());
    }

    auto name_inference = std::make_shared<arm::Inference>(name_model_path, cv::Size(640, 640), "", name_cuda);

    box_grid_detector_ = std::make_unique<arm::BoxGridDetector>(name_inference);

    box_grid_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("box_id_grid", kQueueSize);
    command_sub_ = create_subscription<std_msgs::msg::Int32>(
      "arm_command", kQueueSize, [this](const std_msgs::msg::Int32::SharedPtr msg) {
        enqueueCommand(msg);
      });
    command_worker_ = std::thread(&ArmNode::commandWorker, this);

    RCLCPP_INFO(get_logger(), "arm_node 已启动，等待 arm_command 控制命令");
  }

  ~ArmNode() override
  {
    command_worker_running_.store(false);
    command_signal_.post();
    if (command_worker_.joinable()) {
      command_worker_.join();
    }
  }

private:
  void enqueueCommand(const std_msgs::msg::Int32::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      command_queue_.push(msg->data);
    }

    if (!command_signal_.post()) {
      RCLCPP_ERROR(get_logger(), "命令信号量通知失败，命令可能无法及时执行: %d", msg->data);
    }
  }

  void commandWorker()
  {
    while (command_worker_running_.load()) {
      if (!command_signal_.wait()) {
        RCLCPP_ERROR(get_logger(), "等待命令信号量失败");
        continue;
      }

      if (!command_worker_running_.load()) {
        break;
      }

      const auto command = takeCommand();
      if (!command) {
        continue;
      }

      handleCommand(*command);
    }
  }

  std::optional<int> takeCommand()
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (command_queue_.empty()) {
      return std::nullopt;
    }

    const int command = command_queue_.front();
    command_queue_.pop();
    return command;
  }

  void handleCommand(const int command)
  {
    try {
      switch (command) {
        case 0:
          // 退出命令只负责触发 ROS2 关闭，不再继续执行后续识别流程。
          RCLCPP_INFO(get_logger(), "收到退出命令");
          rclcpp::shutdown();
          break;
        case 1:
          runBoxGrid();
          break;
        default:
          RCLCPP_WARN(get_logger(), "未知命令: %d", command);
          break;
      }
    } catch (const std::exception & exception) {
      RCLCPP_ERROR(get_logger(), "执行命令 %d 失败: %s", command, exception.what());
      cv::destroyAllWindows();
      releaseCamera();
    }
  }

  std::string cameraSourceLabel() const
  {
    if (!camera_device_.empty()) {
      return camera_device_;
    }
    return std::to_string(camera_index_);
  }

  bool openCamera()
  {
    if (camera_.isOpened()) {
      return true;
    }

    if (!camera_device_.empty()) {
      camera_.open(camera_device_, cv::CAP_V4L2);
    } else {
      camera_.open(camera_index_, cv::CAP_V4L2);
    }

    if (camera_.isOpened()) {
      camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
      return true;
    }

    if (!camera_device_.empty()) {
      camera_.open(camera_device_);
    } else {
      camera_.open(camera_index_);
    }

    if (camera_.isOpened()) {
      camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    return camera_.isOpened();
  }

  void releaseCamera()
  {
    if (camera_.isOpened()) {
      camera_.release();
    }
  }

  bool readCameraFrame(cv::Mat & frame, const char * task_name)
  {
    for (int attempt = 1; attempt <= kCameraReadMaxRetries; ++attempt) {
      frame.release();
      if (openCamera() && camera_.read(frame) && !frame.empty()) {
        return true;
      }

      RCLCPP_WARN(
        get_logger(), "%s 读取摄像头画面失败，正在重新打开摄像头后重试 %d/%d，索引: %s",
        task_name, attempt, kCameraReadMaxRetries, cameraSourceLabel().c_str());
      releaseCamera();
      std::this_thread::sleep_for(kCameraReadRetryDelay);
    }

    RCLCPP_ERROR(
      get_logger(), "%s 连续 %d 次无法读取有效摄像头画面，结束本次命令",
      task_name, kCameraReadMaxRetries);
    return false;
  }

  void runBoxGrid()
  {
    RCLCPP_INFO(get_logger(), "开始识别箱子矩阵");
    cv::Mat frame;
    if (!readCameraFrame(frame, "箱子矩阵")) {
      cv::destroyAllWindows();
      return;
    }

    const auto grid = box_grid_detector_->detectStableGrid(camera_);
    if (!grid) {
      RCLCPP_WARN(get_logger(), "未获得稳定的 8 个箱子识别结果");
      cv::destroyAllWindows();
      releaseCamera();
      return;
    }

    std_msgs::msg::Float64MultiArray message;
    message.data.resize(grid->size());
    for (std::size_t i = 0; i < grid->size(); ++i) {
      message.data[i] = static_cast<double>((*grid)[i]);
    }
    box_grid_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 box_id_grid");

    cv::destroyAllWindows();
    releaseCamera();
  }

  int camera_index_{kDefaultCameraIndex};
  std::string camera_device_;
  cv::VideoCapture camera_;
  std::unique_ptr<arm::BoxGridDetector> box_grid_detector_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr box_grid_pub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr command_sub_;
  CommandSemaphore command_signal_;
  std::atomic_bool command_worker_running_{true};
  std::mutex command_mutex_;
  std::queue<int> command_queue_;
  std::thread command_worker_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<ArmNode>());
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(rclcpp::get_logger("arm_node_main"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
