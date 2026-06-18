// 机械臂视觉 ROS2 主节点，负责相机初始化、命令分发和箱子矩阵识别结果发布。
#include <arm/arm_node.hpp>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#include <arm/inference.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>

namespace
{
constexpr int kQueueSize = 50000;
constexpr int kDefaultCameraIndex = 0;
constexpr int kCameraReadMaxRetries = 10;
constexpr std::chrono::milliseconds kCameraReadRetryDelay{100};
}  // namespace

namespace arm
{

CommandSemaphore::CommandSemaphore()
{
  if (sem_init(&semaphore_, 0, 0) != 0) {
    throw std::system_error(errno, std::generic_category(), "sem_init");
  }
}

CommandSemaphore::~CommandSemaphore()
{
  sem_destroy(&semaphore_);
}

bool CommandSemaphore::post() noexcept
{
  return sem_post(&semaphore_) == 0;
}

bool CommandSemaphore::wait() noexcept
{
  while (sem_wait(&semaphore_) != 0) {
    if (errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

ArmNode::ArmNode()
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

  auto name_inference = std::make_shared<Inference>(name_model_path, cv::Size(640, 640), "", name_cuda);

  box_grid_detector_ = std::make_unique<BoxGridDetector>(name_inference);

  box_grid_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("box_id_grid", kQueueSize);
  command_sub_ = create_subscription<std_msgs::msg::Int32>(
    "arm_command", kQueueSize, [this](const std_msgs::msg::Int32::SharedPtr msg) {
      onCommand(msg);
    });
  vision_worker_ = std::thread(&ArmNode::visionWorker, this);

  RCLCPP_INFO(get_logger(), "arm_node 已启动，等待 arm_command 控制命令");
}

ArmNode::~ArmNode()
{
  worker_running_.store(false);
  command_signal_.post();
  if (vision_worker_.joinable()) {
    vision_worker_.join();
  }
}

void ArmNode::onCommand(const std_msgs::msg::Int32::SharedPtr msg)
{
  switch (msg->data) {
    case 0:
      // 退出命令不进入识别线程，避免被正在执行的识别任务阻塞。
      RCLCPP_INFO(get_logger(), "收到退出命令");
      rclcpp::shutdown();
      return;
    case 1:
      break;
    default:
      RCLCPP_WARN(get_logger(), "未知命令: %d", msg->data);
      return;
  }

  if (vision_task_busy_.exchange(true)) {
    RCLCPP_WARN(get_logger(), "当前识别任务仍在执行，丢弃命令: %d", msg->data);
    return;
  }

  if (!command_signal_.post()) {
    vision_task_busy_.store(false);
    RCLCPP_ERROR(get_logger(), "识别任务信号量通知失败，丢弃命令: %d", msg->data);
  }
}

void ArmNode::visionWorker()
{
  while (worker_running_.load()) {
    if (!command_signal_.wait()) {
      RCLCPP_ERROR(get_logger(), "等待识别任务信号量失败");
      continue;
    }

    if (!worker_running_.load()) {
      break;
    }

    handleCommand();
    vision_task_busy_.store(false);
  }
}

void ArmNode::handleCommand()
{
  try {
    runBoxGrid();
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "执行视觉识别任务失败: %s", exception.what());
    cv::destroyAllWindows();
    releaseCamera();
  }
}

std::string ArmNode::cameraSourceLabel() const
{
  if (!camera_device_.empty()) {
    return camera_device_;
  }
  return std::to_string(camera_index_);
}

bool ArmNode::openCamera()
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

void ArmNode::releaseCamera()
{
  if (camera_.isOpened()) {
    camera_.release();
  }
}

bool ArmNode::readCameraFrame(cv::Mat & frame, const char * task_name)
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

void ArmNode::runBoxGrid()
{
  RCLCPP_INFO(get_logger(), "开始识别箱子矩阵");
  cv::Mat frame;
  if (!readCameraFrame(frame, "箱子矩阵")) {
    cv::destroyAllWindows();
    return;
  }

  const auto grid = box_grid_detector_->detectStableGrid(camera_, worker_running_);
  if (!grid) {
    RCLCPP_INFO(get_logger(), "识别任务已停止");
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

}  // namespace arm

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<arm::ArmNode>());
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
