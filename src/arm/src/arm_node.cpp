// 机械臂视觉 ROS2 主节点，负责相机初始化、命令分发和识别结果发布。
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <arm/box_grid_detector.hpp>
#include <arm/grip_detector.hpp>
#include <arm/inference.hpp>
#include <arm/pnp_detector.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robot_msgs/msg/box_id_grid.hpp>
#include <robot_msgs/msg/vis.hpp>
#include <std_msgs/msg/int32.hpp>

namespace
{
constexpr int kQueueSize = 50000;
constexpr int kDefaultCameraIndex = 0;
constexpr std::chrono::seconds kPnpTimeout{5};
constexpr double kPnpStableDistanceMeters = 0.009;
constexpr int kPnpStableHitsRequired = 5;
constexpr int kCameraReadMaxRetries = 10;
constexpr std::chrono::milliseconds kCameraReadRetryDelay{100};
}  // namespace

// ROS2 主节点：订阅控制命令并按需触发箱子矩阵识别、PnP 位姿识别和抓取判断。
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
    const std::string pnp_model_path = declare_parameter<std::string>(
      "pnp_model_path", "/home/pc2/Desktop/arm/src/arm/best.onnx");
    const bool name_cuda = declare_parameter<bool>("name_run_with_cuda", false);
    const bool pnp_cuda = declare_parameter<bool>("pnp_run_with_cuda", true);
    const double grip_threshold = declare_parameter<double>("grip_brightness_threshold", 60.0);

    openCamera();
    if (!camera_.isOpened()) {
      throw std::runtime_error("无法打开 USB 摄像头，索引: " + cameraSourceLabel());
    }

    auto name_inference = std::make_shared<arm::Inference>(name_model_path, cv::Size(640, 640), "", name_cuda);
    auto pnp_inference = std::make_shared<arm::Inference>(pnp_model_path, cv::Size(640, 640), "", pnp_cuda);

    box_grid_detector_ = std::make_unique<arm::BoxGridDetector>(name_inference);
    pnp_detector_ = std::make_unique<arm::PnpDetector>(pnp_inference);
    grip_detector_ = std::make_unique<arm::GripDetector>(grip_threshold);

    box_grid_pub_ = create_publisher<robot_msgs::msg::BoxIdGrid>("box_id_grid", kQueueSize);
    pnp_pub_ = create_publisher<robot_msgs::msg::Vis>("pnp_move", kQueueSize);
    grip_pub_ = create_publisher<robot_msgs::msg::Vis>("detect_result", kQueueSize);
    command_sub_ = create_subscription<std_msgs::msg::Int32>(
      "arm_command", kQueueSize, std::bind(&ArmNode::handleCommand, this, std::placeholders::_1));
    finish_scan_ = create_publisher<robot_msgs::msg::Vis>("scan_finish", kQueueSize);

    RCLCPP_INFO(get_logger(), "arm_node 已启动，等待 arm_command 控制命令");
  }

private:
  void handleCommand(const std_msgs::msg::Int32::SharedPtr msg)
  {
    // busy_ 用于避免多个命令在一次功能执行未结束时重入。
    if (busy_) {
      RCLCPP_WARN(get_logger(), "当前功能仍在执行，忽略命令: %d", msg->data);
      return;
    }

    busy_ = true;
    try {
      switch (msg->data) {
        case 0:
          // 退出命令只负责触发 ROS2 关闭，不再继续执行后续识别流程。
          RCLCPP_INFO(get_logger(), "收到退出命令");
          rclcpp::shutdown();
          break;
        case 1:
          runBoxGrid();
          break;
        case 2:
          runPnp();
          break;
        case 3:
          runGripDetect();
          break;
        default:
          RCLCPP_WARN(get_logger(), "未知命令: %d", msg->data);
          break;
      }
    } catch (const std::exception & exception) {
      RCLCPP_ERROR(get_logger(), "执行命令 %d 失败: %s", msg->data, exception.what());
      cv::destroyAllWindows();
      releaseCamera();
    }
    busy_ = false;
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
      robot_msgs::msg::Vis msg;
      // 额外发布 scan_finish，通知外部流程本次矩阵扫描已完成。
      msg.x = 1;
      finish_scan_->publish(msg);
      RCLCPP_INFO(get_logger(), "已发布 scan_finish");
      return;
    }

    robot_msgs::msg::BoxIdGrid message;
    for (std::size_t i = 0; i < grid->size(); ++i) {
      message.data[i] = (*grid)[i];
    }
    box_grid_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 box_id_grid");

    robot_msgs::msg::Vis msg;
    // 额外发布 scan_finish，通知外部流程本次矩阵扫描已完成。
    msg.x = 1;
    finish_scan_->publish(msg);
    RCLCPP_INFO(get_logger(), "已发布 scan_finish");
    cv::destroyAllWindows();
    releaseCamera();
  }

  void runPnp()
  {
    RCLCPP_INFO(get_logger(), "开始执行 PnP 位姿识别");

    const auto start_time = std::chrono::steady_clock::now();
    std::optional<arm::PnpResult> last_valid_result;
    int stable_hits = 0;

    while (std::chrono::steady_clock::now() - start_time < kPnpTimeout) {
      cv::Mat frame;
      if (!readCameraFrame(frame, "PnP")) {
        cv::destroyAllWindows();
        return;
      }

      const auto result = pnp_detector_->detectOnce(frame);
      if (!result) {
        continue;
      }

      if (!last_valid_result) {
        last_valid_result = result;
        stable_hits = 0;
        RCLCPP_INFO(
          get_logger(), "PnP 获得首个有效结果，等待后续稳定: x=%.4f y=%.4f z=%.4f",
          result->x, result->y, result->z);
        continue;
      }

      const double dx = result->x - last_valid_result->x;
      const double dy = result->y - last_valid_result->y;
      // 稳定判定只看水平平面上的 x/y 漂移，z 仍随最终结果一起发布。
      const double xy_distance = std::sqrt(dx * dx + dy * dy);
      last_valid_result = result;

      if (xy_distance <= kPnpStableDistanceMeters) {
        ++stable_hits;
        RCLCPP_INFO(
          get_logger(), "PnP 稳定命中 %d/%d，xy 误差=%.4f m",
          stable_hits, kPnpStableHitsRequired, xy_distance);
      } else {
        stable_hits = 0;
        RCLCPP_INFO(get_logger(), "PnP 结果抖动，重置稳定计数，xy 误差=%.4f m", xy_distance);
        continue;
      }

      if (stable_hits < kPnpStableHitsRequired) {
        continue;
      }

      robot_msgs::msg::Vis message;
      message.x = static_cast<float>(result->x);
      message.y = static_cast<float>(result->y);
      message.z = static_cast<float>(result->z);
      pnp_pub_->publish(message);
      RCLCPP_INFO(get_logger(), "已发布稳定 pnp_move: x=%.4f y=%.4f z=%.4f", result->x, result->y, result->z);
      cv::destroyAllWindows();
      releaseCamera();
      return;
    }

    RCLCPP_WARN(get_logger(), "PnP 位姿识别在 %.1f 秒内未达到稳定条件，已结束", kPnpTimeout.count() * 1.0);
    cv::destroyAllWindows();
    releaseCamera();
  }

  void runGripDetect()
  {
    RCLCPP_INFO(get_logger(), "开始判断抓取结果");
    cv::Mat frame;
    if (!readCameraFrame(frame, "抓取判断")) {
      cv::destroyWindow("...");
      return;
    }

    robot_msgs::msg::Vis message;
    // 当前仅复用 Vis 消息的 x 字段承载抓取判断结果。
    message.x = grip_detector_->detectOnce(frame);
    grip_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 detect_result: %.1f", message.x);
    cv::destroyWindow("...");
    releaseCamera();
  }

  bool busy_{false};
  int camera_index_{kDefaultCameraIndex};
  std::string camera_device_;
  cv::VideoCapture camera_;
  std::unique_ptr<arm::BoxGridDetector> box_grid_detector_;
  std::unique_ptr<arm::PnpDetector> pnp_detector_;
  std::unique_ptr<arm::GripDetector> grip_detector_;
  rclcpp::Publisher<robot_msgs::msg::BoxIdGrid>::SharedPtr box_grid_pub_;
  rclcpp::Publisher<robot_msgs::msg::Vis>::SharedPtr pnp_pub_;
  rclcpp::Publisher<robot_msgs::msg::Vis>::SharedPtr finish_scan_;
  rclcpp::Publisher<robot_msgs::msg::Vis>::SharedPtr grip_pub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr command_sub_;
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
