#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <arm/box_grid_detector.hpp>
#include <arm/grip_detector.hpp>
#include <arm/inference.hpp>
#include <arm/pnp_detector.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robot_interfaces/msg/box_id_grid.hpp>
#include <robot_interfaces/msg/vis.hpp>
#include <std_msgs/msg/int32.hpp>

namespace
{
constexpr int kQueueSize = 10;
}  // namespace

// ROS2 主节点：订阅控制命令并按需触发箱子矩阵识别、PnP 位姿识别和抓取判断。
class ArmNode : public rclcpp::Node
{
public:
  ArmNode()
  : Node("arm_node")
  {
    const int camera_index = declare_parameter<int>("camera_index", 2);
    // 这里的模型路径只是当前环境下的默认值，部署到其他机器时应优先通过参数覆写。
    const std::string name_model_path = declare_parameter<std::string>(
      "name_model_path", "/home/yuan/Vscode_word/awork_mycode/arm/src/arm/best.onnx");
    const std::string pnp_model_path = declare_parameter<std::string>(
      "pnp_model_path", "/home/yuan/Vscode_word/awork_mycode/arm/src/arm/best.onnx");
    const bool name_cuda = declare_parameter<bool>("name_run_with_cuda", false);
    const bool pnp_cuda = declare_parameter<bool>("pnp_run_with_cuda", true);
    const double grip_threshold = declare_parameter<double>("grip_brightness_threshold", 60.0);

    camera_.open(camera_index);
    if (!camera_.isOpened()) {
      throw std::runtime_error("无法打开 USB 摄像头，索引: " + std::to_string(camera_index));
    }

    auto name_inference = std::make_shared<arm::Inference>(name_model_path, cv::Size(640, 640), "", name_cuda);
    auto pnp_inference = std::make_shared<arm::Inference>(pnp_model_path, cv::Size(640, 640), "", pnp_cuda);

    box_grid_detector_ = std::make_unique<arm::BoxGridDetector>(name_inference);
    pnp_detector_ = std::make_unique<arm::PnpDetector>(pnp_inference);
    grip_detector_ = std::make_unique<arm::GripDetector>(grip_threshold);

    box_grid_pub_ = create_publisher<robot_interfaces::msg::BoxIdGrid>("box_id_grid", kQueueSize);
    pnp_pub_ = create_publisher<robot_interfaces::msg::Vis>("pnp_move", kQueueSize);
    grip_pub_ = create_publisher<robot_interfaces::msg::Vis>("detect_result", kQueueSize);
    command_sub_ = create_subscription<std_msgs::msg::Int32>(
      "arm_command", kQueueSize, std::bind(&ArmNode::handleCommand, this, std::placeholders::_1));
    finish_scan_ = create_publisher<robot_interfaces::msg::Vis>("scan_finish", kQueueSize);

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
    }
    busy_ = false;
  }

  void runBoxGrid()
  {
    RCLCPP_INFO(get_logger(), "开始识别箱子矩阵");
    const auto grid = box_grid_detector_->detectStableGrid(camera_);
    if (!grid) {
      RCLCPP_WARN(get_logger(), "未获得稳定的 2x4 箱子矩阵");
      return;
    }

    robot_interfaces::msg::BoxIdGrid message;
    for (std::size_t i = 0; i < grid->size(); ++i) {
      message.data[i] = (*grid)[i];
    }
    box_grid_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 box_id_grid");

    robot_interfaces::msg::Vis msg;
    // 额外发布 scan_finish，通知外部流程本次矩阵扫描已完成。
    msg.x = 1;
    finish_scan_->publish(msg);
    RCLCPP_INFO(get_logger(), "已发布 scan_finish");
  }

  void runPnp()
  {
    RCLCPP_INFO(get_logger(), "开始执行 PnP 位姿识别");
    cv::Mat frame;
    if (!camera_.read(frame) || frame.empty()) {
      RCLCPP_WARN(get_logger(), "读取摄像头画面失败");
      return;
    }

    const auto result = pnp_detector_->detectOnce(frame);
    if (!result) {
      RCLCPP_WARN(get_logger(), "PnP 位姿识别失败");
      return;
    }

    robot_interfaces::msg::Vis message;
    message.x = static_cast<float>(result->x);
    message.y = static_cast<float>(result->y);
    message.z = static_cast<float>(result->z);
    pnp_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 pnp_move: x=%.4f y=%.4f z=%.4f", result->x, result->y, result->z);
  }

  void runGripDetect()
  {
    RCLCPP_INFO(get_logger(), "开始判断抓取结果");
    cv::Mat frame;
    if (!camera_.read(frame) || frame.empty()) {
      RCLCPP_WARN(get_logger(), "读取摄像头画面失败");
      return;
    }

    robot_interfaces::msg::Vis message;
    // 当前仅复用 Vis 消息的 x 字段承载抓取判断结果。
    message.x = grip_detector_->detectOnce(frame);
    grip_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 detect_result: %.1f", message.x);
  }

  bool busy_{false};
  cv::VideoCapture camera_;
  std::unique_ptr<arm::BoxGridDetector> box_grid_detector_;
  std::unique_ptr<arm::PnpDetector> pnp_detector_;
  std::unique_ptr<arm::GripDetector> grip_detector_;
  rclcpp::Publisher<robot_interfaces::msg::BoxIdGrid>::SharedPtr box_grid_pub_;
  rclcpp::Publisher<robot_interfaces::msg::Vis>::SharedPtr pnp_pub_;
  rclcpp::Publisher<robot_interfaces::msg::Vis>::SharedPtr finish_scan_;
  rclcpp::Publisher<robot_interfaces::msg::Vis>::SharedPtr grip_pub_;
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
