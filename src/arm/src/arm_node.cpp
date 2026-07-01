// 机械臂视觉 ROS2 主节点，负责相机初始化、命令分发和识别结果发布。
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <arm/arm_node.hpp>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <arm/inference.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>

namespace
{
namespace fs = std::filesystem;

// ROS2 发布/订阅队列深度，取较大值以容忍下游短暂处理不及时。
constexpr int kQueueSize = 50000;
// 默认相机索引，供未显式指定设备路径时回退使用。
const char * kDefaultCameraIndex = "0";
// 包内默认模型相对路径。
const char * kDefaultModelRelativePath = "best.onnx";
// 单次读帧失败后的最大重试次数，超过则放弃本次识别命令。
constexpr int kCameraReadMaxRetries = 10;
// 触发箱子矩阵识别的布尔参数名。
constexpr char kStartRecognitionParameter[] = "start_recognition";
// 触发/停止箱子位置 PnP 的布尔参数名。
constexpr char kStartPnpParameter[] = "start_pnp";
// 取消当前识别任务的布尔参数名。
constexpr char kCancelRecognitionParameter[] = "cancel_recognition";
// 对外暴露当前识别稳定度（方差和）的浮点参数名。
constexpr char kVisionVarianceParameter[] = "vision_variance";
// 两次读帧重试之间的间隔。
constexpr std::chrono::milliseconds kCameraReadRetryDelay{100};
// vision_variance 参数的最小更新间隔，避免过于频繁地写参数。
constexpr std::chrono::milliseconds kVisionVarianceUpdateInterval{500};
// PnP 稳定判定所用的滑动窗口帧数。
constexpr std::size_t kPnpWindowSize = 5;
// 归一化方差时分母附加的极小值，防止均值接近 0 时除零。
constexpr double kVarianceNormalizationEpsilon = 1e-6;
// 箱子位置 PnP 连续若干帧归一化方差和低于该阈值即视为稳定，自动发布并结束。
constexpr double kPnpStableVarianceThreshold = 0.01;
// 色块正方形 PnP 连续若干帧归一化方差和低于该阈值即视为稳定，自动发布并结束。
constexpr double kColorPnpStableVarianceThreshold = 0.01;

cv::String resolveModelPath(const std::string & configured_path)
{
  if (configured_path.empty()) {
    throw std::runtime_error("模型路径不能为空");
  }

  const fs::path path(configured_path);
  if (path.is_absolute()) {
    return configured_path;
  }

  const fs::path share_dir = ament_index_cpp::get_package_share_directory("arm");
  const fs::path direct_candidate = share_dir / path;
  if (fs::exists(direct_candidate)) {
    return direct_candidate.string();
  }

  const fs::path model_candidate = share_dir / "models" / path;
  if (fs::exists(model_candidate)) {
    return model_candidate.string();
  }

#ifdef ARM_SOURCE_DIR
  const fs::path source_candidate = fs::path(ARM_SOURCE_DIR) / path;
  if (fs::exists(source_candidate)) {
    return source_candidate.string();
  }
#endif

  throw std::runtime_error(
          "无法解析模型路径: " + configured_path +
          "，已尝试包资源目录中的相对路径");
}
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
  // sem_wait 可能被信号中断（EINTR），此时需重新等待而非视为失败。
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
  camera_index_ = declare_parameter<std::string>("camera_index", kDefaultCameraIndex);
  camera_device_ = declare_parameter<std::string>("camera_device", "");
  const std::string name_model_path = declare_parameter<std::string>(
    "name_model_path", kDefaultModelRelativePath);
  const std::string pnp_model_path = declare_parameter<std::string>(
    "pnp_model_path", kDefaultModelRelativePath);
  const bool name_cuda = declare_parameter<bool>("name_run_with_cuda", false);
  const bool pnp_cuda = declare_parameter<bool>("pnp_run_with_cuda", false);
  declare_parameter<bool>(kStartRecognitionParameter, false);
  declare_parameter<bool>(kStartPnpParameter, false);
  declare_parameter<bool>(kCancelRecognitionParameter, false);
  declare_parameter<double>(kVisionVarianceParameter, 0.0);

  const cv::String resolved_name_model_path = resolveModelPath(name_model_path);
  const cv::String resolved_pnp_model_path = resolveModelPath(pnp_model_path);

  openCamera();
  if (!camera_.isOpened()) {
    throw std::runtime_error("无法打开 USB 摄像头，索引: " + cameraSourceLabel());
  }

  auto name_inference = std::make_shared<Inference>(resolved_name_model_path, cv::Size(640, 640), "", name_cuda);
  // pnp 与 name 默认指向同一个 ONNX 模型，且两个检测器都在同一后台线程串行调用推理，
  // 因此模型路径与后端标志完全一致时复用同一实例，省去重复的磁盘加载、解析与内存占用。
  std::shared_ptr<Inference> pnp_inference =
    (resolved_pnp_model_path == resolved_name_model_path && pnp_cuda == name_cuda)
    ? name_inference
    : std::make_shared<Inference>(resolved_pnp_model_path, cv::Size(640, 640), "", pnp_cuda);

  box_grid_detector_ = std::make_unique<BoxGridDetector>(name_inference);
  pnp_detector_ = std::make_unique<PnpDetector>(pnp_inference);
  color_pnp_detector_ = std::make_unique<ColorPnpDetector>();

  box_grid_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("box_id_grid", kQueueSize);
  pnp_box_id_pub_ = create_publisher<std_msgs::msg::Int32>("pnp_box_index", kQueueSize);
  pnp_pub_ = create_publisher<geometry_msgs::msg::Point>("pnp_move", kQueueSize);
  color_pnp_pub_ = create_publisher<geometry_msgs::msg::Point>("color_pnp_move", kQueueSize);
  command_sub_ = create_subscription<std_msgs::msg::Int32>(
    "arm_command", kQueueSize, [this](const std_msgs::msg::Int32::SharedPtr msg) {
      onCommand(msg);
    });
  parameter_callback_handle_ = add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & parameters) {
      return onParametersChanged(parameters);
    });
  vision_worker_ = std::thread(&ArmNode::visionWorker, this);

  RCLCPP_INFO(
    get_logger(),
    "arm_node 已启动，等待 arm_command 控制命令、start_recognition/start_pnp 参数触发或 cancel_recognition 参数取消");
}

ArmNode::~ArmNode()
{
  worker_running_.store(false);
  recognition_keep_running_.store(false);
  command_signal_.post();
  if (vision_worker_.joinable()) {
    vision_worker_.join();
  }
  cleanupRecognitionResources();
}

void ArmNode::onCommand(const std_msgs::msg::Int32::SharedPtr msg)
{
  // arm_command 整数命令映射：0 退出，1 箱子矩阵，2 箱子位置 PnP，5 色块正方形 PnP。
  switch (msg->data) {
    case 0:
      RCLCPP_INFO(get_logger(), "收到退出命令");
      rclcpp::shutdown();
      return;
    case 1:
      requestRecognition(RecognitionTask::BoxGrid, "arm_command");
      return;
    case 2:
      requestRecognition(RecognitionTask::Pnp, "arm_command");
      return;
    case 5:
      requestRecognition(RecognitionTask::ColorPnp, "arm_command");
      return;
    default:
      RCLCPP_WARN(get_logger(), "未知命令: %d", msg->data);
      return;
  }
}

rcl_interfaces::msg::SetParametersResult ArmNode::onParametersChanged(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  bool start_recognition_requested = false;
  bool start_pnp_requested = false;
  bool stop_pnp_requested = false;
  bool cancel_requested = false;

  // 先汇总本批次参数变更的意图，再统一处理，避免同一回调内多次相互覆盖。
  for (const auto & parameter : parameters) {
    const std::string & parameter_name = parameter.get_name();
    if (parameter_name != kStartRecognitionParameter &&
      parameter_name != kStartPnpParameter &&
      parameter_name != kCancelRecognitionParameter)
    {
      continue;
    }

    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
      result.successful = false;
      result.reason = parameter_name + " 必须是 bool 类型";
      return result;
    }

    const bool value = parameter.as_bool();
    if (parameter_name == kStartRecognitionParameter) {
      start_recognition_requested = value;
    } else if (parameter_name == kStartPnpParameter) {
      // start_pnp 置真为触发，置假为停止当前 PnP 识别。
      if (value) {
        start_pnp_requested = true;
      } else {
        stop_pnp_requested = true;
      }
    } else if (parameter_name == kCancelRecognitionParameter) {
      cancel_requested = value;
    }
  }

  if (cancel_requested) {
    cancelRecognition("cancel_recognition 参数");
  }

  if (start_recognition_requested) {
    requestRecognition(RecognitionTask::BoxGrid, "start_recognition 参数");
  }

  if (start_pnp_requested) {
    requestRecognition(RecognitionTask::Pnp, "start_pnp 参数");
  }

  if (stop_pnp_requested) {
    std::lock_guard<std::mutex> lock(recognition_state_mutex_);
    // 仅当当前确实在执行 PnP 任务时才响应停止，避免误停其他任务。
    if (active_task_ && *active_task_ == RecognitionTask::Pnp) {
      recognition_keep_running_.store(false);
      destroyRecognitionUi();
      RCLCPP_INFO(get_logger(), "已请求停止当前箱子位置识别，来源: start_pnp=false");
    }
  }

  return result;
}

bool ArmNode::requestRecognition(RecognitionTask task, const char * source)
{
  std::lock_guard<std::mutex> lock(recognition_state_mutex_);
  if (vision_task_busy_.load()) {
    // PnP 任务正在运行时再次触发 PnP，视为“保持运行”而非重复启动。
    if (task == RecognitionTask::Pnp && active_task_ && *active_task_ == RecognitionTask::Pnp) {
      recognition_keep_running_.store(true);
      RCLCPP_INFO(get_logger(), "箱子位置识别继续运行，来源: %s", source);
      return true;
    }

    RCLCPP_WARN(get_logger(), "当前识别任务仍在执行，丢弃触发来源: %s", source);
    return false;
  }

  // 记录待执行任务并通过信号量唤醒后台 visionWorker 线程。
  pending_task_ = task;
  recognition_keep_running_.store(true);
  vision_task_busy_.store(true);
  if (!command_signal_.post()) {
    recognition_keep_running_.store(false);
    vision_task_busy_.store(false);
    RCLCPP_ERROR(get_logger(), "识别任务信号量通知失败，丢弃触发来源: %s", source);
    return false;
  }

  RCLCPP_INFO(get_logger(), "已触发识别任务，来源: %s", source);
  return true;
}

bool ArmNode::cancelRecognition(const char * source)
{
  std::lock_guard<std::mutex> lock(recognition_state_mutex_);
  recognition_keep_running_.store(false);
  destroyRecognitionUi();

  if (!vision_task_busy_.load()) {
    releaseCamera();
    RCLCPP_INFO(get_logger(), "当前没有正在执行的识别任务，取消来源: %s", source);
    return false;
  }

  RCLCPP_INFO(get_logger(), "已请求停止当前识别任务，来源: %s", source);
  return true;
}

void ArmNode::visionWorker()
{
  // 后台识别线程：阻塞等待信号量唤醒，逐个串行执行被触发的识别任务。
  while (worker_running_.load()) {
    if (!command_signal_.wait()) {
      RCLCPP_ERROR(get_logger(), "等待识别任务信号量失败");
      continue;
    }

    // 析构阶段会 post 信号量并置位退出标志，这里需再次检查以便及时收尾。
    if (!worker_running_.load()) {
      break;
    }

    {
      std::lock_guard<std::mutex> lock(recognition_state_mutex_);
      active_task_ = pending_task_;
    }

    handleCommand();

    // 任务结束后清空活动任务标记并复位忙状态，等待下一次触发。
    {
      std::lock_guard<std::mutex> lock(recognition_state_mutex_);
      active_task_.reset();
    }
    vision_task_busy_.store(false);
    resetCancelRecognitionParameter();
  }
}

void ArmNode::handleCommand()
{
  // 根据当前活动任务类型分派到对应的识别流程，任意异常都就地清理资源。
  try {
    if (active_task_ && *active_task_ == RecognitionTask::Pnp) {
      runPnp();
    } else if (active_task_ && *active_task_ == RecognitionTask::ColorPnp) {
      runColorPnp();
    } else {
      runBoxGrid();
    }
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "执行视觉识别任务失败: %s", exception.what());
    cleanupRecognitionResources();
  }
}

std::string ArmNode::cameraSourceLabel() const
{
  // 优先使用显式设备路径，否则回退到 by-id 索引，仅用于日志展示。
  if (!camera_device_.empty()) {
    return camera_device_;
  }
  return camera_index_;
}

bool ArmNode::openCamera()
{
  if (camera_.isOpened()) {
    return true;
  }

  // 优先以 V4L2 后端打开，可正确设置缓冲区等参数。
  if (!camera_device_.empty()) {
    camera_.open(camera_device_, cv::CAP_V4L2);
  } else {
    camera_.open(camera_index_, cv::CAP_V4L2);
  }

  if (camera_.isOpened()) {
    // 缓冲区设为 1，确保每次读到的是最新帧而非积压的旧帧。
    camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    return true;
  }

  // V4L2 打开失败时退回默认后端再试一次。
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

void ArmNode::destroyRecognitionUi()
{
  BoxGridDetector::destroyPreviewWindows();
  cv::destroyAllWindows();
}

void ArmNode::cleanupRecognitionResources()
{
  destroyRecognitionUi();
  releaseCamera();
}

bool ArmNode::readCameraFrame(cv::Mat & frame, const char * task_name)
{
  // 在最大重试次数内尝试读取一帧有效画面，期间若任务被取消则立即返回。
  for (int attempt = 1; attempt <= kCameraReadMaxRetries; ++attempt) {
    if (!worker_running_.load() || !recognition_keep_running_.load()) {
      RCLCPP_INFO(get_logger(), "%s 任务已取消，停止读取摄像头画面", task_name);
      return false;
    }

    frame.release();
    if (openCamera() && camera_.read(frame) && !frame.empty()) {
      return true;
    }

    // 读取失败可能是设备短暂掉线，释放后重开再试。
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
  // 先确认能取到画面，避免在设备异常时空跑稳定判定。
  cv::Mat frame;
  if (!readCameraFrame(frame, "箱子矩阵")) {
    cleanupRecognitionResources();
    return;
  }

  // 由检测器内部多帧投票得到稳定的 8 位编号序列。
  const auto grid = box_grid_detector_->detectStableGrid(camera_, recognition_keep_running_);
  if (!grid) {
    RCLCPP_INFO(get_logger(), "识别任务已停止");
    cleanupRecognitionResources();
    return;
  }

  if (!publishGridIfRecognitionActive(*grid)) {
    RCLCPP_INFO(get_logger(), "识别任务已停止，放弃发布 box_id_grid");
    cleanupRecognitionResources();
    return;
  }

  RCLCPP_INFO(get_logger(), "已发布 box_id_grid");
  cleanupRecognitionResources();
}

bool ArmNode::publishGridIfRecognitionActive(const std::array<int32_t, 8> & grid)
{
  std::lock_guard<std::mutex> lock(recognition_state_mutex_);
  // 投票过程中任务可能已被取消，发布前再次确认仍处于激活状态。
  if (!worker_running_.load() || !recognition_keep_running_.load()) {
    return false;
  }

  std_msgs::msg::Int32MultiArray message;
  message.data.resize(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    message.data[i] = grid[i];
  }
  box_grid_pub_->publish(message);
  return true;
}

void ArmNode::runPnp()
{
  RCLCPP_INFO(get_logger(), "开始识别箱子位置");
  resetPnpWindow();

  // 进入 PnP 位姿估计前，先发布一次 YOLO 识别到的箱子索引。
  if (!publishPnpBoxIndex()) {
    cleanupRecognitionResources();
    return;
  }

  last_variance_update_ = std::chrono::steady_clock::now();

  // 持续采样，待最近若干帧中心点坐标足够稳定后发布均值并结束。
  while (worker_running_.load() && recognition_keep_running_.load()) {
    cv::Mat frame;
    if (!readCameraFrame(frame, "箱子位置")) {
      break;
    }

    const auto result = pnp_detector_->detectOnce(frame);
    if (!result) {
      // 本帧未解算出位姿，清空窗口重新累积，并把方差置高表示当前不稳定。
      pnp_window_.clear();
      updateVisionVarianceParameter(10.0);
      continue;
    }

    appendPnpSample(*result);
    if (!hasFullPnpWindow()) {
      continue;
    }

    const PnpWindowStats stats = computePnpWindowStats();
    updateVisionVarianceParameter(stats.variance_sum);

    // 窗口内坐标足够稳定，发布一次中心点均值后结束本次任务。
    if (stats.variance_sum <= kPnpStableVarianceThreshold) {
      publishPnpPoint(stats.mean_x, stats.mean_y, stats.mean_z);
      RCLCPP_INFO(
        get_logger(), "箱子位置识别已稳定，发布中心点: x=%.4f y=%.4f z=%.4f",
        stats.mean_x, stats.mean_y, stats.mean_z);
      break;
    }
  }

  cleanupRecognitionResources();
  RCLCPP_INFO(get_logger(), "箱子位置识别已结束，继续等待下一次外部参数触发");
}

void ArmNode::runColorPnp()
{
  RCLCPP_INFO(get_logger(), "开始识别色块正方形位置");
  resetPnpWindow();
  last_variance_update_ = std::chrono::steady_clock::now();

  // 持续采样，待最近若干帧中心点坐标足够稳定后发布均值并结束。
  while (worker_running_.load() && recognition_keep_running_.load()) {
    cv::Mat frame;
    if (!readCameraFrame(frame, "色块正方形位置")) {
      break;
    }

    const auto result = color_pnp_detector_->detectOnce(frame);
    if (!result) {
      // 本帧未提取到完整四色块，清空窗口重新累积并标记为不稳定。
      pnp_window_.clear();
      updateVisionVarianceParameter(10.0);
      continue;
    }

    appendPnpSample(*result);
    if (!hasFullPnpWindow()) {
      continue;
    }

    const PnpWindowStats stats = computePnpWindowStats();
    updateVisionVarianceParameter(stats.variance_sum);

    // 窗口内坐标足够稳定，发布一次中心点均值后结束本次任务。
    if (stats.variance_sum <= kColorPnpStableVarianceThreshold) {
      publishColorPnpPoint(stats.mean_x, stats.mean_y, stats.mean_z);
      RCLCPP_INFO(
        get_logger(), "色块正方形识别已稳定，发布中心点: x=%.4f y=%.4f z=%.4f",
        stats.mean_x, stats.mean_y, stats.mean_z);
      break;
    }
  }

  cleanupRecognitionResources();
  RCLCPP_INFO(get_logger(), "色块正方形识别已结束，继续等待下一次外部命令触发");
}

bool ArmNode::publishColorPnpPoint(double x, double y, double z)
{
  geometry_msgs::msg::Point message;
  message.x = x;
  message.y = y;
  message.z = z;
  color_pnp_pub_->publish(message);
  return true;
}

bool ArmNode::publishPnpBoxIndex()
{
  while (worker_running_.load() && recognition_keep_running_.load()) {
    cv::Mat frame;
    if (!readCameraFrame(frame, "箱子索引")) {
      return false;
    }

    const auto box_index = pnp_detector_->detectBoxIndex(frame);
    if (!box_index) {
      // 当前帧未识别到箱子，继续读取下一帧重试。
      continue;
    }

    std_msgs::msg::Int32 message;
    message.data = *box_index;
    pnp_box_id_pub_->publish(message);
    RCLCPP_INFO(get_logger(), "已发布 pnp_box_index: %d", *box_index);
    return true;
  }

  RCLCPP_INFO(get_logger(), "识别任务已停止，放弃发布 pnp_box_index");
  return false;
}

void ArmNode::resetPnpWindow()
{
  pnp_window_.clear();
  try {
    set_parameter(rclcpp::Parameter(kVisionVarianceParameter, 0.0));
  } catch (const std::exception & exception) {
    RCLCPP_WARN(get_logger(), "重置 vision_variance 参数失败: %s", exception.what());
  }
}

void ArmNode::appendPnpSample(const PnpResult & result)
{
  // 维护固定长度的滑动窗口：新样本入队，超出窗口大小时弹出最旧样本。
  pnp_window_.push_back(result);
  if (pnp_window_.size() > kPnpWindowSize) {
    pnp_window_.pop_front();
  }
}

bool ArmNode::hasFullPnpWindow() const
{
  return pnp_window_.size() == kPnpWindowSize;
}

ArmNode::PnpWindowStats ArmNode::computePnpWindowStats() const
{
  PnpWindowStats stats;
  if (pnp_window_.empty()) {
    return stats;
  }

  // 第一遍累加求三轴均值。
  for (const auto & sample : pnp_window_) {
    stats.mean_x += sample.x;
    stats.mean_y += sample.y;
    stats.mean_z += sample.z;
  }

  const double sample_count = static_cast<double>(pnp_window_.size());
  stats.mean_x /= sample_count;
  stats.mean_y /= sample_count;
  stats.mean_z /= sample_count;

  // 第二遍累加各轴离差平方，得到方差。
  for (const auto & sample : pnp_window_) {
    const double dx = sample.x - stats.mean_x;
    const double dy = sample.y - stats.mean_y;
    const double dz = sample.z - stats.mean_z;
    stats.var_x += dx * dx;
    stats.var_y += dy * dy;
    stats.var_z += dz * dz;
  }

  stats.var_x /= sample_count;
  stats.var_y /= sample_count;
  stats.var_z /= sample_count;
  const double normalized_var_x = stats.var_x / (std::abs(stats.mean_x) + kVarianceNormalizationEpsilon);
  const double normalized_var_y = stats.var_y / (std::abs(stats.mean_y) + kVarianceNormalizationEpsilon);
  const double normalized_var_z = stats.var_z / (std::abs(stats.mean_z) + kVarianceNormalizationEpsilon);
  // XYZ 三轴同等重要，归一化方差直接求和作为整体稳定度衡量。
  stats.variance_sum = normalized_var_x + normalized_var_y + normalized_var_z;
  return stats;
}

bool ArmNode::publishPnpPoint(double x, double y, double z)
{
  geometry_msgs::msg::Point message;
  message.x = x;
  message.y = y;
  message.z = z;
  pnp_pub_->publish(message);
  return true;
}

void ArmNode::updateVisionVarianceParameter(double variance_sum)
{
  // 限流：距上次更新不足设定间隔则跳过，避免频繁写参数。
  const auto now = std::chrono::steady_clock::now();
  if (now - last_variance_update_ < kVisionVarianceUpdateInterval) {
    return;
  }

  // 窗口未满时尚无可信方差，统一上报一个较大值表示当前不稳定。
  const bool has_valid_window = hasFullPnpWindow();
  const double value_to_publish = has_valid_window ? variance_sum : 10.0;

  try {
    set_parameter(rclcpp::Parameter(kVisionVarianceParameter, value_to_publish));
    last_variance_update_ = now;
  } catch (const std::exception & exception) {
    RCLCPP_WARN(get_logger(), "更新 vision_variance 参数失败: %s", exception.what());
  }
}

void ArmNode::resetCancelRecognitionParameter()
{
  // 每次任务结束后把取消参数复位为 false，便于下一轮重新触发取消。
  try {
    set_parameter(rclcpp::Parameter(kCancelRecognitionParameter, false));
  } catch (const std::exception & exception) {
    RCLCPP_WARN(get_logger(), "重置 cancel_recognition 参数失败: %s", exception.what());
  }
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
