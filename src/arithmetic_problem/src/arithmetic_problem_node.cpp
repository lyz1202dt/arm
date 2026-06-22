#include "arithmetic_problem/arithmetic_problem_node.hpp"

#include "camera_driver.h"

#include <opencv2/highgui.hpp>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

constexpr int kQueueSize = 10;
constexpr char kStartCalcParameter[] = "start_calc";
constexpr char kShowImageParameter[] = "show_image";
constexpr char kPreviewWindowName[] = "Arithmetic Camera Preview";

std::string getEnvOrDefault(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return value;
}

int positiveIntOrDefault(int value, int fallback) {
    return value > 0 ? value : fallback;
}

}  // namespace

namespace arithmetic_problem {

CommandSemaphore::CommandSemaphore() {
    if (sem_init(&semaphore_, 0, 0) != 0) {
        throw std::system_error(errno, std::generic_category(), "sem_init");
    }
}

CommandSemaphore::~CommandSemaphore() {
    sem_destroy(&semaphore_);
}

bool CommandSemaphore::post() noexcept {
    return sem_post(&semaphore_) == 0;
}

bool CommandSemaphore::wait() noexcept {
    while (sem_wait(&semaphore_) != 0) {
        if (errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

ArithmeticProblemNode::ArithmeticProblemNode()
: Node("arithmetic_problem_node") {
    const std::string package_source_dir = ARITHMETIC_PROBLEM_SOURCE_DIR;

    declare_parameter<std::string>(
        "camera_config_path",
        getEnvOrDefault(
            "ARITHMETIC_CAMERA_CONFIG",
            package_source_dir + "/camera_driver/camera_init/HIKcamera0.yaml"));
    declare_parameter<std::string>(
        "onnx_model_path",
        getEnvOrDefault("ARITHMETIC_ONNX_MODEL", package_source_dir + "/src/best.onnx"));
    declare_parameter<int>("model_input_width", 640);
    declare_parameter<int>("model_input_height", 640);
    declare_parameter<bool>("run_with_cuda", true);
    declare_parameter<bool>("show_window", true);
    declare_parameter<int>("min_samples", 20);
    declare_parameter<int>("max_samples", 100);
    declare_parameter<double>("dominance_threshold", 0.80);
    declare_parameter<int>("timeout_ms", 5000);
    declare_parameter<bool>(kStartCalcParameter, false);
    declare_parameter<bool>(kShowImageParameter, false);

    vip_box_pub_ = create_publisher<std_msgs::msg::Int32>(
        "vip_box_id", rclcpp::QoS(1).reliable().transient_local());

    parameter_callback_handle_ = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
            return onParametersChanged(parameters);
        });

    calc_worker_ = std::thread(&ArithmeticProblemNode::calculationWorker, this);
    preview_worker_ = std::thread(&ArithmeticProblemNode::previewWorker, this);

    RCLCPP_INFO(
        get_logger(),
        "arithmetic_problem_node 已启动，等待 start_calc 参数服务触发计算任务");
}

ArithmeticProblemNode::~ArithmeticProblemNode() {
    worker_running_.store(false);
    preview_enabled_.store(false);
    calc_signal_.post();
    if (calc_worker_.joinable()) {
        calc_worker_.join();
    }
    if (preview_worker_.joinable()) {
        preview_worker_.join();
    }
}

rcl_interfaces::msg::SetParametersResult ArithmeticProblemNode::onParametersChanged(
    const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    bool start_requested = false;
    bool preview_enabled = preview_enabled_.load();
    bool preview_param_updated = false;
    for (const auto& parameter : parameters) {
        if (parameter.get_name() == kStartCalcParameter) {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "start_calc 必须是 bool 类型";
                return result;
            }

            if (parameter.as_bool()) {
                start_requested = true;
            }
            continue;
        }

        if (parameter.get_name() == kShowImageParameter) {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                result.successful = false;
                result.reason = "show_image 必须是 bool 类型";
                return result;
            }

            preview_enabled = parameter.as_bool();
            preview_param_updated = true;
        }
    }

    if (preview_param_updated) {
        updatePreviewState(preview_enabled);
    }

    if (start_requested) {
        requestCalculation("start_calc 参数");
    }

    return result;
}

bool ArithmeticProblemNode::requestCalculation(const char* source) {
    std::lock_guard<std::mutex> lock(calc_state_mutex_);
    if (calc_task_busy_.exchange(true)) {
        RCLCPP_WARN(get_logger(), "当前计算任务仍在执行，丢弃触发来源: %s", source);
        return false;
    }

    preview_needs_restart_.store(preview_enabled_.load());
    preview_enabled_.store(false);

    if (!calc_signal_.post()) {
        calc_task_busy_.store(false);
        preview_enabled_.store(preview_needs_restart_.load());
        RCLCPP_ERROR(get_logger(), "计算任务信号量通知失败，丢弃触发来源: %s", source);
        return false;
    }

    RCLCPP_INFO(get_logger(), "已触发识别计算任务，来源: %s", source);
    return true;
}

void ArithmeticProblemNode::calculationWorker() {
    while (worker_running_.load()) {
        if (!calc_signal_.wait()) {
            RCLCPP_ERROR(get_logger(), "等待计算任务信号量失败");
            continue;
        }

        if (!worker_running_.load()) {
            break;
        }

        handleCalculation();
        calc_task_busy_.store(false);

        const bool should_resume_preview = preview_needs_restart_.exchange(false);
        if (should_resume_preview) {
            preview_enabled_.store(true);
        }

        try {
            set_parameter(rclcpp::Parameter(kStartCalcParameter, false));
        } catch (const std::exception& exception) {
            RCLCPP_WARN(get_logger(), "重置 start_calc 参数失败: %s", exception.what());
        }
    }
}

void ArithmeticProblemNode::updatePreviewState(bool enabled) {
    std::lock_guard<std::mutex> lock(preview_state_mutex_);
    preview_enabled_.store(enabled);
    if (!enabled) {
        preview_needs_restart_.store(false);
    }
}

void ArithmeticProblemNode::previewWorker() {
    bool window_open = false;

    while (worker_running_.load()) {
        if (!preview_enabled_.load()) {
            if (window_open) {
                cv::destroyWindow(kPreviewWindowName);
                for (int i = 0; i < 5; ++i) {
                    cv::waitKey(1);
                }
                window_open = false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        Calculator::Config config;
        try {
            config = loadCalculatorConfig();
        } catch (const std::exception& exception) {
            RCLCPP_ERROR(get_logger(), "加载预览配置失败: %s", exception.what());
            preview_enabled_.store(false);
            continue;
        }

        auto camera = std::make_unique<Camera>(config.camera_config_path);
        if (!camera->isOpened()) {
            RCLCPP_ERROR(get_logger(), "预览模式打开相机失败: %s", config.camera_config_path.c_str());
            preview_enabled_.store(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        cv::namedWindow(kPreviewWindowName, cv::WINDOW_NORMAL);
        window_open = true;

        while (worker_running_.load() && preview_enabled_.load()) {
            cv::Mat frame;
            if (!camera->getFrame(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            cv::Mat display = arithmetic_problem::makeArithmeticDisplayFrame(frame);
            if (display.empty()) {
                continue;
            }

            cv::imshow(kPreviewWindowName, display);
            cv::waitKey(1);
        }

        camera.reset();
    }

    if (window_open) {
        cv::destroyWindow(kPreviewWindowName);
        for (int i = 0; i < 5; ++i) {
            cv::waitKey(1);
        }
    }
}

void ArithmeticProblemNode::handleCalculation() {
    try {
        Calculator calculator(loadCalculatorConfig());
        const auto result = calculator.run();

        if (!result.valid) {
            RCLCPP_WARN(
                get_logger(),
                "计算结果稳定性不足: answer=%d, samples=%zu, dominance=%.2f%%",
                result.answer,
                result.sample_count,
                result.dominance * 100.0);
        }

        if (!publishResultIfActive(result)) {
            RCLCPP_INFO(get_logger(), "节点正在停止，放弃发布 vip_box_id");
            return;
        }

        RCLCPP_INFO(
            get_logger(),
            "已发布 vip_box_id=%d, samples=%zu, dominance=%.2f%%",
            result.answer,
            result.sample_count,
            result.dominance * 100.0);
    } catch (const std::exception& exception) {
        RCLCPP_ERROR(get_logger(), "执行识别计算任务失败: %s", exception.what());
    }
}

Calculator::Config ArithmeticProblemNode::loadCalculatorConfig() const {
    Calculator::Config config;

    config.camera_config_path = get_parameter("camera_config_path").as_string();
    config.onnx_model_path = get_parameter("onnx_model_path").as_string();
    config.run_with_cuda = get_parameter("run_with_cuda").as_bool();
    config.show_window = get_parameter("show_window").as_bool();

    const int width = positiveIntOrDefault(
        static_cast<int>(get_parameter("model_input_width").as_int()),
        config.model_input_shape.width);
    const int height = positiveIntOrDefault(
        static_cast<int>(get_parameter("model_input_height").as_int()),
        config.model_input_shape.height);
    config.model_input_shape = cv::Size(width, height);

    config.min_samples = static_cast<std::size_t>(
        positiveIntOrDefault(
            static_cast<int>(get_parameter("min_samples").as_int()),
            static_cast<int>(config.min_samples)));
    config.max_samples = static_cast<std::size_t>(
        positiveIntOrDefault(
            static_cast<int>(get_parameter("max_samples").as_int()),
            static_cast<int>(config.max_samples)));
    config.dominance_threshold = get_parameter("dominance_threshold").as_double();
    config.timeout = std::chrono::milliseconds(
        positiveIntOrDefault(
            static_cast<int>(get_parameter("timeout_ms").as_int()),
            static_cast<int>(config.timeout.count())));

    RCLCPP_INFO(get_logger(), "camera config: %s", config.camera_config_path.c_str());
    RCLCPP_INFO(get_logger(), "onnx model: %s", config.onnx_model_path.c_str());

    return config;
}

bool ArithmeticProblemNode::publishResultIfActive(const Calculator::Result& result) {
    std::lock_guard<std::mutex> lock(calc_state_mutex_);
    if (!worker_running_.load()) {
        return false;
    }

    std_msgs::msg::Int32 message;
    message.data = result.answer;
    vip_box_pub_->publish(message);
    return true;
}

}  // namespace arithmetic_problem
