#include "arithmetic_problem/calculate.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

namespace {

std::string getEnvOrDefault(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("arithmetic_problem_node");

    arithmetic_problem::Calculator::Config config;
    const std::string package_source_dir = ARITHMETIC_PROBLEM_SOURCE_DIR;
    config.camera_config_path = getEnvOrDefault(
        "ARITHMETIC_CAMERA_CONFIG",
        package_source_dir + "/include/camera_driver/camera_init/HIKcamera0.yaml");
    config.onnx_model_path = getEnvOrDefault(
        "ARITHMETIC_ONNX_MODEL",
        package_source_dir + "/src/best.onnx");
    config.run_with_cuda = node->declare_parameter<bool>("run_with_cuda", true);
    config.show_window = node->declare_parameter<bool>("show_window", false);
    config.min_samples = static_cast<std::size_t>(
        node->declare_parameter<int>("min_samples", static_cast<int>(config.min_samples)));
    config.max_samples = static_cast<std::size_t>(
        node->declare_parameter<int>("max_samples", static_cast<int>(config.max_samples)));
    config.dominance_threshold = node->declare_parameter<double>(
        "dominance_threshold", config.dominance_threshold);
    config.timeout = std::chrono::milliseconds(
        node->declare_parameter<int>("timeout_ms", static_cast<int>(config.timeout.count())));

    RCLCPP_INFO(node->get_logger(), "camera config: %s", config.camera_config_path.c_str());
    RCLCPP_INFO(node->get_logger(), "onnx model: %s", config.onnx_model_path.c_str());

    auto publisher = node->create_publisher<std_msgs::msg::Int32>(
        "vip_box_id", rclcpp::QoS(1).reliable().transient_local());

    arithmetic_problem::Calculator calculator(config);
    const auto result = calculator.run();

    if (!result.valid) {
        RCLCPP_WARN(
            node->get_logger(),
            "calculator returned an unstable result: answer=%d, samples=%zu, dominance=%.2f%%",
            result.answer,
            result.sample_count,
            result.dominance * 100.0);
    }

    std_msgs::msg::Int32 msg;
    msg.data = result.answer;
    publisher->publish(msg);

    RCLCPP_INFO(
        node->get_logger(),
        "published vip_box_id=%d, samples=%zu, dominance=%.2f%%",
        msg.data,
        result.sample_count,
        result.dominance * 100.0);

    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    rclcpp::shutdown();
    return result.answer == 0 ? 1 : 0;
}
