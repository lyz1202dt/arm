#ifndef ARITHMETIC_PROBLEM_ARITHMETIC_PROBLEM_NODE_HPP
#define ARITHMETIC_PROBLEM_ARITHMETIC_PROBLEM_NODE_HPP

#include "arithmetic_problem/calculate.hpp"

#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <semaphore.h>
#include <std_msgs/msg/int32.hpp>

namespace arithmetic_problem {

class CommandSemaphore {
public:
    CommandSemaphore();
    ~CommandSemaphore();

    CommandSemaphore(const CommandSemaphore&) = delete;
    CommandSemaphore& operator=(const CommandSemaphore&) = delete;

    bool post() noexcept;
    bool wait() noexcept;

private:
    sem_t semaphore_{};
};

class ArithmeticProblemNode : public rclcpp::Node {
public:
    ArithmeticProblemNode();
    ~ArithmeticProblemNode() override;

private:
    rcl_interfaces::msg::SetParametersResult onParametersChanged(
        const std::vector<rclcpp::Parameter>& parameters);
    bool requestCalculation(const char* source);
    void calculationWorker();
    void previewWorker();
    void handleCalculation();
    void updatePreviewState(bool enabled);
    Calculator::Config loadCalculatorConfig() const;
    bool publishResultIfActive(const Calculator::Result& result);

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr vip_box_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

    CommandSemaphore calc_signal_;
    std::atomic_bool worker_running_{true};
    std::atomic_bool preview_enabled_{false};
    std::atomic_bool preview_needs_restart_{false};
    std::atomic_bool calc_task_busy_{false};
    std::mutex calc_state_mutex_;
    std::mutex preview_state_mutex_;
    std::thread calc_worker_;
    std::thread preview_worker_;
};

}  // namespace arithmetic_problem

#endif  // ARITHMETIC_PROBLEM_ARITHMETIC_PROBLEM_NODE_HPP
