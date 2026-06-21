#include "arithmetic_problem/arithmetic_problem_node.hpp"

#include <memory>
#include <exception>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<arithmetic_problem::ArithmeticProblemNode>());
    rclcpp::shutdown();
    return 0;
}
