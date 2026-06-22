#ifndef ARITHMETIC_PROBLEM_CALCULATE_HPP
#define ARITHMETIC_PROBLEM_CALCULATE_HPP

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class Camera;
class Inference;
struct Detection;

namespace arithmetic_problem {

// Calculator 把"取帧 -> 推理 -> 筛选 -> 拼算式 -> 计算 -> 多次结果众数统计"
// 封装为一次 run() 调用，返回最终稳定结果。
cv::Mat makeArithmeticDisplayFrame(const cv::Mat& frame);

class Calculator {
public:
    struct Config {
        std::string camera_config_path;
        std::string onnx_model_path;
        cv::Size model_input_shape{640, 640};
        bool run_with_cuda{true};

        std::size_t max_samples{100};
        std::size_t min_samples{20};
        double dominance_threshold{0.80};
        std::chrono::milliseconds timeout{5000};
        bool show_window{true};
    };

    struct Result {
        bool valid{false};
        int answer{0};
        double dominance{0.0};
        std::size_t sample_count{0};
        std::string last_expression;
    };

    explicit Calculator(const Config& config);
    ~Calculator();

    Calculator(const Calculator&) = delete;
    Calculator& operator=(const Calculator&) = delete;

    Result run();

private:
    Config config_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<Inference> detector_;
};

cv::Mat drawArithmeticDebugFrame(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& expression,
    const std::string& status);

}  // namespace arithmetic_problem

#endif  // ARITHMETIC_PROBLEM_CALCULATE_HPP
