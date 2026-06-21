#ifndef ARITHMETIC_PROBLEM_CALCULATE_H
#define ARITHMETIC_PROBLEM_CALCULATE_H

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

// 前置声明，避免头文件污染
class Camera;
class Inference;

namespace arithmetic_problem {

// Calculator 把"取帧 → 推理 → 筛选 → 拼算式 → 计算 → 取模 → 多次结果众数统计"
// 这一整条流程封装在内部，对外只暴露一个 run() 方法返回最终稳定结果。
class Calculator {
public:
    struct Config {
        std::string camera_config_path;          // 相机 yaml 配置路径
        std::string onnx_model_path;             // ONNX 模型路径
        cv::Size model_input_shape{640, 640};    // 模型输入尺寸
        bool run_with_cuda{true};                // 是否使用 CUDA

        std::size_t max_samples{100};            // 最多统计 100 项结果
        std::size_t min_samples{20};             // 至少统计 20 项才允许提前返回
        double dominance_threshold{0.80};        // 占比超过 80% 即可提前返回
        std::chrono::milliseconds timeout{5000}; // 超过 5 秒强制返回当前众数
        bool show_window{false};                 // 是否打开 imshow 窗口可视化
    };

    struct Result {
        bool valid{false};        // 是否拿到了有效结果
        int answer{0};            // 对 4 取模后的结果（1-4）
        double dominance{0.0};    // 众数在已统计样本中的占比
        std::size_t sample_count{0}; // 实际参与统计的样本数
        std::string last_expression; // 最近一次拼出的算式（调试用）
    };

    explicit Calculator(const Config& config);
    ~Calculator();

    Calculator(const Calculator&) = delete;
    Calculator& operator=(const Calculator&) = delete;

    // 启动主循环，返回时给出经过众数统计的最终结果。
    Result run();

private:
    Config config_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<Inference> detector_;
};

}  // namespace arithmetic_problem

#endif  // ARITHMETIC_PROBLEM_CALCULATE_H
