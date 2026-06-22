#include "arithmetic_problem/calculate.hpp"

#include "arithmetic_problem/arithmetic_utils.hpp"
#include "camera_driver.h"
#include "inference.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using arithmetic_problem::Calculator;
using arithmetic_problem::classIdToChar;
using arithmetic_problem::isAcceptableExpression;
using arithmetic_problem::isOperator;
using arithmetic_problem::normalizeParentheses;
using arithmetic_problem::tryCalcExpression;

constexpr char kWindowName[] = "Arithmetic Recognition";

enum class ExpressionStatus {
    kOk,
    kOddParentheses,
    kInvalidStructure,
    kCalculationFailed,
};

struct EvaluationResult {
    ExpressionStatus status{ExpressionStatus::kOk};
    long long raw{0};
    int answer{0};
};

struct SampleStatistics {
    explicit SampleStatistics(std::size_t max_samples) {
        samples.reserve(max_samples);
    }

    void reset() {
        samples.clear();
        histogram.clear();
        leader_answer = 0;
        leader_count = 0;
    }

    void add(int answer) {
        samples.push_back(answer);
        const std::size_t count = ++histogram[answer];
        if (count > leader_count) {
            leader_answer = answer;
            leader_count = count;
        }
    }

    bool empty() const {
        return samples.empty();
    }

    std::size_t size() const {
        return samples.size();
    }

    double dominance() const {
        if (samples.empty()) {
            return 0.0;
        }
        return static_cast<double>(leader_count) / static_cast<double>(samples.size());
    }

    std::vector<int> samples;
    std::unordered_map<int, std::size_t> histogram;
    int leader_answer{0};
    std::size_t leader_count{0};
};

struct PendingObservation {
    PendingObservation() {
        answers.reserve(3);
    }

    void reset() {
        count = 0;
        streak = 0;
        answers.clear();
    }

    bool matches(std::size_t current_count) const {
        return streak != 0 && current_count == count;
    }

    void start(std::size_t current_count, int answer) {
        count = current_count;
        streak = 1;
        answers.clear();
        answers.push_back(answer);
    }

    void advance(int answer) {
        ++streak;
        answers.push_back(answer);
    }

    bool hasTwoMatchingAnswers() const {
        std::unordered_map<int, std::size_t> answer_counts;
        for (int answer : answers) {
            if (++answer_counts[answer] >= 2) {
                return true;
            }
        }
        return false;
    }

    std::size_t count{0};
    std::size_t streak{0};
    std::vector<int> answers;
};

}  // namespace

namespace arithmetic_problem {

cv::Mat makeArithmeticDisplayFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }

    cv::Mat display;
    if (frame.depth() == CV_8U) {
        display = frame.clone();
    } else {
        cv::normalize(frame, display, 0, 255, cv::NORM_MINMAX);
        display.convertTo(display, CV_8U);
    }

    if (display.channels() == 1) {
        cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    } else if (display.channels() == 4) {
        cv::cvtColor(display, display, cv::COLOR_BGRA2BGR);
    }

    return display;
}

cv::Mat drawArithmeticDebugFrame(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& expression,
    const std::string& status) {
    cv::Mat display = makeArithmeticDisplayFrame(frame);
    if (display.empty()) {
        return display;
    }

    for (const auto& detection : detections) {
        cv::rectangle(display, detection.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(
            display,
            detection.className,
            cv::Point(detection.box.x, std::max(20, detection.box.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2);
    }

    const std::string expr_line = expression.empty() ? "Expr: -" : "Expr: " + expression;
    cv::putText(
        display,
        expr_line,
        cv::Point(20, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 0, 255),
        2);
    cv::putText(
        display,
        status,
        cv::Point(20, 80),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 0, 255),
        2);

    return display;
}

class DebugWindow {
public:
    explicit DebugWindow(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        }
    }

    ~DebugWindow() {
        close();
    }

    void close() noexcept {
        if (!enabled_ || closed_) {
            return;
        }

        try {
            cv::destroyWindow(kWindowName);
            for (int i = 0; i < 5; ++i) {
                cv::waitKey(1);
            }
        } catch (const cv::Exception& exception) {
            std::cerr << "[Calculator] 销毁调试窗口失败: " << exception.what() << std::endl;
        }

        closed_ = true;
    }

    DebugWindow(const DebugWindow&) = delete;
    DebugWindow& operator=(const DebugWindow&) = delete;

    bool showStatus(const std::string& status) const {
        if (!enabled_) {
            return true;
        }

        cv::Mat display = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(
            display,
            status,
            cv::Point(20, 60),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 0, 255),
            2);

        cv::imshow(kWindowName, display);
        const char key = static_cast<char>(cv::waitKey(1));
        return key != 'q' && key != 27;
    }

    bool show(
        const cv::Mat& frame,
        const std::vector<Detection>& detections,
        const std::string& expression,
        const std::string& status) const {
        if (!enabled_) {
            return true;
        }

        cv::Mat display = drawArithmeticDebugFrame(frame, detections, expression, status);
        if (display.empty()) {
            return showStatus("No displayable frame");
        }

        cv::imshow(kWindowName, display);
        const char key = static_cast<char>(cv::waitKey(1));
        return key != 'q' && key != 27;
    }

private:
    bool enabled_{false};
    bool closed_{false};
};

void logRuntimeEnvironment() {
    std::cout << "===== Calculator 启动 =====" << std::endl;
    std::cout << "OpenCV 版本: " << CV_VERSION
              << ", CUDA 设备数: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
}

std::vector<Detection> filterExpressionTargets(const std::vector<Detection>& outputs) {
    const Detection* special_box = nullptr;
    for (const auto& detection : outputs) {
        if (isOperator(detection.class_id)) {
            special_box = &detection;
            break;
        }
    }

    if (special_box == nullptr) {
        return {};
    }

    const int ref_y = special_box->box.y;
    const int ref_h = special_box->box.height;

    std::vector<Detection> filtered;
    filtered.reserve(outputs.size());
    for (const auto& detection : outputs) {
        if (std::abs(detection.box.y - ref_y) > ref_h) {
            continue;
        }

        if (detection.box.height > 2 * ref_h) {
            continue;
        }

        filtered.push_back(detection);
    }

    return filtered;
}

std::string buildExpression(std::vector<Detection>& detections) {
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& lhs, const Detection& rhs) {
            return lhs.box.x < rhs.box.x;
        });

    std::string expression;
    expression.reserve(detections.size());
    for (const auto& detection : detections) {
        expression += classIdToChar(detection.class_id);
    }
    return expression;
}

EvaluationResult evaluateExpression(std::string& expression) {
    EvaluationResult result;

    if (!normalizeParentheses(expression)) {
        result.status = ExpressionStatus::kOddParentheses;
        return result;
    }

    if (!isAcceptableExpression(expression)) {
        result.status = ExpressionStatus::kInvalidStructure;
        return result;
    }

    if (!tryCalcExpression(expression, result.raw)) {
        result.status = ExpressionStatus::kCalculationFailed;
        return result;
    }

    result.answer = result.raw;
    return result;
}

void logRejectedExpression(ExpressionStatus status, const std::string& expression) {
    switch (status) {
        case ExpressionStatus::kOddParentheses:
            std::cout << "[Calculator] 括号数量为奇数，丢弃本帧: " << expression << std::endl;
            break;
        case ExpressionStatus::kInvalidStructure:
            std::cout << "[Calculator] 表达式结构非法，丢弃本帧: " << expression << std::endl;
            break;
        case ExpressionStatus::kCalculationFailed:
            std::cout << "[Calculator] 表达式计算失败，丢弃本帧: " << expression << std::endl;
            break;
        case ExpressionStatus::kOk:
            break;
    }
}

std::string expressionStatusText(ExpressionStatus status) {
    switch (status) {
        case ExpressionStatus::kOddParentheses:
            return "Rejected: odd parentheses";
        case ExpressionStatus::kInvalidStructure:
            return "Rejected: invalid expression";
        case ExpressionStatus::kCalculationFailed:
            return "Rejected: calculation failed";
        case ExpressionStatus::kOk:
            return "OK";
    }
    return "Unknown";
}

void resetInvalidLargerCandidate(
    bool is_larger_candidate,
    std::size_t current_count,
    PendingObservation& pending) {
    if (!is_larger_candidate) {
        return;
    }

    std::cout << "[Calculator] 更大候选数 " << current_count
              << " 的观察链因表达式非法而重置" << std::endl;
    pending.reset();
}

bool shouldAddSampleForFrame(
    bool is_complete_frame,
    std::size_t current_count,
    int answer,
    std::size_t& max_count_seen,
    PendingObservation& pending,
    SampleStatistics& statistics) {
    if (is_complete_frame) {
        pending.reset();
        return true;
    }

    if (!pending.matches(current_count)) {
        pending.start(current_count, answer);
        std::cout << "[Calculator] 首次观察到更大候选数 " << current_count
                  << "，标记为可疑过计帧" << std::endl;
        return false;
    }

    pending.advance(answer);
    std::cout << "[Calculator] 更大候选数 " << current_count
              << " 进入观察态，第 " << pending.streak << " 帧" << std::endl;

    if (pending.streak < 3) {
        return false;
    }

    if (pending.hasTwoMatchingAnswers()) {
        max_count_seen = pending.count;
        statistics.reset();
        std::cout << "[Calculator] 刷新最大目标数基准: " << max_count_seen
                  << "，清空旧统计" << std::endl;
        pending.reset();
        return true;
    }

    std::cout << "[Calculator] 更大候选数 " << current_count
              << " 连续三帧结果一致性不足，放弃本轮观察" << std::endl;
    pending.reset();
    return false;
}

bool reachedTimeout(
    const std::chrono::steady_clock::time_point& start_time,
    std::chrono::milliseconds timeout) {
    return std::chrono::steady_clock::now() - start_time >= timeout;
}

Calculator::Result makeConvergedResult(
    const SampleStatistics& statistics,
    const std::string& last_expression) {
    Calculator::Result result;
    result.valid = true;
    result.answer = statistics.leader_answer;
    result.dominance = statistics.dominance();
    result.sample_count = statistics.size();
    result.last_expression = last_expression;
    return result;
}

void fillFinalResult(
    const SampleStatistics& statistics,
    std::size_t min_samples,
    Calculator::Result& result) {
    if (statistics.empty()) {
        std::cout << "[Calculator] 未采到任何有效样本" << std::endl;
        return;
    }

    result.valid = statistics.size() >= min_samples;
    result.answer = statistics.leader_answer;
    result.dominance = statistics.dominance();
    result.sample_count = statistics.size();
    std::cout << "[Calculator] 最终结果: 众数=" << result.answer
              << ", 占比=" << (result.dominance * 100.0) << "%"
              << ", 样本数=" << result.sample_count << std::endl;
}

}  // namespace

namespace arithmetic_problem {

Calculator::Calculator(const Config& config) : config_(config) {
    camera_ = std::make_unique<Camera>(config_.camera_config_path);
    if (!camera_->isOpened()) {
        std::cerr << "[Calculator] 相机打开失败: " << config_.camera_config_path << std::endl;
    }

    const std::string empty_classes_path;
    detector_ = std::make_unique<Inference>(
        config_.onnx_model_path,
        config_.model_input_shape,
        empty_classes_path,
        config_.run_with_cuda);

    cv::Mat warmup = cv::Mat::zeros(config_.model_input_shape, CV_8UC3);
    detector_->runInference(warmup);
}

Calculator::~Calculator() {
}

Calculator::Result Calculator::run() {
    Result result;

    if (camera_ == nullptr || !camera_->isOpened()) {
        std::cerr << "[Calculator] 相机不可用，直接返回" << std::endl;
        return result;
    }

    logRuntimeEnvironment();
    DebugWindow debug_window(config_.show_window);

    SampleStatistics statistics(config_.max_samples);
    PendingObservation pending;
    std::size_t max_count_seen = 0;
    std::string last_expression;

    const auto start_time = std::chrono::steady_clock::now();

    while (statistics.size() < config_.max_samples) {
        if (reachedTimeout(start_time, config_.timeout)) {
            std::cout << "[Calculator] 已超过 " << config_.timeout.count()
                      << " ms，停止采样" << std::endl;
            break;
        }

        cv::Mat frame;
        if (!camera_->getFrame(frame) || frame.empty()) {
            if (!debug_window.showStatus("No camera frame")) {
                break;
            }
            continue;
        }

        const std::string frame_status = "Camera frame: " +
            std::to_string(frame.cols) + "x" + std::to_string(frame.rows) +
            "  channels: " + std::to_string(frame.channels());
        if (!debug_window.show(frame, {}, "", frame_status)) {
            break;
        }

        const std::vector<Detection> detections = detector_->runInference(frame);
        std::vector<Detection> filtered = filterExpressionTargets(detections);
        if (filtered.empty()) {
            if (!debug_window.show(frame, detections, "", "Waiting for expression")) {
                break;
            }
            continue;
        }

        const std::size_t current_count = filtered.size();
        if (current_count < max_count_seen) {
            pending.reset();
            if (!debug_window.show(frame, filtered, "", "Skipped: incomplete frame")) {
                break;
            }
            continue;
        }

        const bool is_complete_frame = current_count == max_count_seen;
        const bool is_larger_candidate = current_count > max_count_seen;

        std::string expression = buildExpression(filtered);
        EvaluationResult evaluation = evaluateExpression(expression);
        if (evaluation.status != ExpressionStatus::kOk) {
            logRejectedExpression(evaluation.status, expression);
            resetInvalidLargerCandidate(is_larger_candidate, current_count, pending);
            if (!debug_window.show(frame, filtered, expression, expressionStatusText(evaluation.status))) {
                break;
            }
            continue;
        }

        std::cout << "[Calculator] 算式: " << expression
                  << " | 原始: " << evaluation.raw
                  << " | 结果: " << evaluation.answer << std::endl;

        const bool should_add_sample = shouldAddSampleForFrame(
            is_complete_frame,
            current_count,
            evaluation.answer,
            max_count_seen,
            pending,
            statistics);

        if (should_add_sample) {
            statistics.add(evaluation.answer);
            last_expression = expression;
            result.last_expression = last_expression;
        }

        const std::string status = "Answer: " + std::to_string(evaluation.answer) +
            "  Samples: " + std::to_string(statistics.size()) +
            "/" + std::to_string(config_.max_samples);
        if (!debug_window.show(frame, filtered, expression, status)) {
            break;
        }

        if (statistics.size() < config_.min_samples) {
            continue;
        }

        const double dominance = statistics.dominance();
        if (dominance > config_.dominance_threshold) {
            result = makeConvergedResult(statistics, last_expression);
            std::cout << "[Calculator] 提前收敛: 众数=" << result.answer
                      << ", 占比=" << (result.dominance * 100.0) << "%"
                      << ", 样本数=" << result.sample_count << std::endl;
            debug_window.close();
            return result;
        }
    }

    fillFinalResult(statistics, config_.min_samples, result);
    debug_window.close();
    return result;
}

}  // namespace arithmetic_problem
