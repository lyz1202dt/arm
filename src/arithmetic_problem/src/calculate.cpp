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

namespace arithmetic_problem {

namespace {

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

    result.answer = modTo1_4(result.raw);
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

bool showDebugFrame(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& expression,
    int answer) {
    cv::Mat display = frame.clone();

    for (const auto& detection : detections) {
        cv::rectangle(display, detection.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(
            display,
            detection.className,
            cv::Point(detection.box.x, detection.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2);
    }

    const std::string info = "Expr: " + expression + "  mod4 = " + std::to_string(answer);
    cv::putText(
        display,
        info,
        cv::Point(20, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 0, 255),
        2);
    cv::imshow(kWindowName, display);

    const char key = static_cast<char>(cv::waitKey(1));
    return key != 'q' && key != 27;
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
    if (config_.show_window) {
        cv::destroyAllWindows();
    }
}

Calculator::Result Calculator::run() {
    Result result;

    if (camera_ == nullptr || !camera_->isOpened()) {
        std::cerr << "[Calculator] 相机不可用，直接返回" << std::endl;
        return result;
    }

    logRuntimeEnvironment();

    if (config_.show_window) {
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    }

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
            continue;
        }

        std::vector<Detection> filtered = filterExpressionTargets(detector_->runInference(frame));
        if (filtered.empty()) {
            continue;
        }

        const std::size_t current_count = filtered.size();
        if (current_count < max_count_seen) {
            pending.reset();
            continue;
        }

        const bool is_complete_frame = current_count == max_count_seen;
        const bool is_larger_candidate = current_count > max_count_seen;

        std::string expression = buildExpression(filtered);
        EvaluationResult evaluation = evaluateExpression(expression);
        if (evaluation.status != ExpressionStatus::kOk) {
            logRejectedExpression(evaluation.status, expression);
            resetInvalidLargerCandidate(is_larger_candidate, current_count, pending);
            continue;
        }

        std::cout << "[Calculator] 算式: " << expression
                  << " | 原始: " << evaluation.raw
                  << " | mod4: " << evaluation.answer << std::endl;

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

        if (config_.show_window &&
            !showDebugFrame(frame, filtered, expression, evaluation.answer)) {
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
            return result;
        }
    }

    fillFinalResult(statistics, config_.min_samples, result);
    return result;
}

}  // namespace arithmetic_problem
