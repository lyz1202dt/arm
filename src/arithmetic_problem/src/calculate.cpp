// calculate.cpp - Calculator 类实现
//
// Calculator 是 arithmetic_problem 的核心业务模块，负责把相机图像转换成最终答案。
//   1. 从工业相机获取图像
//   2. opencv dnn 加载 onnx 模型推理 outputs
//   3. 遍历 outputs 拿到每个 output.box
//   4. 用运算符号 special_box 做行筛选 + 高度筛选
//   5. 按 x 坐标拼算式计算，结果对 4 取模
//   6. cout 输出结果
//
// 在此基础上加入多帧稳定机制：
//   - 只统计“完整帧 + 可接受表达式”得到的答案样本
//   - 运行时维护已确认接受的最大候选算式目标数 max_count_seen
//   - 更大数量先进入“可疑过计帧 / 观察态”，只有连续三帧且满足一致性门槛才刷新基准
//   - 刷新成功时清空旧统计，并从触发刷新门槛的当帧重新累计
//   - 样本数达到 min_samples 后，如果某个答案占比 > dominance_threshold，就提前返回
//   - 如果超过 timeout 或达到 max_samples，则返回当前众数

#include "arithmetic_problem/calculate.h"

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
#include <unordered_map>
#include <vector>

namespace arithmetic_problem {

namespace {

void resetStatistics(std::vector<int>& samples,
                     std::unordered_map<int, std::size_t>& histogram,
                     int& leader_answer,
                     std::size_t& leader_count) {
    samples.clear();
    histogram.clear();
    leader_answer = 0;
    leader_count = 0;
}

void addSample(int answer,
               std::vector<int>& samples,
               std::unordered_map<int, std::size_t>& histogram,
               int& leader_answer,
               std::size_t& leader_count) {
    samples.push_back(answer);
    const std::size_t count = ++histogram[answer];
    if (count > leader_count) {
        leader_answer = answer;
        leader_count = count;
    }
}

void resetPendingObservation(std::size_t& pending_count,
                             std::size_t& pending_streak,
                             std::vector<int>& pending_answers) {
    pending_count = 0;
    pending_streak = 0;
    pending_answers.clear();
}

bool hasTwoMatchingAnswers(const std::vector<int>& answers) {
    std::unordered_map<int, std::size_t> counts;
    for (int answer : answers) {
        if (++counts[answer] >= 2) {
            return true;
        }
    }
    return false;
}

}  // namespace

Calculator::Calculator(const Config& config) : config_(config) {
    // Camera 只负责从配置文件初始化工业相机；如果失败，run() 会直接返回无效结果。
    camera_ = std::make_unique<Camera>(config_.camera_config_path);
    if (!camera_->isOpened()) {
        std::cerr << "[Calculator] 相机打开失败: " << config_.camera_config_path << std::endl;
    }

    // Inference 封装 OpenCV DNN + ONNX 模型。类别名直接使用 inference.h 里的默认 classes。
    const std::string empty_classes_path;
    detector_ = std::make_unique<Inference>(
        config_.onnx_model_path,
        config_.model_input_shape,
        empty_classes_path,
        config_.run_with_cuda);

    // 预热一次，吃掉首次推理时模型加载、显存初始化等额外开销。
    cv::Mat warmup = cv::Mat::zeros(config_.model_input_shape, CV_8UC3);
    detector_->runInference(warmup);
}

Calculator::~Calculator() {
    // show_window 打开时由本类创建 OpenCV 窗口，因此析构时也由本类统一关闭。
    if (config_.show_window) {
        cv::destroyAllWindows();
    }
}

Calculator::Result Calculator::run() {
    Result result{};

    if (camera_ == nullptr || !camera_->isOpened()) {
        std::cerr << "[Calculator] 相机不可用，直接返回" << std::endl;
        return result;
    }

    // 启动时打印一次运行环境，便于判断 DNN 是否可能使用 CUDA。
    std::cout << "===== Calculator 启动 =====" << std::endl;
    std::cout << "OpenCV 版本: " << CV_VERSION
              << ", CUDA 设备数: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

    if (config_.show_window) {
        cv::namedWindow("Arithmetic Recognition", cv::WINDOW_NORMAL);
    }

    // 频率统计表：key 是最终答案（1-4），value 是该答案出现次数。
    std::unordered_map<int, std::size_t> histogram;

    // samples 保存每一帧得到的有效答案，用于判断样本数和计算占比。
    std::vector<int> samples;
    samples.reserve(config_.max_samples);

    std::size_t max_count_seen = 0;
    int leader_answer = 0;
    std::size_t leader_count = 0;
    std::size_t pending_count = 0;
    std::size_t pending_streak = 0;
    std::vector<int> pending_answers;
    pending_answers.reserve(3);

    const auto start_time = std::chrono::steady_clock::now();

    while (samples.size() < config_.max_samples) {
        // 超过最大运行时间就停止采样，避免一直等不到稳定结果导致节点卡死。
        auto now = std::chrono::steady_clock::now();
        if (now - start_time >= config_.timeout) {
            std::cout << "[Calculator] 已超过 " << config_.timeout.count()
                      << " ms，停止采样" << std::endl;
            break;
        }

        // ---- Step 1: 从工业相机获取图像 ----
        cv::Mat frame;
        if (!camera_->getFrame(frame) || frame.empty()) {
            // 取帧失败不计入样本，继续等待下一帧。
            continue;
        }
        cv::imshow("..10", frame);
        cv::waitKey(1);

        // ---- Step 2: ONNX 推理 ----
        // outputs 中每个 Detection 都包含类别 id、类别名、置信度和检测框 box。
        std::vector<Detection> outputs = detector_->runInference(frame);

        // ---- Step 3+4: 遍历 + 用 special_box 筛选 ----
        // special_box 选取第一个运算符框。算式中的运算符通常位于目标行，
        // 因此用它的 y 坐标和高度作为参考，过滤掉日期、背景数字等干扰框。
        const Detection* special_box = nullptr;
        for (const auto& d : outputs) {
            if (isOperator(d.class_id)) {
                special_box = &d;
                break;
            }
        }
        if (special_box == nullptr) {
            // 没有运算符就无法可靠定位算式行，跳过本帧。
            continue;
        }

        const int ref_y = special_box->box.y;
        const int ref_h = special_box->box.height;

        std::vector<Detection> filtered;
        filtered.reserve(outputs.size());
        for (const auto& d : outputs) {
            // y 坐标相差大于一个 special_box 高度 → 不在同一行。
            if (std::abs(d.box.y - ref_y) > ref_h) continue;

            // 同行但框过大（> 2 倍 special_box 高度）→ 视为干扰目标。
            if (d.box.height > 2 * ref_h) continue;

            filtered.push_back(d);
        }
        if (filtered.empty()) {
            continue;
        }

        const std::size_t current_count = filtered.size();
        if (current_count < max_count_seen) {
            resetPendingObservation(pending_count, pending_streak, pending_answers);
            continue;
        }

        const bool is_complete_frame = (current_count == max_count_seen);
        const bool is_larger_candidate = (current_count > max_count_seen);

        // ---- Step 5: 按 x 坐标拼算式 + 计算 + 对 4 取模 ----
        // 检测框从左到右排序后，类别 id 就可以转换成算式字符序列。
        std::sort(filtered.begin(), filtered.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.box.x < b.box.x;
                  });

        std::string expression;
        expression.reserve(filtered.size());
        for (const auto& d : filtered) {
            expression += classIdToChar(d.class_id);
        }

        // 模型可能把左右括号识别反，按出现顺序重新配对修正；奇数个括号直接视为无效帧。
        if (!normalizeParentheses(expression)) {
            std::cout << "[Calculator] 括号数量为奇数，丢弃本帧: " << expression << std::endl;
            if (is_larger_candidate) {
                std::cout << "[Calculator] 更大候选数 " << current_count
                          << " 的观察链因表达式非法而重置" << std::endl;
                resetPendingObservation(pending_count, pending_streak, pending_answers);
            }
            continue;
        }

        if (!isAcceptableExpression(expression)) {
            std::cout << "[Calculator] 表达式结构非法，丢弃本帧: " << expression << std::endl;
            if (is_larger_candidate) {
                std::cout << "[Calculator] 更大候选数 " << current_count
                          << " 的观察链因表达式非法而重置" << std::endl;
                resetPendingObservation(pending_count, pending_streak, pending_answers);
            }
            continue;
        }

        long long raw = 0;
        if (!tryCalcExpression(expression, raw)) {
            std::cout << "[Calculator] 表达式计算失败，丢弃本帧: " << expression << std::endl;
            if (is_larger_candidate) {
                std::cout << "[Calculator] 更大候选数 " << current_count
                          << " 的观察链因表达式非法而重置" << std::endl;
                resetPendingObservation(pending_count, pending_streak, pending_answers);
            }
            continue;
        }

        const int answer = modTo1_4(raw);

        // ---- Step 6: cout 输出 ----
        std::cout << "[Calculator] 算式: " << expression
                  << " | 原始: " << raw
                  << " | mod4: " << answer << std::endl;

        bool should_add_sample = false;
        if (is_complete_frame) {
            resetPendingObservation(pending_count, pending_streak, pending_answers);
            should_add_sample = true;
        } else if (pending_streak == 0 || current_count != pending_count) {
            pending_count = current_count;
            pending_streak = 1;
            pending_answers.clear();
            pending_answers.push_back(answer);
            std::cout << "[Calculator] 首次观察到更大候选数 " << current_count
                      << "，标记为可疑过计帧" << std::endl;
        } else {
            ++pending_streak;
            pending_answers.push_back(answer);
            std::cout << "[Calculator] 更大候选数 " << current_count
                      << " 进入观察态，第 " << pending_streak << " 帧" << std::endl;

            if (pending_streak >= 3) {
                if (hasTwoMatchingAnswers(pending_answers)) {
                    max_count_seen = pending_count;
                    resetStatistics(samples, histogram, leader_answer, leader_count);
                    std::cout << "[Calculator] 刷新最大目标数基准: " << max_count_seen
                              << "，清空旧统计" << std::endl;
                    should_add_sample = true;
                } else {
                    std::cout << "[Calculator] 更大候选数 " << current_count
                              << " 连续三帧结果一致性不足，放弃本轮观察" << std::endl;
                }
                resetPendingObservation(pending_count, pending_streak, pending_answers);
            }
        }

        if (should_add_sample) {
            addSample(answer, samples, histogram, leader_answer, leader_count);
            result.last_expression = expression;
        }

        // 可视化只在调试时开启；部署为 ROS2 节点时默认关闭。
        if (config_.show_window) {
            cv::Mat display = frame.clone();

            // 绿色框表示经过 special_box 规则筛选后真正参与算式拼接的目标。
            for (const auto& d : filtered) {
                cv::rectangle(display, d.box, cv::Scalar(0, 255, 0), 2);
                cv::putText(display, d.className,
                            cv::Point(d.box.x, d.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 0), 2);
            }

            std::string info = "Expr: " + expression + "  mod4 = " + std::to_string(answer);
            cv::putText(display, info, cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 0, 255), 2);
            cv::imshow("Arithmetic Recognition", display);

            // 调试窗口中按 q 或 ESC 可以提前结束统计。
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 27) break;
        }

        // 达到最少样本数后才检查"占比 > 30%"的提前返回条件，避免少量样本误判。
        if (samples.size() >= config_.min_samples) {
            const double dominance =
                static_cast<double>(leader_count) / static_cast<double>(samples.size());

            if (dominance > config_.dominance_threshold) {
                result.valid = true;
                result.answer = leader_answer;
                result.dominance = dominance;
                result.sample_count = samples.size();
                std::cout << "[Calculator] 提前收敛: 众数=" << leader_answer
                          << ", 占比=" << (dominance * 100.0) << "%"
                          << ", 样本数=" << samples.size() << std::endl;
                return result;
            }
        }
    }

    // 退出循环（超时、达到 max_samples 或调试窗口退出）后，把当前众数作为最终结果。
    if (!samples.empty()) {
        const double dominance =
            static_cast<double>(leader_count) / static_cast<double>(samples.size());

        // 样本数不足 min_samples 时，answer 仍给出当前众数，但 valid=false 提醒上层可信度不足。
        result.valid = (samples.size() >= config_.min_samples);
        result.answer = leader_answer;
        result.dominance = dominance;
        result.sample_count = samples.size();
        std::cout << "[Calculator] 最终结果: 众数=" << leader_answer
                  << ", 占比=" << (dominance * 100.0) << "%"
                  << ", 样本数=" << samples.size() << std::endl;
    } else {
        std::cout << "[Calculator] 未采到任何有效样本" << std::endl;
    }

    return result;
}

}  // namespace arithmetic_problem
