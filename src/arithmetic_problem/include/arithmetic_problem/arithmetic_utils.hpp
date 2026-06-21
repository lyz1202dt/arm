#ifndef ARITHMETIC_PROBLEM_ARITHMETIC_UTILS_HPP
#define ARITHMETIC_PROBLEM_ARITHMETIC_UTILS_HPP

// 算式识别相关的纯函数工具
// 这些函数从原 arithmetic.cpp 中抽出，便于在不同模块（如 calculate 库）复用

#include <cctype>
#include <string>
#include <vector>

namespace arithmetic_problem {

// 模型类别索引（按 inference.h 中 classes 顺序）：
// 0-9: 数字 "0"-"9"，10: "+"，11: "-"，12: "×"，13: "÷"，14: "("，15: ")"
inline bool isOperator(int class_id) {
    // 运算符号 +, -, ×, ÷
    return class_id >= 10 && class_id <= 13;
}

inline char classIdToChar(int class_id) {
    if (class_id >= 0 && class_id <= 9) return static_cast<char>('0' + class_id);
    switch (class_id) {
        case 10: return '+';
        case 11: return '-';
        case 12: return '*';  // × 转为 *
        case 13: return '/';  // ÷ 转为 /
        case 14: return '(';
        case 15: return ')';
        default: return '?';
    }
}

inline bool isBinaryOperator(char ch) {
    return ch == '+' || ch == '-' || ch == '*' || ch == '/';
}

inline bool normalizeParentheses(std::string& expr) {
    std::vector<size_t> paren_positions;
    for (size_t i = 0; i < expr.size(); ++i) {
        if (expr[i] == '(' || expr[i] == ')') {
            paren_positions.push_back(i);
        }
    }

    if (paren_positions.size() % 2 == 1) {
        return false;
    }

    for (size_t k = 0; k < paren_positions.size(); ++k) {
        expr[paren_positions[k]] = (k % 2 == 0) ? '(' : ')';
    }
    return true;
}

inline bool isAcceptableExpression(const std::string& expr) {
    if (expr.empty()) return false;

    int balance = 0;
    char prev = '\0';

    for (size_t i = 0; i < expr.size(); ++i) {
        const char ch = expr[i];
        const bool is_digit = std::isdigit(static_cast<unsigned char>(ch)) != 0;
        const bool is_left_paren = ch == '(';
        const bool is_right_paren = ch == ')';
        const bool is_op = isBinaryOperator(ch);

        if (!is_digit && !is_left_paren && !is_right_paren && !is_op) {
            return false;
        }

        if (i == 0) {
            if (is_op || is_right_paren) return false;
        } else {
            const bool prev_is_digit = std::isdigit(static_cast<unsigned char>(prev)) != 0;
            const bool prev_is_left_paren = prev == '(';
            const bool prev_is_right_paren = prev == ')';
            const bool prev_is_op = isBinaryOperator(prev);

            if (prev_is_op && (is_op || is_right_paren)) return false;
            if (prev_is_left_paren && (ch == '*' || ch == '/' || is_right_paren || ch == '-')) return false;
            if ((prev_is_digit || prev_is_right_paren) && is_left_paren) return false;
            if (prev_is_right_paren && is_digit) return false;
        }

        if (is_left_paren) {
            ++balance;
        } else if (is_right_paren) {
            --balance;
            if (balance < 0) return false;
        }

        prev = ch;
    }

    if (balance != 0) return false;
    if (isBinaryOperator(prev) || prev == '(') return false;

    return true;
}

// 四则运算计算器（支持 + - * / 和括号，按运算优先级）
inline bool tryCalcExpression(const std::string& expr, long long& result) {
    std::vector<long long> nums;
    std::vector<char> ops;

    auto precedence = [](char op) {
        if (op == '(' || op == ')') return 0;
        if (op == '+' || op == '-') return 1;
        if (op == '*' || op == '/') return 2;
        return 0;
    };

    auto applyTop = [&]() -> bool {
        if (ops.empty() || nums.size() < 2) return false;
        const char op = ops.back();
        ops.pop_back();
        const long long b = nums.back();
        nums.pop_back();
        const long long a = nums.back();
        nums.pop_back();

        long long r = 0;
        switch (op) {
            case '+': r = a + b; break;
            case '-': r = a - b; break;
            case '*': r = a * b; break;
            case '/':
                if (b == 0) return false;
                r = a / b;
                break;
            default:
                return false;
        }
        nums.push_back(r);
        return true;
    };

    size_t i = 0;
    while (i < expr.size()) {
        const char ch = expr[i];
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            long long val = 0;
            while (i < expr.size() && std::isdigit(static_cast<unsigned char>(expr[i]))) {
                val = val * 10 + (expr[i] - '0');
                ++i;
            }
            nums.push_back(val);
            continue;
        }

        if (ch == '(') {
            ops.push_back(ch);
            ++i;
            continue;
        }

        if (ch == ')') {
            while (!ops.empty() && ops.back() != '(') {
                if (!applyTop()) return false;
            }
            if (ops.empty() || ops.back() != '(') return false;
            ops.pop_back();
            ++i;
            continue;
        }

        if (isBinaryOperator(ch)) {
            while (!ops.empty() && ops.back() != '(' &&
                   precedence(ops.back()) >= precedence(ch)) {
                if (!applyTop()) return false;
            }
            ops.push_back(ch);
            ++i;
            continue;
        }

        return false;
    }

    while (!ops.empty()) {
        if (ops.back() == '(') return false;
        if (!applyTop()) return false;
    }

    if (nums.size() != 1) return false;
    result = nums.back();
    return true;
}

// 把结果对 4 取模，得到 1-4 之间的正整数
inline int modTo1_4(long long result) {
    long long answer = result;
    if (answer <= 0) answer += 4;
    return static_cast<int>(answer);
}

}  // namespace arithmetic_problem

#endif  // ARITHMETIC_PROBLEM_ARITHMETIC_UTILS_HPP
