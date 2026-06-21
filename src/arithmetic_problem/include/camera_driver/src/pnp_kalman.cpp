#include "pnp_kalman.h"

    ExtendedKalmanFilter::ExtendedKalmanFilter() {
        X = cv::Mat::zeros(9, 1, CV_64F);
        P = cv::Mat::eye(9, 9, CV_64F) * 0.1;

        // 9 维状态采用“位置 + 速度 + 加速度”模型，便于在 PnP 观测抖动时保留运动连续性。
        F = cv::Mat::eye(9, 9, CV_64F);

        // 过程噪声刻意给加速度更高权重，让滤波器在目标快速移动时更容易跟随观测变化。
        Q = cv::Mat::eye(9, 9, CV_64F) * 0.01;
        Q.at<double>(6,6) = Q.at<double>(7,7) = Q.at<double>(8,8) = 0.1;  // 加速度噪声
        Q *= 0.01;

        // 只观测三维位置，速度和加速度由模型在跨帧预测中自行估计。
        H = cv::Mat::zeros(3, 9, CV_64F);
        H.at<double>(0, 0) = 1.0;
        H.at<double>(1, 1) = 1.0;
        H.at<double>(2, 2) = 1.0;

        // 观测噪声越大，越说明当前帧 PnP 结果不可靠，滤波器会更保守。
        R = cv::Mat::eye(3, 3, CV_64F) * 0.05;
    }

    void ExtendedKalmanFilter::predict(double dt) {
        // 把 dt 写回 F，确保状态转移矩阵和实际帧间隔一致，而不是假设固定帧率。
        F.at<double>(0, 3) = dt;
        F.at<double>(0, 6) = dt*dt*0.5;
        F.at<double>(1, 4) = dt;
        F.at<double>(1, 7) = dt*dt*0.5;
        F.at<double>(2, 5) = dt;
        F.at<double>(2, 8) = dt*dt*0.5;


        // 速度项继续叠加加速度，保证位置预测不会在连续帧中突然停住。
        F.at<double>(3, 6) = dt;       // vx += ax*dt
        F.at<double>(4, 7) = dt;
        F.at<double>(5, 8) = dt;

        // 先预测，再用新一帧 PnP 结果修正，这是 EKF 的标准顺序。
        X = F * X;

        // 协方差同步扩大，表示经过一次预测后状态不确定性增加。
        P = F * P * F.t() + Q;
    }

    void ExtendedKalmanFilter::update(cv::Mat Z) {
        // 残差只比较三维位置，避免把速度/加速度的隐状态直接和观测硬对齐。
        cv::Mat Y = Z - H * X;  // 计算残差
        cv::Mat S = H * P * H.t() + R;
        cv::Mat K = P * H.t() * S.inv();  // 计算卡尔曼增益

        // 用观测修正预测结果，目标是把单帧 PnP 抖动压到更平滑的轨迹上。
        X = X + K * Y;

        // 更新后的协方差会收缩，表示融合过观测后状态更确定。
        P = (cv::Mat::eye(9, 9, CV_64F) - K * H) * P;
    }



    // 构造函数：初始化相机参数和物体3D模型
    PnPEKFTracker::PnPEKFTracker(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                  float width , float height , float depth )
        : cameraMatrix(camera_matrix), distCoeffs(dist_coeffs), isInitialized(false) {
        (void)height;
        (void)depth;


        // 四个 3D 点必须与传入的 2D 像素点保持同一顺序，否则 solvePnP 会得到错误位姿。
        objectPoints = {
            {-width, width, 0},          // 左上角
            {width, width, 0},           // 右上角
            {width, -width, 0},           // 右下角
            {-width, -width, 0}            // 左下角
        };
    }

    // 预测位置：输入2D像素点，输出预测的3D位置 (x, y, z)
    cv::Mat PnPEKFTracker::predict_position(const std::vector<cv::Point2f>& image_points, double dt ) {
        // PnP 给出当前帧的绝对位置观测，EKF 再把它融合进跨帧运动模型。
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(objectPoints, image_points, cameraMatrix, distCoeffs,
                                   rvec, tvec, false, cv::SOLVEPNP_P3P);

        if (!success) {
            std::cerr << "solvePnP 失败! 返回上一次的预测值" << std::endl;
            cv::Mat state = cv::Mat::zeros(1, 6, CV_32F);
            state.at<float>(0) = ekf.X.at<double>(0);
            state.at<float>(1) = ekf.X.at<double>(1);
            state.at<float>(2) = ekf.X.at<double>(2);
            state.at<float>(3) = ekf.X.at<double>(3);
            state.at<float>(4) = ekf.X.at<double>(4);
            state.at<float>(5) = ekf.X.at<double>(5);
            return state;
        }

        // 第一帧没有历史速度/加速度，只能用 PnP 的位置作为滤波初值。
        if (!isInitialized) {
            ekf.X.at<double>(0) = tvec.at<double>(0);
            ekf.X.at<double>(1) = tvec.at<double>(1);
            ekf.X.at<double>(2) = tvec.at<double>(2);
            isInitialized = true;

            cv::Mat state = cv::Mat::zeros(1, 6, CV_32F);
            state.at<float>(0) = tvec.at<double>(0);
            state.at<float>(1) = tvec.at<double>(1);
            state.at<float>(2) = tvec.at<double>(2);
            return state;
        }

        // 先根据运动模型预测，再用当前帧 PnP 观测修正，输出更平滑的 3D 位置。
        ekf.predict(dt);

        cv::Mat Z = tvec.clone();
        ekf.update(Z);

        cv::Mat state = cv::Mat::zeros(1, 6, CV_32F);
        state.at<float>(0) = ekf.X.at<double>(0);
        state.at<float>(1) = ekf.X.at<double>(1);
        state.at<float>(2) = ekf.X.at<double>(2);
        state.at<float>(3) = ekf.X.at<double>(3);
        state.at<float>(4) = ekf.X.at<double>(4);
        state.at<float>(5) = ekf.X.at<double>(5);
        return state;
    }
