#ifndef PNP_KALMAN_H
#define PNP_KALMAN_H

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>



class ExtendedKalmanFilter {
    public:
        // 状态向量为 [x, y, z, vx, vy, vz, ax, ay, az]，观测只来自 PnP 的三维平移 tvec。
        cv::Mat X;
        cv::Mat P;  // 误差协方差矩阵
        cv::Mat F;  // 状态转移矩阵
        cv::Mat Q;  // 过程噪声
        cv::Mat H;  // 观测矩阵
        cv::Mat R;  // 观测噪声

        ExtendedKalmanFilter();

        void predict(double dt);
        void update(cv::Mat Z);
    };
class PnPEKFTracker {
    private:
        ExtendedKalmanFilter ekf;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        std::vector<cv::Point3f> objectPoints;
        bool isInitialized;
    public:
        // 默认内参与畸变参数用于现有相机调试；正式部署时应优先传入 YAML 标定结果。
        PnPEKFTracker(
            const cv::Mat& camera_matrix = (cv::Mat_<double>(3, 3) << 2669.911147, 0.000000, 644.911381,
            0.000000, 2610.250794, 491.904992,
            0.000000, 0.000000, 1.000000
                ),
                const cv::Mat& dist_coeffs = (cv::Mat_<double>(1, 5) << 0.392756, -0.798889, -0.005208, 0.000421, 0.000000
                ),
                float width = 0.12f, float height = 0.12f, float depth = 0.12f) ;


        // 输入目标四角点的像素坐标，返回 [x, y, z, vx, vy, vz]，用于得到平滑的相机坐标系位置。
        cv::Mat predict_position(const std::vector<cv::Point2f>& image_points, double dt = 0.1);
    };
    


#endif // PNP_KALMAN_H