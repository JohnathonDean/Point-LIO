#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <cmath>
#include <math.h>
#include <csignal>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>

#include "common_lib.h"

#define MAX_INI_COUNT (100)

class ImuProcess {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    // 重置 IMU 初始化阶段的统计量与状态机标志。
    // 一般在处理第一帧时或外部要求重新初始化时调用。
    void Reset();

    // 处理一组已经完成时间对齐的 LiDAR + IMU 测量。
    // 当前实现的核心职责是：
    // 1. 在系统启动早期累计 IMU 均值，用于静止初始化；
    // 2. 在初始化完成前阻止后续流程进入正式建图；
    // 3. 初始化完成后直接将输入点云透传到输出点云。
    // 注意：虽然接口名和上层注释中常提到“去畸变”，但这份实现里并未真正执行点云运动补偿。
    void Process(const MeasureGroup &meas, PointCloudXYZI::Ptr pcl_un_);

    // 设置陀螺仪噪声缩放参数。当前文件内仅保存该参数，未在本类中实际参与计算。
    void set_gyr_cov(const V3D &scaler);

    // 设置加速度相关噪声缩放参数。变量名沿用了历史命名，这里实际写入 cov_vel_scale。
    void set_acc_cov(const V3D &scaler);

    // 根据当前重力估计 gravity_ 与目标重力方向 tmp_gravity，计算二者对齐所需旋转 rot。
    // 该函数只负责求旋转，不在本类中直接修改状态。
    void Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot);

    // 初始化阶段使用的 12x12 状态协方差占位矩阵。
    // 当前类中仅在构造时置为单位阵，后续未直接参与 Process 流程。
    MD(12, 12) state_cov = MD(12, 12)::Identity();

    // 当前重力向量估计值，供 Set_init 计算姿态对齐关系使用。
    V3D gravity_;

    // 是否启用 IMU 处理。
    // 为 false 时，Process 直接透传 LiDAR 点云。
    bool imu_en;

    // 初始化阶段累计得到的加速度均值。
    // 静止假设下，其方向可近似看作重力方向的反向。
    V3D mean_acc;

    // 是否仍处于 IMU 初始化阶段。
    // true: 继续累计均值并阻塞后续依赖 IMU 初始化的流程。
    // false: 认为初始化完成，可以进入正常处理。
    bool imu_need_init_ = true;

    // 是否已经跨过“初始化刚结束”的边界。
    // 上层会用该标志判断是否可以正式进入激光更新流程。
    bool after_imu_init_ = false;

    // 是否为进入 IMU 初始化逻辑后的第一帧。
    // 第一帧会触发 Reset，并用首个 IMU 测量初始化均值。
    bool b_first_frame_ = true;

    // 记录上一帧扫描时间戳，当前文件中仅在 Reset 时清零，暂未进一步使用。
    double time_last_scan = 0.0;

    // 陀螺仪协方差缩放系数，占位保留。
    V3D cov_gyr_scale = V3D(0.0001, 0.0001, 0.0001);

    // 加速度/速度相关协方差缩放系数，占位保留。
    V3D cov_vel_scale = V3D(0.0001, 0.0001, 0.0001);

private:
    // 在静止初始化阶段累计 IMU 加速度与角速度均值。
    // N 作为累计样本计数器，以在线均值公式逐步更新 mean_acc / mean_gyr。
    void IMU_init(const MeasureGroup &meas, int &N);

    // 初始化阶段累计得到的角速度均值。
    // 静止假设下可近似反映陀螺零偏趋势。
    V3D mean_gyr;

    // IMU 初始化累计样本数。
    // 这里按“处理过的 IMU 样本数量”递增，而不是按 LiDAR 帧数递增。
    int init_iter_num = 1;
};


#endif // IMU_PROCESSING_H
