#include "IMU_Processing.h"

#include "so3_math.h"

// 构造函数只做轻量级状态初始化，不执行任何外部依赖操作。
// 默认进入“需要 IMU 初始化”的状态，等待后续 Process 累计前几批 IMU 数据。
ImuProcess::ImuProcess(): b_first_frame_(true), imu_need_init_(true) {
    imu_en = true;
    init_iter_num = 1;
    mean_acc = V3D(0, 0, 0.0);
    mean_gyr = V3D(0, 0, 0);
    after_imu_init_ = false;
    state_cov.setIdentity();
}

ImuProcess::~ImuProcess() {}

// 将 IMU 初始化相关统计量恢复到初始状态。
// 注意这里不会修改 b_first_frame_，因为谁来认定“下一次进入初始化的是首帧”
// 由外层初始化流程配合控制。
void ImuProcess::Reset() {
    ROS_WARN("Reset ImuProcess");
    mean_acc = V3D(0, 0, 0.0);
    mean_gyr = V3D(0, 0, 0);
    imu_need_init_ = true;
    init_iter_num = 1;
    after_imu_init_ = false;

    time_last_scan = 0.0;
}

void ImuProcess::set_gyr_cov(const V3D& scaler) { cov_gyr_scale = scaler; }

void ImuProcess::set_acc_cov(const V3D& scaler) { cov_vel_scale = scaler; }

// 计算一个旋转矩阵 rot，使当前重力估计 gravity_ 尽量对齐到 tmp_gravity。
// 常见用途是：根据静止期估计到的重力方向，求出初始姿态的校正量。
void ImuProcess::Set_init(Eigen::Vector3d& tmp_gravity, Eigen::Matrix3d& rot) {
    // hat_grav 相当于 gravity_ 对应的反对称矩阵，用于构造叉乘运算。
    // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
    M3D hat_grav;
    hat_grav << 0.0, gravity_(2), -gravity_(1), -gravity_(2), 0.0, gravity_(0), gravity_(1), -gravity_(0),
        0.0;

    // 用叉积模长衡量两向量的不平行程度，用点积衡量夹角余弦。
    double align_norm = (hat_grav * tmp_gravity).norm() / gravity_.norm() / tmp_gravity.norm();
    double align_cos = gravity_.transpose() * tmp_gravity;
    align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();

    // 若两向量几乎平行，则旋转只有“同向”和“反向”两种退化情况：
    // 1. 同向：无需旋转；
    // 2. 反向：直接取 -I 作为结果。
    // 注意：严格从旋转群 SO(3) 看，-I 并不是合法旋转矩阵；
    // 这里保留原实现语义，仅在注释中说明这一点。
    if (align_norm < 1e-6) {
        if (align_cos > 1e-6) {
            rot = Eye3d;
        } else {
            rot = -Eye3d;
        }
    } else {
        // 一般情况下，旋转轴由 gravity_ x tmp_gravity 的方向给出，
        // 旋转角由两者夹角 acos(align_cos) 给出，再通过指数映射转为旋转矩阵。
        V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos);
        rot = Exp(align_angle(0), align_angle(1), align_angle(2));
    }
}

// 在系统启动初期统计 IMU 的加速度与角速度均值。
// 这里默认传感器在初始化窗口内基本静止，因此：
// 1. 加速度均值可用于估计重力方向；
// 2. 角速度均值可用于估计陀螺零偏趋势。
void ImuProcess::IMU_init(const MeasureGroup& meas, int& N) {
    ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
    V3D cur_acc, cur_gyr;

    if (b_first_frame_) {
        // 第一次进入初始化流程时先清空旧状态，避免重复启动时残留历史统计量。
        Reset();

        // 将样本计数器置 1，并用第一帧中的第一个 IMU 测量初始化均值。
        // 这样后面的在线均值公式可以直接工作。
        N = 1;
        b_first_frame_ = false;
        const auto& imu_acc = meas.imu.front()->linear_acceleration;
        const auto& gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    }

    for (const auto& imu: meas.imu) {
        const auto& imu_acc = imu->linear_acceleration;
        const auto& gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        // 在线均值更新：
        // new_mean = old_mean + (sample - old_mean) / N
        // 这里 N 是当前样本纳入前的累计计数，因此更新后再自增。
        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        N++;
    }
}

// 处理一组时间同步后的传感器数据。
// 当前版本流程比较简单：
// 1. 若未启用 IMU，则直接透传 LiDAR 点云；
// 2. 若启用 IMU 但还未完成初始化，则只累计 IMU 统计量，并暂不放行正常处理；
// 3. 初始化完成后，将输入 LiDAR 点云原样拷贝到输出。
void ImuProcess::Process(const MeasureGroup& meas, PointCloudXYZI::Ptr cur_pcl_un_) {
    if (imu_en) {
        // 开启 IMU 模式时，若当前测量组没有 IMU 数据，则无法做初始化统计，直接返回。
        if (meas.imu.empty()) return;

        if (imu_need_init_) {
            {
                // 系统刚启动时，用前 MAX_INI_COUNT 个左右的 IMU 样本建立静止均值估计。
                IMU_init(meas, init_iter_num);

                // 保持在初始化状态，直到累计样本数达到阈值。
                imu_need_init_ = true;

                if (init_iter_num > MAX_INI_COUNT) {
                    // 初始化完成后，当前帧点云开始被放行给后续模块。
                    ROS_INFO("IMU Initializing: %.1f %%", 100.0);
                    imu_need_init_ = false;
                    *cur_pcl_un_ = *(meas.lidar);
                }
            }
            return;
        }

        // 第一次进入正常处理分支时拉起标志，供上层判断“IMU 已可用”。
        if (!after_imu_init_) after_imu_init_ = true;

        // 当前实现并未执行去畸变或预积分，只是简单透传点云。
        *cur_pcl_un_ = *(meas.lidar);
        return;
    }

    // 未启用 IMU 时，完全跳过初始化和补偿逻辑，直接使用原始 LiDAR 点云。
    *cur_pcl_un_ = *(meas.lidar);
    return;
}
