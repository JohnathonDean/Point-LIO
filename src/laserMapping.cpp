// #include <so3_math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include "LaserMapping.h"
#include <malloc.h>
// #include <cv_bridge/cv_bridge.h>
// #include "matplotlibcpp.h"
// #include <ros/console.h>

using namespace std;

#define PUBFRAME_PERIOD     (20)

const float MOV_THRESHOLD = 1.5f;

LaserMapping *LaserMapping::active_instance_ = nullptr;

LaserMapping &LaserMapping::activeInstance()
{
    return *active_instance_;
}

void LaserMapping::handleSignal(int sig)
{
    if (active_instance_ != nullptr) {
        active_instance_->exit_requested_.store(true);
    }
    ROS_WARN("catch sig %d", sig);
}

Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
    Eigen::Matrix<double, 24, 24> cov;
    cov.setZero();
    cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
    cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
    cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
    cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
    return cov;
}

Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
    Eigen::Matrix<double, 30, 30> cov;
    cov.setZero();
    cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
    cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
    cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
    cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
    cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
    return cov;
}

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    vect3 a_inertial = s.rot * (in.acc - s.ba);
    for (int i = 0; i < 3; i++) {
        res(i) = s.vel[i];
        res(i + 3) = omega[i];
        res(i + 12) = a_inertial[i] + s.gravity[i];
    }
    return res;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
    vect3 a_inertial = s.rot * s.acc;
    for (int i = 0; i < 3; i++) {
        res(i) = s.vel[i];
        res(i + 3) = s.omg[i];
        res(i + 12) = a_inertial[i] + s.gravity[i];
    }
    return res;
}

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
    cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
    vect3 acc_;
    in.acc.boxminus(acc_, s.ba);
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    cov.template block<3, 3>(12, 3) = -s.rot * MTK::hat(acc_);
    cov.template block<3, 3>(12, 18) = -s.rot;
    cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
    return cov;
}

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
    cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(12, 3) = -s.rot * MTK::hat(s.acc);
    cov.template block<3, 3>(12, 18) = s.rot;
    cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity();
    return cov;
}

void LaserMapping::hModelInput(
    state_input &s,
    Eigen::Matrix3d cov_p,
    Eigen::Matrix3d cov_R,
    esekfom::dyn_share_modified<double> &ekfom_data)
{
    auto &mapping = LaserMapping::activeInstance();
    VF(4) pabcd;
    pabcd.setZero();
    mapping.normvec_->resize(mapping.time_seq_[mapping.k_]);
    int effect_num_k = 0;
    for (int j = 0; j < mapping.time_seq_[mapping.k_]; j++) {
        PointType &point_body_j = mapping.feats_down_body_->points[mapping.idx_ + j + 1];
        PointType &point_world_j = mapping.feats_down_world_->points[mapping.idx_ + j + 1];
        mapping.pointBodyToWorld(&point_body_j, &point_world_j);
        V3D p_body = mapping.pbody_list_[mapping.idx_ + j + 1];
        double p_norm = p_body.norm();
        auto &points_near = mapping.nearest_points_[mapping.idx_ + j + 1];
        mapping.ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);
        if ((points_near.size() < NUM_MATCH_POINTS)) {
            mapping.point_selected_surf_[mapping.idx_ + j + 1] = false;
        } else {
            mapping.point_selected_surf_[mapping.idx_ + j + 1] = false;
            if (esti_plane(pabcd, points_near, plane_thr)) {
                float pd2 = fabs(
                    pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y +
                    pabcd(2) * point_world_j.z + pabcd(3));
                if (p_norm > match_s * pd2 * pd2) {
                    mapping.point_selected_surf_[mapping.idx_ + j + 1] = true;
                    mapping.normvec_->points[j].x = pabcd(0);
                    mapping.normvec_->points[j].y = pabcd(1);
                    mapping.normvec_->points[j].z = pabcd(2);
                    mapping.normvec_->points[j].intensity = pabcd(3);
                    effect_num_k ++;
                }
            }
        }
    }
    if (effect_num_k == 0) {
        ekfom_data.valid = false;
        return;
    }
    ekfom_data.M_Noise = laser_point_cov;
    ekfom_data.h_x.resize(effect_num_k, 12);
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
    ekfom_data.z.resize(effect_num_k);
    int m = 0;

    for (int j = 0; j < mapping.time_seq_[mapping.k_]; j++) {
        if (mapping.point_selected_surf_[mapping.idx_ + j + 1]) {
            V3D norm_vec(mapping.normvec_->points[j].x, mapping.normvec_->points[j].y, mapping.normvec_->points[j].z);

            if (extrinsic_est_en) {
                V3D p_body = mapping.pbody_list_[mapping.idx_ + j + 1];
                M3D p_crossmat, p_imu_crossmat;
                p_crossmat << SKEW_SYM_MATRX(p_body);
                V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
                p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(p_imu_crossmat * C);
                V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
                ekfom_data.h_x.block<1, 12>(m, 0)
                    << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            } else {
                M3D point_crossmat = mapping.crossmat_list_[mapping.idx_ + j + 1];
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(point_crossmat * C);
                ekfom_data.h_x.block<1, 12>(m, 0)
                    << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }
            ekfom_data.z(m) =
                -norm_vec(0) * mapping.feats_down_world_->points[mapping.idx_ + j + 1].x
                -norm_vec(1) * mapping.feats_down_world_->points[mapping.idx_ + j + 1].y
                -norm_vec(2) * mapping.feats_down_world_->points[mapping.idx_ + j + 1].z
                -mapping.normvec_->points[j].intensity;
            m++;
        }
    }
    mapping.effct_feat_num_ += effect_num_k;
}

void LaserMapping::hModelOutput(
    state_output &s,
    Eigen::Matrix3d cov_p,
    Eigen::Matrix3d cov_R,
    esekfom::dyn_share_modified<double> &ekfom_data)
{
    auto &mapping = LaserMapping::activeInstance();
    VF(4) pabcd;
    pabcd.setZero();
    mapping.normvec_->resize(mapping.time_seq_[mapping.k_]);
    int effect_num_k = 0;
    for (int j = 0; j < mapping.time_seq_[mapping.k_]; j++) {
        PointType &point_body_j = mapping.feats_down_body_->points[mapping.idx_ + j + 1];
        PointType &point_world_j = mapping.feats_down_world_->points[mapping.idx_ + j + 1];
        mapping.pointBodyToWorld(&point_body_j, &point_world_j);
        V3D p_body = mapping.pbody_list_[mapping.idx_ + j + 1];
        double p_norm = p_body.norm();
        auto &points_near = mapping.nearest_points_[mapping.idx_ + j + 1];

        mapping.ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);

        if ((points_near.size() < NUM_MATCH_POINTS)) {
            mapping.point_selected_surf_[mapping.idx_ + j + 1] = false;
        } else {
            mapping.point_selected_surf_[mapping.idx_ + j + 1] = false;
            if (esti_plane(pabcd, points_near, plane_thr)) {
                float pd2 = fabs(
                    pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y +
                    pabcd(2) * point_world_j.z + pabcd(3));
                if (p_norm > match_s * pd2 * pd2) {
                    mapping.point_selected_surf_[mapping.idx_ + j + 1] = true;
                    mapping.normvec_->points[j].x = pabcd(0);
                    mapping.normvec_->points[j].y = pabcd(1);
                    mapping.normvec_->points[j].z = pabcd(2);
                    mapping.normvec_->points[j].intensity = pabcd(3);
                    effect_num_k ++;
                }
            }
        }
    }
    if (effect_num_k == 0) {
        ekfom_data.valid = false;
        return;
    }
    ekfom_data.M_Noise = laser_point_cov;
    ekfom_data.h_x.resize(effect_num_k, 12);
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
    ekfom_data.z.resize(effect_num_k);
    int m = 0;
    for (int j = 0; j < mapping.time_seq_[mapping.k_]; j++) {
        if (mapping.point_selected_surf_[mapping.idx_ + j + 1]) {
            V3D norm_vec(mapping.normvec_->points[j].x, mapping.normvec_->points[j].y, mapping.normvec_->points[j].z);
            if (extrinsic_est_en) {
                V3D p_body = mapping.pbody_list_[mapping.idx_ + j + 1];
                M3D p_crossmat, p_imu_crossmat;
                p_crossmat << SKEW_SYM_MATRX(p_body);
                V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
                p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(p_imu_crossmat * C);
                V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
                ekfom_data.h_x.block<1, 12>(m, 0)
                    << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            } else {
                M3D point_crossmat = mapping.crossmat_list_[mapping.idx_ + j + 1];
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(point_crossmat * C);
                ekfom_data.h_x.block<1, 12>(m, 0)
                    << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }
            ekfom_data.z(m) =
                -norm_vec(0) * mapping.feats_down_world_->points[mapping.idx_ + j + 1].x
                -norm_vec(1) * mapping.feats_down_world_->points[mapping.idx_ + j + 1].y
                -norm_vec(2) * mapping.feats_down_world_->points[mapping.idx_ + j + 1].z
                -mapping.normvec_->points[j].intensity;

            m++;
        }
    }
    mapping.effct_feat_num_ += effect_num_k;
}

void LaserMapping::hModelImuOutput(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    auto &mapping = LaserMapping::activeInstance();
    std::memset(ekfom_data.satu_check, false, 6);
    ekfom_data.z_IMU.block<3, 1>(0, 0) = mapping.angvel_avr_ - s.omg - s.bg;
    ekfom_data.z_IMU.block<3, 1>(3, 0) = mapping.acc_avr_ * mapping.g_m_s2_ / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
    if (check_satu) {
        if (fabs(mapping.angvel_avr_(0)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[0] = true;
            ekfom_data.z_IMU(0) = 0.0;
        }
        if (fabs(mapping.angvel_avr_(1)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[1] = true;
            ekfom_data.z_IMU(1) = 0.0;
        }
        if (fabs(mapping.angvel_avr_(2)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[2] = true;
            ekfom_data.z_IMU(2) = 0.0;
        }
        if (fabs(mapping.acc_avr_(0)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[3] = true;
            ekfom_data.z_IMU(3) = 0.0;
        }
        if (fabs(mapping.acc_avr_(1)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[4] = true;
            ekfom_data.z_IMU(4) = 0.0;
        }
        if (fabs(mapping.acc_avr_(2)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[5] = true;
            ekfom_data.z_IMU(5) = 0.0;
        }
    }
}

void LaserMapping::pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global;
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            p_global = kf_output_.x_.rot * (kf_output_.x_.offset_R_L_I * p_body + kf_output_.x_.offset_T_L_I) + kf_output_.x_.pos;
        } else {
            p_global = kf_input_.x_.rot * (kf_input_.x_.offset_R_L_I * p_body + kf_input_.x_.offset_T_L_I) + kf_input_.x_.pos;
        }
    } else {
        if (!use_imu_as_input) {
            p_global = kf_output_.x_.rot * (lidar_r_wrt_imu_ * p_body + lidar_t_wrt_imu_) + kf_output_.x_.pos;
        } else {
            p_global = kf_input_.x_.rot * (lidar_r_wrt_imu_ * p_body + lidar_t_wrt_imu_) + kf_input_.x_.pos;
        }
    }

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::dumpLioStateToLog(FILE *fp)
{
    V3D rot_ang;
    if (!use_imu_as_input) {
        rot_ang = SO3ToEuler(kf_output_.x_.rot);
    } else {
        rot_ang = SO3ToEuler(kf_input_.x_.rot);
    }

    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input) {
        fprintf(fp, "%lf %lf %lf ", kf_input_.x_.pos(0), kf_input_.x_.pos(1), kf_input_.x_.pos(2)); // Pos
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
        fprintf(fp, "%lf %lf %lf ", kf_input_.x_.vel(0), kf_input_.x_.vel(1), kf_input_.x_.vel(2)); // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
        fprintf(fp, "%lf %lf %lf ", kf_input_.x_.bg(0), kf_input_.x_.bg(1), kf_input_.x_.bg(2));    // Bias_g
        fprintf(fp, "%lf %lf %lf ", kf_input_.x_.ba(0), kf_input_.x_.ba(1), kf_input_.x_.ba(2));    // Bias_a
        fprintf(fp, "%lf %lf %lf ", kf_input_.x_.gravity(0), kf_input_.x_.gravity(1), kf_input_.x_.gravity(2)); // Bias_a
    } else {
        fprintf(fp, "%lf %lf %lf ", kf_output_.x_.pos(0), kf_output_.x_.pos(1), kf_output_.x_.pos(2)); // Pos
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
        fprintf(fp, "%lf %lf %lf ", kf_output_.x_.vel(0), kf_output_.x_.vel(1), kf_output_.x_.vel(2)); // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
        fprintf(fp, "%lf %lf %lf ", kf_output_.x_.bg(0), kf_output_.x_.bg(1), kf_output_.x_.bg(2));    // Bias_g
        fprintf(fp, "%lf %lf %lf ", kf_output_.x_.ba(0), kf_output_.x_.ba(1), kf_output_.x_.ba(2));    // Bias_a
        fprintf(fp, "%lf %lf %lf ", kf_output_.x_.gravity(0), kf_output_.x_.gravity(1), kf_output_.x_.gravity(2)); // Bias_a
    }
    fprintf(fp, "\r\n");
    fflush(fp);
}

void LaserMapping::pointBodyLidarToImu(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            p_body_imu = kf_output_.x_.offset_R_L_I * p_body_lidar + kf_output_.x_.offset_T_L_I;
        } else {
            p_body_imu = kf_input_.x_.offset_R_L_I * p_body_lidar + kf_input_.x_.offset_T_L_I;
        }
    } else {
        p_body_imu = lidar_r_wrt_imu_ * p_body_lidar + lidar_t_wrt_imu_;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LaserMapping::mapIncremental()
{
    PointVector points_to_add;
    int cur_pts = feats_down_world_->size();
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i) {
        /* decide if need add to map */
        PointType &point_world = feats_down_world_->points[i];
        if (!nearest_points_[i].empty()) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;
            bool need_add = true;
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
                Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
                if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.z()) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    }
    ivox_->AddPoints(points_to_add);
}

void LaserMapping::publishInitMap(const ros::Publisher &pubLaserCloudFullRes)
{
    int size_init_map = init_feats_world_->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*init_feats_world_, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}

void LaserMapping::publishFrameWorld(const ros::Publisher &pubLaserCloudFullRes)
{
    if (scan_pub_en) {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body_);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = feats_down_world_->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world_->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world_->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world_->points[i].intensity;
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        // publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en) {
        int size = feats_down_world_->points.size();
        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = feats_down_world_->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world_->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world_->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world_->points[i].intensity;
        }

        *pcl_wait_save_ += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save_->size() > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
            pcl_wait_save_->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::publishFrameBody(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort_->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
        pointBodyLidarToImu(&feats_undistort_->points[i], &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

void LaserMapping::publishOdometry(const ros::Publisher &pubOdomAftMapped)
{
    odom_aft_mapped_.header.frame_id = "camera_init";
    odom_aft_mapped_.child_frame_id = "body";
    if (publish_odometry_without_downsample) {
        odom_aft_mapped_.header.stamp = ros::Time().fromSec(time_current);
    } else {
        odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time);
    }
    setPoseStamp(odom_aft_mapped_.pose.pose);

    pubOdomAftMapped.publish(odom_aft_mapped_);

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x,
                                    odom_aft_mapped_.pose.pose.position.y,
                                    odom_aft_mapped_.pose.pose.position.z));
    q.setW(odom_aft_mapped_.pose.pose.orientation.w);
    q.setX(odom_aft_mapped_.pose.pose.orientation.x);
    q.setY(odom_aft_mapped_.pose.pose.orientation.y);
    q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odom_aft_mapped_.header.stamp, "camera_init", "body") );
}

void LaserMapping::publishPath(const ros::Publisher &pubPath)
{
    setPoseStamp(msg_body_pose_.pose);
    // msg_body_pose_.header.stamp = ros::Time::now();
    msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose_.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path_.poses.emplace_back(msg_body_pose_);
        pubPath.publish(path_);
    }
}

void LaserMapping::initialize()
{
    readParameters(nh_);
    cout<<"lidar_type: "<<lidar_type<<endl;
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    path_.header.stamp    = ros::Time().fromSec(lidar_end_time);
    path_.header.frame_id ="camera_init";

    memset(point_selected_surf_, true, sizeof(point_selected_surf_));
    down_size_filter_surf_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    down_size_filter_map_.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    lidar_t_wrt_imu_ << VEC_FROM_ARRAY(extrinT);
    lidar_r_wrt_imu_ << MAT_FROM_ARRAY(extrinR);

    if (extrinsic_est_en) {
        kf_output_.x_.offset_R_L_I = lidar_r_wrt_imu_;
        kf_output_.x_.offset_T_L_I = lidar_t_wrt_imu_;
        kf_input_.x_.offset_R_L_I = lidar_r_wrt_imu_;
        kf_input_.x_.offset_T_L_I = lidar_t_wrt_imu_;
    }

    kf_input_.init_dyn_share_modified_2h(get_f_input, df_dx_input, LaserMapping::hModelInput);
    reset_cov(p_init_);
    kf_input_.change_P(p_init_);
    q_input_ = process_noise_cov_input();

    kf_output_.init_dyn_share_modified_3h(get_f_output, df_dx_output, LaserMapping::hModelOutput, LaserMapping::hModelImuOutput);
    reset_cov_output(p_init_output_);
    kf_output_.change_P(p_init_output_);
    q_output_ = process_noise_cov_output();

    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    /*** debug record ***/
    string pos_log_dir = root_dir_ + "/Log/pos_log.txt";
    fp_ = fopen(pos_log_dir.c_str(),"w");
    open_file();

    /*** ROS subscribe initialization ***/
    sub_pcl_ = p_pre->lidar_type == AVIA ?
        nh_.subscribe(lid_topic, 200000, &LaserMapping::livoxPclCallback, this) :
        nh_.subscribe(lid_topic, 200000, &LaserMapping::standardPclCallback, this);
    sub_imu_ = nh_.subscribe(imu_topic, 200000, &LaserMapping::imuCallback, this);

    pub_laser_cloud_full_res_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 1000);
    pub_laser_cloud_full_res_body_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 1000);
    // ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            // ("/cloud_effected", 1000);
    pub_laser_cloud_map_ = nh_.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1000);
    pub_odom_aft_mapped_ = nh_.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 1000);
    pub_path_ = nh_.advertise<nav_msgs::Path>("/path", 1000);
    // ros::Publisher plane_pub = nh.advertise<visualization_msgs::Marker>
            // ("/planner_normal", 1000);
}

void LaserMapping::cleanup()
{
    if (pcl_wait_save_->size() > 0 && pcd_save_en) {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }
    fout_out.close();
    fout_imu_pbp.close();
    if (fp_ != nullptr) {
        fclose(fp_);
        fp_ = nullptr;
    }
}

void LaserMapping::handleReset()
{
    ROS_WARN("reset when rosbag play back");
    p_imu->Reset();
    feats_undistort_.reset(new PointCloudXYZI());
    if (use_imu_as_input) {
        state_in = state_input();
        kf_input_.change_P(p_init_);
    } else {
        state_out = state_output();
        kf_output_.change_P(p_init_output_);
    }
    first_scan_ = true;
    is_first_frame = true;
    reset_requested_ = false;
    init_map_ = false;

    ivox_.reset(new IVoxType(ivox_options_));
}

void LaserMapping::handleFirstScan()
{
    first_lidar_time = Measures.lidar_beg_time;
    first_scan_ = false;
    time_current = 0.0;
    if (imu_en) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (first_imu_time < 1) {
            first_imu_time = imu_next_.header.stamp.toSec();
            printf("first imu time: %f\n", first_imu_time);
        }
        kf_input_.x_.gravity << VEC_FROM_ARRAY(gravity);
        kf_output_.x_.gravity << VEC_FROM_ARRAY(gravity);

        while (Measures.lidar_beg_time > imu_next_.header.stamp.toSec()) {
            imu_deque_.pop_front();
            if (imu_deque_.empty()) {
                break;
            }
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
        }
    } else {
        kf_input_.x_.gravity << VEC_FROM_ARRAY(gravity);
        kf_output_.x_.gravity << VEC_FROM_ARRAY(gravity);
        kf_output_.x_.acc << VEC_FROM_ARRAY(gravity);
        kf_output_.x_.acc *= -1;
        p_imu->imu_need_init_ = false;
    }
    g_m_s2_ = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
}

void LaserMapping::prepareMeasurement()
{
    p_imu->Process(Measures, feats_undistort_);
    if (space_down_sample) {
        down_size_filter_surf_.setInputCloud(feats_undistort_);
        down_size_filter_surf_.filter(*feats_down_body_);
        sort(feats_down_body_->points.begin(), feats_down_body_->points.end(), time_list);
    } else {
        feats_down_body_ = Measures.lidar;
        sort(feats_down_body_->points.begin(), feats_down_body_->points.end(), time_list);
    }

    time_seq_ = time_compressing<int>(feats_down_body_);
    feats_down_size_ = feats_down_body_->points.size();
}

bool LaserMapping::ensureImuInitialized()
{
    if (p_imu->after_imu_init_) {
        return true;
    }

    if (p_imu->imu_need_init_) {
        return false;
    }

    V3D tmp_gravity;
    if (imu_en) {
        tmp_gravity = -p_imu->mean_acc / p_imu->mean_acc.norm() * g_m_s2_;
    } else {
        tmp_gravity << VEC_FROM_ARRAY(gravity_init);
        p_imu->after_imu_init_ = true;
    }

    M3D rot_init;
    p_imu->Set_init(tmp_gravity, rot_init);
    kf_input_.x_.rot = rot_init;
    kf_output_.x_.rot = rot_init;
    kf_output_.x_.acc = -rot_init.transpose() * kf_output_.x_.gravity;
    return true;
}

bool LaserMapping::initializeMapIfNeeded()
{
    if (init_map_) {
        return false;
    }

    feats_down_world_->resize(feats_undistort_->size());
    for (int i = 0; i < feats_undistort_->size(); i++) {
        pointBodyToWorld(&(feats_undistort_->points[i]), &(feats_down_world_->points[i]));
    }
    for (size_t i = 0; i < feats_down_world_->size(); i++) {
        init_feats_world_->points.emplace_back(feats_down_world_->points[i]);
    }
    if (init_feats_world_->size() < init_map_size) {
        init_map_ = false;
    } else {
        ivox_->AddPoints(init_feats_world_->points);
        publishInitMap(pub_laser_cloud_map_);

        init_feats_world_.reset(new PointCloudXYZI());
        init_map_ = true;
    }
    return true;
}

void LaserMapping::prepareStateEstimation()
{
    normvec_->resize(feats_down_size_);
    feats_down_world_->resize(feats_down_size_);
    nearest_points_.resize(feats_down_size_);

    crossmat_list_.reserve(feats_down_size_);
    pbody_list_.reserve(feats_down_size_);

    for (size_t i = 0; i < feats_down_body_->size(); i++) {
        V3D point_this(
            feats_down_body_->points[i].x,
            feats_down_body_->points[i].y,
            feats_down_body_->points[i].z);
        pbody_list_[i] = point_this;
        if (!extrinsic_est_en) {
            point_this = lidar_r_wrt_imu_ * point_this + lidar_t_wrt_imu_;
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);
            crossmat_list_[i] = point_crossmat;
        }
    }
}

void LaserMapping::standardPclCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    double preprocess_start_time = omp_get_wtime();
    int current_scan_count = 0;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        scan_count_ ++;
        current_scan_count = scan_count_;
        if (msg->header.stamp.toSec() < last_timestamp_lidar) {
            ROS_ERROR("lidar loop back, clear buffer");
            return;
        }
        last_timestamp_lidar = msg->header.stamp.toSec();
    }

    if ((lidar_type == VELO16 || lidar_type == OUST64 || lidar_type == HESAIxt32) && cut_frame_init) {
        deque<PointCloudXYZI::Ptr> ptr;
        deque<double> timestamp_lidar;
        p_pre->process_cut_frame_pcl2(msg, ptr, timestamp_lidar, cut_frame_num, current_scan_count);

        std::lock_guard<std::mutex> lock(buffer_mutex_);
        while (!ptr.empty() && !timestamp_lidar.empty()) {
            lidar_buffer_.push_back(ptr.front());
            ptr.pop_front();
            time_buffer_.push_back(timestamp_lidar.front() / double(1000));
            timestamp_lidar.pop_front();
        }
        s_plot11_[current_scan_count] = omp_get_wtime() - preprocess_start_time;
        buffer_cv_.notify_all();
        return;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI(20000, 1));
    p_pre->process(msg, ptr);
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (con_frame) {
        if (frame_ct_ == 0) {
            time_con = last_timestamp_lidar;
        }
        if (frame_ct_ < 10) {
            for (int i = 0; i < ptr->size(); i++) {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con_->push_back(ptr->points[i]);
            }
            frame_ct_ ++;
        } else {
            PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI(10000, 1));
            *ptr_con_i = *ptr_con_;
            lidar_buffer_.push_back(ptr_con_i);
            time_buffer_.push_back(time_con);
            ptr_con_->clear();
            frame_ct_ = 0;
        }
    } else if (ptr->points.size() > 0) {
        lidar_buffer_.emplace_back(ptr);
        time_buffer_.emplace_back(msg->header.stamp.toSec());
    }
    s_plot11_[current_scan_count] = omp_get_wtime() - preprocess_start_time;
    buffer_cv_.notify_all();
}

void LaserMapping::livoxPclCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    double preprocess_start_time = omp_get_wtime();
    int current_scan_count = 0;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        scan_count_ ++;
        current_scan_count = scan_count_;
        if (msg->header.stamp.toSec() < last_timestamp_lidar) {
            ROS_ERROR("lidar loop back, clear buffer");
            return;
        }
        last_timestamp_lidar = msg->header.stamp.toSec();
    }

    if (cut_frame_init) {
        deque<PointCloudXYZI::Ptr> ptr;
        deque<double> timestamp_lidar;
        p_pre->process_cut_frame_livox(msg, ptr, timestamp_lidar, cut_frame_num, current_scan_count);

        std::lock_guard<std::mutex> lock(buffer_mutex_);
        while (!ptr.empty() && !timestamp_lidar.empty()) {
            lidar_buffer_.push_back(ptr.front());
            ptr.pop_front();
            time_buffer_.push_back(timestamp_lidar.front() / double(1000));
            timestamp_lidar.pop_front();
        }
        s_plot11_[current_scan_count] = omp_get_wtime() - preprocess_start_time;
        buffer_cv_.notify_all();
        return;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI(10000, 1));
    p_pre->process(msg, ptr);
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (con_frame) {
        if (frame_ct_ == 0) {
            time_con = last_timestamp_lidar;
        }
        if (frame_ct_ < 10) {
            for (int i = 0; i < ptr->size(); i++) {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con_->push_back(ptr->points[i]);
            }
            frame_ct_ ++;
        } else {
            PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI(10000, 1));
            *ptr_con_i = *ptr_con_;
            lidar_buffer_.push_back(ptr_con_i);
            time_buffer_.push_back(time_con);
            ptr_con_->clear();
            frame_ct_ = 0;
        }
    } else if (ptr->points.size() > 0) {
        lidar_buffer_.emplace_back(ptr);
        time_buffer_.emplace_back(msg->header.stamp.toSec());
    }
    s_plot11_[current_scan_count] = omp_get_wtime() - preprocess_start_time;
    buffer_cv_.notify_all();
}

void LaserMapping::imuCallback(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    msg->header.stamp = ros::Time().fromSec(
        msg->header.stamp.toSec() - timediff_imu_wrt_lidar_ - time_lag_imu_wrt_lidar_);

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp_imu) {
        ROS_ERROR("imu loop back, clear deque");
        return;
    }

    imu_deque_.emplace_back(msg);
    last_timestamp_imu = timestamp;
    buffer_cv_.notify_all();
}

bool LaserMapping::syncPackages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (!imu_en) {
        if (!lidar_buffer_.empty()) {
            if (!lidar_pushed_) {
                meas.lidar = lidar_buffer_.front();
                meas.lidar_beg_time = time_buffer_.front();
                lose_lid_ = false;
                if (meas.lidar->points.size() < 1) {
                    cout << "lose lidar" << std::endl;
                    lose_lid_ = true;
                } else {
                    double end_time = meas.lidar->points.back().curvature;
                    for (auto pt : meas.lidar->points) {
                        if (pt.curvature > end_time) {
                            end_time = pt.curvature;
                        }
                    }
                    lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
                    meas.lidar_last_time = lidar_end_time;
                }
                lidar_pushed_ = true;
            }

            time_buffer_.pop_front();
            lidar_buffer_.pop_front();
            lidar_pushed_ = false;
            return !lose_lid_;
        }
        return false;
    }

    if (lidar_buffer_.empty() || imu_deque_.empty()) {
        return false;
    }

    if (!lidar_pushed_) {
        lose_lid_ = false;
        meas.lidar = lidar_buffer_.front();
        meas.lidar_beg_time = time_buffer_.front();
        if (meas.lidar->points.size() < 1) {
            cout << "lose lidar" << endl;
            lose_lid_ = true;
        } else {
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt : meas.lidar->points) {
                if (pt.curvature > end_time) {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            meas.lidar_last_time = lidar_end_time;
        }
        lidar_pushed_ = true;
    }

    if (!lose_lid_ && (last_timestamp_imu < lidar_end_time)) {
        return false;
    }
    if (lose_lid_ && last_timestamp_imu < meas.lidar_beg_time + lidar_time_inte) {
        return false;
    }

    if (!lose_lid_ && !imu_pushed_) {
        if (p_imu->imu_need_init_) {
            double imu_time = imu_deque_.front()->header.stamp.toSec();
            imu_next_ = *(imu_deque_.front());
            meas.imu.shrink_to_fit();
            while (imu_time < lidar_end_time) {
                meas.imu.emplace_back(imu_deque_.front());
                imu_last_ = imu_next_;
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_time = imu_deque_.front()->header.stamp.toSec();
                imu_next_ = *(imu_deque_.front());
            }
        }
        imu_pushed_ = true;
    }

    if (lose_lid_ && !imu_pushed_) {
        if (p_imu->imu_need_init_) {
            double imu_time = imu_deque_.front()->header.stamp.toSec();
            meas.imu.shrink_to_fit();
            imu_next_ = *(imu_deque_.front());
            while (imu_time < meas.lidar_beg_time + lidar_time_inte) {
                meas.imu.emplace_back(imu_deque_.front());
                imu_last_ = imu_next_;
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_time = imu_deque_.front()->header.stamp.toSec();
                imu_next_ = *(imu_deque_.front());
            }
        }
        imu_pushed_ = true;
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    imu_pushed_ = false;
    return true;
}

void LaserMapping::processOutputStateMode(V3D &euler_cur)
{
    effct_feat_num_ = 0;
    if (time_seq_.size() > 0) {
        processOutputStateWithTimeSequence(euler_cur);
    } else {
        processOutputStateWithoutTimeSequence();
    }
}

void LaserMapping::processInputStateMode(V3D &euler_cur)
{
    effct_feat_num_ = 0;
    if (time_seq_.size() > 0) {
        processInputStateWithTimeSequence(euler_cur);
    } else {
        processInputStateWithoutTimeSequence();
    }
}

void LaserMapping::processOutputStateWithTimeSequence(V3D &euler_cur)
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    bool imu_upda_cov = false;
    double pcl_beg_time = Measures.lidar_beg_time;
    idx_ = -1;
    for (k_ = 0; k_ < time_seq_.size(); k_++) {
        PointType &point_body = feats_down_body_->points[idx_ + time_seq_[k_]];

        time_current = point_body.curvature / 1000.0 + pcl_beg_time;

        if (is_first_frame) {
            if (imu_en) {
                while (time_current > imu_next_.header.stamp.toSec()) {
                    imu_deque_.pop_front();
                    if (imu_deque_.empty()) break;
                    imu_last_ = imu_next_;
                    imu_next_ = *(imu_deque_.front());
                }
                angvel_avr_ << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y, imu_last_.angular_velocity.z;
                acc_avr_ << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y, imu_last_.linear_acceleration.z;
            }
            is_first_frame = false;
            imu_upda_cov = true;
            time_update_last = time_current;
            time_predict_last_const = time_current;
        }
        if (imu_en && !imu_deque_.empty()) {
            bool last_imu = imu_next_.header.stamp.toSec() == imu_deque_.front()->header.stamp.toSec();
            while (imu_next_.header.stamp.toSec() < time_predict_last_const && !imu_deque_.empty()) {
                if (!last_imu) {
                    imu_last_ = imu_next_;
                    imu_next_ = *(imu_deque_.front());
                    break;
                } else {
                    imu_deque_.pop_front();
                    if (imu_deque_.empty()) break;
                    imu_last_ = imu_next_;
                    imu_next_ = *(imu_deque_.front());
                }
            }
            bool imu_comes = time_current > imu_next_.header.stamp.toSec();
            while (imu_comes) {
                imu_upda_cov = true;
                angvel_avr_ << imu_next_.angular_velocity.x, imu_next_.angular_velocity.y, imu_next_.angular_velocity.z;
                acc_avr_ << imu_next_.linear_acceleration.x, imu_next_.linear_acceleration.y, imu_next_.linear_acceleration.z;

                double dt = imu_next_.header.stamp.toSec() - time_predict_last_const;
                kf_output_.predict(dt, q_output_, input_in_, true, false);
                time_predict_last_const = imu_next_.header.stamp.toSec();

                {
                    double dt_cov = imu_next_.header.stamp.toSec() - time_update_last;

                    if (dt_cov > 0.0) {
                        time_update_last = imu_next_.header.stamp.toSec();
                        double propag_imu_start = omp_get_wtime();

                        kf_output_.predict(dt_cov, q_output_, input_in_, false, true);

                        propag_time_ += omp_get_wtime() - propag_imu_start;
                        double solve_imu_start = omp_get_wtime();
                        kf_output_.update_iterated_dyn_share_IMU();
                        solve_time_ += omp_get_wtime() - solve_imu_start;
                    }
                }
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_last_ = imu_next_;
                imu_next_ = *(imu_deque_.front());
                imu_comes = time_current > imu_next_.header.stamp.toSec();
            }
        }
        if (reset_requested_) {
            break;
        }

        double dt = time_current - time_predict_last_const;
        double propag_state_start = omp_get_wtime();
        if (!prop_at_freq_of_imu) {
            double dt_cov = time_current - time_update_last;
            if (dt_cov > 0.0) {
                kf_output_.predict(dt_cov, q_output_, input_in_, false, true);
                time_update_last = time_current;
            }
        }
        kf_output_.predict(dt, q_output_, input_in_, true, false);
        propag_time_ += omp_get_wtime() - propag_state_start;
        time_predict_last_const = time_current;
        double t_update_start = omp_get_wtime();

        if (feats_down_size_ < 1) {
            ROS_WARN("No point, skip this scan!\n");
            idx_ += time_seq_[k_];
            continue;
        }
        if (!kf_output_.update_iterated_dyn_share_modified()) {
            idx_ = idx_ + time_seq_[k_];
            continue;
        }
        double solve_start = omp_get_wtime();

        if (publish_odometry_without_downsample) {
            publishOdometry(pub_odom_aft_mapped_);
            if (runtime_pos_log) {
                euler_cur = SO3ToEuler(kf_output_.x_.rot);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output_.x_.pos.transpose() << " " << kf_output_.x_.vel.transpose() \
                <<" "<<kf_output_.x_.omg.transpose()<<" "<<kf_output_.x_.acc.transpose()<<" "<<kf_output_.x_.gravity.transpose()<<" "<<kf_output_.x_.bg.transpose()<<" "<<kf_output_.x_.ba.transpose()<<" "<<feats_undistort_->points.size()<<endl;
            }
        }

        for (int j = 0; j < time_seq_[k_]; j++) {
            PointType &point_body_j = feats_down_body_->points[idx_ + j + 1];
            PointType &point_world_j = feats_down_world_->points[idx_ + j + 1];
            pointBodyToWorld(&point_body_j, &point_world_j);
        }

        solve_time_ += omp_get_wtime() - solve_start;

        update_time_ += omp_get_wtime() - t_update_start;
        idx_ += time_seq_[k_];
    }
}

void LaserMapping::processOutputStateWithoutTimeSequence()
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    bool imu_upda_cov = false;
    if (!imu_deque_.empty()) {
        imu_last_ = imu_next_;
        imu_next_ = *(imu_deque_.front());

        while (imu_next_.header.stamp.toSec() > time_current && ((imu_next_.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte ))) {
            if (is_first_frame) {
                {
                    {
                        while (imu_next_.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte) {
                            imu_deque_.pop_front();
                            if (imu_deque_.empty()) break;
                            imu_last_ = imu_next_;
                            imu_next_ = *(imu_deque_.front());
                        }
                    }
                    break;
                }
                angvel_avr_ << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y, imu_last_.angular_velocity.z;

                acc_avr_ << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y, imu_last_.linear_acceleration.z;

                imu_upda_cov = true;
                time_update_last = time_current;
                time_predict_last_const = time_current;

                is_first_frame = false;
            }
            time_current = imu_next_.header.stamp.toSec();

            if (!is_first_frame) {
                double dt = time_current - time_predict_last_const;

                double dt_cov = time_current - time_update_last;
                if (dt_cov > 0.0) {
                    kf_output_.predict(dt_cov, q_output_, input_in_, false, true);
                    time_update_last = time_current;
                }
                kf_output_.predict(dt, q_output_, input_in_, true, false);

                time_predict_last_const = time_current;

                angvel_avr_ << imu_next_.angular_velocity.x, imu_next_.angular_velocity.y, imu_next_.angular_velocity.z;
                acc_avr_ << imu_next_.linear_acceleration.x, imu_next_.linear_acceleration.y, imu_next_.linear_acceleration.z;
                kf_output_.update_iterated_dyn_share_IMU();
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_last_ = imu_next_;
                imu_next_ = *(imu_deque_.front());
            } else {
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_last_ = imu_next_;
                imu_next_ = *(imu_deque_.front());
            }
        }
    }
}

void LaserMapping::processInputStateWithTimeSequence(V3D &euler_cur)
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    bool imu_prop_cov = false;
    double pcl_beg_time = Measures.lidar_beg_time;
    idx_ = -1;
    for (k_ = 0; k_ < time_seq_.size(); k_++) {
        PointType &point_body = feats_down_body_->points[idx_ + time_seq_[k_]];
        time_current = point_body.curvature / 1000.0 + pcl_beg_time;
        if (is_first_frame) {
            while (time_current > imu_next_.header.stamp.toSec()) {
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_last_ = imu_next_;
                imu_next_ = *(imu_deque_.front());
            }
            imu_prop_cov = true;

            is_first_frame = false;
            t_last = time_current;
            time_update_last = time_current;
            {
                input_in_.gyro << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y, imu_last_.angular_velocity.z;
                input_in_.acc << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y, imu_last_.linear_acceleration.z;
                input_in_.acc = input_in_.acc * g_m_s2_ / acc_norm;
            }
        }

        while (time_current > imu_next_.header.stamp.toSec()) {
            imu_deque_.pop_front();

            input_in_.gyro << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y, imu_last_.angular_velocity.z;
            input_in_.acc << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y, imu_last_.linear_acceleration.z;
            input_in_.acc = input_in_.acc * g_m_s2_ / acc_norm;
            double dt = imu_last_.header.stamp.toSec() - t_last;

            double dt_cov = imu_last_.header.stamp.toSec() - time_update_last;
            if (dt_cov > 0.0) {
                kf_input_.predict(dt_cov, q_input_, input_in_, false, true);
                time_update_last = imu_last_.header.stamp.toSec();
            }
            kf_input_.predict(dt, q_input_, input_in_, true, false);
            t_last = imu_last_.header.stamp.toSec();
            imu_prop_cov = true;

            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
        }
        if (reset_requested_) {
            break;
        }
        double dt = time_current - t_last;
        t_last = time_current;
        double propag_start = omp_get_wtime();

        if (!prop_at_freq_of_imu) {
            double dt_cov = time_current - time_update_last;
            if (dt_cov > 0.0) {
                kf_input_.predict(dt_cov, q_input_, input_in_, false, true);
                time_update_last = time_current;
            }
        }
        kf_input_.predict(dt, q_input_, input_in_, true, false);

        propag_time_ += omp_get_wtime() - propag_start;

        double t_update_start = omp_get_wtime();

        if (feats_down_size_ < 1) {
            ROS_WARN("No point, skip this scan!\n");

            idx_ += time_seq_[k_];
            continue;
        }
        if (!kf_input_.update_iterated_dyn_share_modified()) {
            idx_ = idx_ + time_seq_[k_];
            continue;
        }

        double solve_start = omp_get_wtime();

        if (publish_odometry_without_downsample) {
            publishOdometry(pub_odom_aft_mapped_);
            if (runtime_pos_log) {
                euler_cur = SO3ToEuler(kf_input_.x_.rot);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input_.x_.pos.transpose() << " " << kf_input_.x_.vel.transpose() << " " <<kf_input_.x_.bg.transpose()<<" "<<kf_input_.x_.ba.transpose()<<" "<<kf_input_.x_.gravity.transpose()<<" "<<feats_undistort_->points.size()<<endl;
            }
        }

        for (int j = 0; j < time_seq_[k_]; j++) {
            PointType &point_body_j = feats_down_body_->points[idx_ + j + 1];
            PointType &point_world_j = feats_down_world_->points[idx_ + j + 1];
            pointBodyToWorld(&point_body_j, &point_world_j);
        }
        solve_time_ += omp_get_wtime() - solve_start;

        update_time_ += omp_get_wtime() - t_update_start;
        idx_ = idx_ + time_seq_[k_];
    }
}

void LaserMapping::processInputStateWithoutTimeSequence()
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    bool imu_prop_cov = false;
    if (!imu_deque_.empty()) {
        imu_last_ = imu_next_;
        imu_next_ = *(imu_deque_.front());
        while (imu_next_.header.stamp.toSec() > time_current && ((imu_next_.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte))) {
            if (is_first_frame) {
                while (imu_next_.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte) {
                    imu_deque_.pop_front();
                    if (imu_deque_.empty()) break;
                    imu_last_ = imu_next_;
                    imu_next_ = *(imu_deque_.front());
                }
                break;

                imu_prop_cov = true;

                t_last = time_current;
                time_update_last = time_current;
                input_in_.gyro << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y, imu_last_.angular_velocity.z;
                input_in_.acc << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y, imu_last_.linear_acceleration.z;
                input_in_.acc = input_in_.acc * g_m_s2_ / acc_norm;

                is_first_frame = false;
            }
            time_current = imu_next_.header.stamp.toSec();

            if (!is_first_frame) {
                double dt = time_current - t_last;

                double dt_cov = time_current - time_update_last;
                if (dt_cov > 0.0) {
                    time_update_last = imu_next_.header.stamp.toSec();
                }

                t_last = imu_next_.header.stamp.toSec();

                input_in_.gyro << imu_next_.angular_velocity.x, imu_next_.angular_velocity.y, imu_next_.angular_velocity.z;
                input_in_.acc << imu_next_.linear_acceleration.x, imu_next_.linear_acceleration.y, imu_next_.linear_acceleration.z;
                input_in_.acc = input_in_.acc * g_m_s2_ / acc_norm;
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_last_ = imu_next_;
                imu_next_ = *(imu_deque_.front());
            } else {
                imu_deque_.pop_front();
                if (imu_deque_.empty()) break;
                imu_last_ = imu_next_;
                imu_next_ = *(imu_deque_.front());
            }
        }
    }
}

void LaserMapping::publishAndLog(
    double t0,
    double t1,
    double t3,
    double t5,
    int &frame_num,
    double &aver_time_consu,
    double &aver_time_icp,
    double &aver_time_match,
    double &aver_time_solve,
    double &aver_time_propag,
    V3D &euler_cur)
{
    if (!publish_odometry_without_downsample) {
        publishOdometry(pub_odom_aft_mapped_);
    }

    if (feats_down_size_ > 4) {
        mapIncremental();
    }

    if (path_en) publishPath(pub_path_);
    if (scan_pub_en || pcd_save_en) publishFrameWorld(pub_laser_cloud_full_res_);
    if (scan_pub_en && scan_body_pub_en) publishFrameBody(pub_laser_cloud_full_res_body_);

    if (!runtime_pos_log) {
        return;
    }

    frame_num ++;
    aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
    aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time_ / frame_num;
    aver_time_match = aver_time_match * (frame_num - 1) / frame_num + match_time_ / frame_num;
    aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time_ / frame_num;
    aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time_ / frame_num;
    t1_log_[time_log_counter_] = Measures.lidar_beg_time;
    s_plot_[time_log_counter_] = t5 - t0;
    s_plot2_[time_log_counter_] = feats_undistort_->points.size();
    s_plot3_[time_log_counter_] = aver_time_consu;
    time_log_counter_ ++;
    printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",
           t1 - t0,
           aver_time_match,
           aver_time_solve,
           t3 - t1,
           t5 - t3,
           aver_time_consu,
           aver_time_icp,
           aver_time_propag);
    if (!publish_odometry_without_downsample) {
        if (!use_imu_as_input) {
            euler_cur = SO3ToEuler(kf_output_.x_.rot);
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output_.x_.pos.transpose() << " " << kf_output_.x_.vel.transpose() \
            <<" "<<kf_output_.x_.omg.transpose()<<" "<<kf_output_.x_.acc.transpose()<<" "<<kf_output_.x_.gravity.transpose()<<" "<<kf_output_.x_.bg.transpose()<<" "<<kf_output_.x_.ba.transpose()<<" "<<feats_undistort_->points.size()<<endl;
        } else {
            euler_cur = SO3ToEuler(kf_input_.x_.rot);
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input_.x_.pos.transpose() << " " << kf_input_.x_.vel.transpose() \
            <<" "<<kf_input_.x_.bg.transpose()<<" "<<kf_input_.x_.ba.transpose()<<" "<<kf_input_.x_.gravity.transpose()<<" "<<feats_undistort_->points.size()<<endl;
        }
    }
    dumpLioStateToLog(fp_);
}

int LaserMapping::run()
{
    initialize();
//------------------------------------------------------------------------------------------------------
    int frame_num = 0;
    double aver_time_consu = 0;
    double aver_time_icp = 0;
    double aver_time_match = 0;
    double aver_time_incre = 0;
    double aver_time_solve = 0;
    double aver_time_propag = 0;
    V3D euler_cur;

    signal(SIGINT, LaserMapping::handleSignal);
    ros::Rate loop_rate(500);
    while (ros::ok()) {
        if (exit_requested_.load()) break;

        if (syncPackages(Measures)) {
            if (reset_requested_) {
                handleReset();
            }

            if (first_scan_) {
                handleFirstScan();
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            match_time_ = 0;
            solve_time_ = 0;
            propag_time_ = 0;
            update_time_ = 0;
            t0 = omp_get_wtime();

            /*** downsample the feature points in a scan ***/
            t1 = omp_get_wtime();
            prepareMeasurement();

            if (!ensureImuInitialized()) {
                continue;
            }

            if (initializeMapIfNeeded()) {
                continue;
            }

            /*** ICP and Kalman filter update ***/
            prepareStateEstimation();

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            if (!use_imu_as_input) {
                processOutputStateMode(euler_cur);
            } else {
                processInputStateMode(euler_cur);
            }
            // M3D rot_cur_lidar;
            // {
            //     rot_cur_lidar = state.rot_end;
            // }
            // euler_cur = RotMtoEuler(rot_cur_lidar);
            // geoQuat = tf::createQuaternionMsgFromRollPitchYaw
            //                     (euler_cur(0), euler_cur(1), euler_cur(2));
            t3 = omp_get_wtime();
            t5 = omp_get_wtime();
            publishAndLog(
                t0,
                t1,
                t3,
                t5,
                frame_num,
                aver_time_consu,
                aver_time_icp,
                aver_time_match,
                aver_time_solve,
                aver_time_propag,
                euler_cur);
        }
        loop_rate.sleep();
    }

    cleanup();
    return 0;
}

LaserMapping::LaserMapping(ros::NodeHandle &nh)
    : nh_(nh)
{
    active_instance_ = this;
}
