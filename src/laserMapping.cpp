#include "LaserMapping.h"

#include "Estimator.hpp"

const bool time_list(PointType& x, PointType& y) { 
    return (x.curvature < y.curvature);
}

LaserMapping::LaserMapping() {
    p_imu = std::make_shared<ImuProcess>();
    p_pre = std::make_shared<Preprocess>();
}

LaserMapping::~LaserMapping() {
}

void LaserMapping::InitROS(ros::NodeHandle &nh) {
    LoadParams(nh);
    SubAndPubToROS(nh);

    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    memset(point_selected_surf, true, sizeof(point_selected_surf));

    kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
    kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
    kf_output.init_dyn_share_modified_3h(
        get_f_output, df_dx_output,
        [this](state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R,
               esekfom::dyn_share_modified<double> &ekfom_data) {
            h_model_output(s, cov_p, cov_R, ekfom_data);
        },
        [this](state_output &s, esekfom::dyn_share_modified<double> &ekfom_data) {
            h_model_IMU_output(s, ekfom_data);
        });
    Eigen::Matrix<double, 30, 30> P_init_output; // = MD(24, 24)::Identity() * 0.01;
    P_init_output = MD(30, 30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6,6)::Identity() * 0.001;
    kf_output.change_P(P_init_output);
    Q_output = process_noise_cov_output();

    kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
    kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
    kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input,
        [this](state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data) {
            h_model_input(s, cov_p, cov_R, ekfom_data);
    });
    Eigen::Matrix<double, 24, 24> P_init; // = MD(18, 18)::Identity() * 0.1;
    P_init = MD(24, 24)::Identity() * 0.1;
    P_init.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6,6)::Identity() * 0.001;
    kf_input.change_P(P_init);
    Q_input = process_noise_cov_input();

}

bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
    int ivox_nearby_type = 6;
    int point_filter_num = 2;
    
    nh.param<bool>("use_imu_as_input", use_imu_as_input, 0);
    nh.param<bool>("check_satu", check_satu, true);
    nh.param<bool>("prop_at_freq_of_imu", prop_at_freq_of_imu, 1);

    nh.param<int>("preprocess/lidar_type", lidar_type, 1);
    nh.param<int>("point_filter_num", point_filter_num, 2);
    nh.param<std::string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<std::string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/cut_frame",cut_frame,false);
    nh.param<bool>("common/con_frame",con_frame,false);
    nh.param<int>("common/con_frame_num",con_frame_num,1);

    nh.param<bool>("space_down_sample", space_down_sample, 1);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<int>("init_map_size", init_map_size, 100);

    nh.param<bool>("mapping/imu_en", imu_en, true);
    nh.param<bool>("mapping/extrinsic_est_en",extrinsic_est_en,true);
    nh.param<double>("mapping/gyr_cov_input",gyr_cov_input,0.1);
    nh.param<double>("mapping/acc_cov_input",acc_cov_input,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("mapping/vel_cov",vel_cov,20);
    nh.param<double>("mapping/gyr_cov_output",gyr_cov_output,0.1);
    nh.param<double>("mapping/acc_cov_output",acc_cov_output,0.1);
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());
    nh.param<std::vector<double>>("mapping/gravity", gravity, std::vector<double>());
    nh.param<std::vector<double>>("mapping/gravity_init", gravity_init, std::vector<double>());
    nh.param<double>("mapping/imu_meas_omg_cov", imu_meas_omg_cov, 0.1);
    nh.param<double>("mapping/imu_meas_acc_cov", imu_meas_acc_cov, 0.1);
    nh.param<double>("mapping/acc_norm", acc_norm, 1.0);
    nh.param<double>("mapping/satu_acc", satu_acc, 3.0);
    nh.param<double>("mapping/satu_gyro", satu_gyro, 35.0);

    nh.param<double>("mapping/lidar_time_inte",lidar_time_inte,0.1);
    nh.param<double>("mapping/lidar_meas_cov",laser_point_cov,0.1);
    nh.param<double>("mapping/plane_thr", plane_thr, 0.1);
    nh.param<double>("mapping/match_s", match_s, 81.0);

    nh.param<bool>("odometry/publish_odometry_without_downsample", publish_odometry_without_downsample, false);
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en,true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en,true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<std::string>("pcd_save/dir", pcd_save_dir, "/tmp/pcd_save/");

    nh.param<float>("mapping/ivox_grid_resolution", ivox_options_.resolution_, 0.2);
    nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);
    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        // LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    down_size_filter_surf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    down_size_filter_map.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);

    p_pre->lidar_type = lidar_type;
    p_pre->point_filter_num = point_filter_num;
    p_imu->imu_en = imu_en;
    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);

    return true;
}

void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
    /*** ROS subscribe initialization ***/
    sub_pcl_ = p_pre->lidar_type == AVIA ?
        nh.subscribe(lid_topic, 200000, &LaserMapping::LivoxPclCallback, this) :
        nh.subscribe(lid_topic, 200000, &LaserMapping::StandardPclCallback, this);
    sub_imu_ = nh.subscribe(imu_topic, 200000, &LaserMapping::ImuCallback, this);

    path_.header.stamp = ros::Time().now();
    path_.header.frame_id = "camera_init";

    pub_laser_cloud_full_res_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 1000);
    pub_laser_cloud_full_res_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 1000);
    pub_laser_cloud_map_ = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1000);
    pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 1000);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 1000);

}

void LaserMapping::Run() {
    if (SyncPackages(meas)) {
        if (flg_first_scan) {
            first_lidar_time = meas.lidar_beg_time;
            flg_first_scan = false;
            if (first_imu_time < 1) {
                first_imu_time = imu_next.header.stamp.toSec();
                printf("first imu time: %f\n", first_imu_time);
            }
            time_current = 0.0;
            if (imu_en) {
                // imu_next = *(imu_deque.front());
                kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
                kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
                // kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
                // kf_output.x_.acc *= -1;
                
                while (meas.lidar_beg_time > imu_next.header.stamp.toSec())  // if it is needed for the new map?
                {
                    imu_deque.pop_front();
                    if (imu_deque.empty()) { break; }
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    // imu_deque.pop();
                }
            } else {
                kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);   // _init);
                kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);  //_init);
                kf_output.x_.acc << VEC_FROM_ARRAY(gravity);      //_init);
                kf_output.x_.acc *= -1;
                p_imu->imu_need_init_ = false;
                // p_imu->after_imu_init_ = true;
            }
            gravity_norm = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
        }

        p_imu->Process(meas, feats_undistort);
        if (space_down_sample) {
            down_size_filter_surf.setInputCloud(feats_undistort);
            down_size_filter_surf.filter(*feats_down_body);
            std::sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        } else {
            feats_down_body = meas.lidar;
            std::sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
        }
        
        time_seq = time_compressing<int>(feats_down_body);
        feats_down_size = feats_down_body->points.size();

        if (!p_imu->after_imu_init_) {
            if (!p_imu->imu_need_init_) {
                V3D tmp_gravity;
                if (imu_en) {
                    tmp_gravity = -p_imu->mean_acc / p_imu->mean_acc.norm() * gravity_norm;
                } else {
                    tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                    p_imu->after_imu_init_ = true;
                }
                // V3D tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                M3D rot_init;
                p_imu->Set_init(tmp_gravity, rot_init);
                kf_input.x_.rot = rot_init;
                kf_output.x_.rot = rot_init;
                // kf_input.x_.rot; //.normalize();
                // kf_output.x_.rot; //.normalize();
                kf_output.x_.acc = -rot_init.transpose() * kf_output.x_.gravity;
            } else {
                return;
            }
        }

        /*** initialize the map ***/
        if (!init_map) {
            feats_down_world->resize(feats_undistort->size());
            for (int i = 0; i < feats_undistort->size(); i++) {
                PointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i])); 
            }
            for (size_t i = 0; i < feats_down_world->size(); i++) {
                init_feats_world->points.emplace_back(feats_down_world->points[i]);
            }
            if (init_feats_world->size() < init_map_size) {
                init_map = false;
            } else {
                ivox_->AddPoints(init_feats_world->points);
                PublishInitMap(pub_laser_cloud_map_);

                init_feats_world.reset(new PointCloudXYZI());
                init_map = true;
            }
            return;
        }
        
        /*** ICP and Kalman filter update ***/
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        nearest_points.resize(feats_down_size);
        crossmat_list.resize(feats_down_size);
        pbody_list.resize(feats_down_size);

        for (size_t i = 0; i < feats_down_body->size(); i++) {
            V3D point_this(feats_down_body->points[i].x,
                           feats_down_body->points[i].y,
                           feats_down_body->points[i].z);
            pbody_list[i] = point_this;
            if (!extrinsic_est_en) {
                point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);
                crossmat_list[i] = point_crossmat;
            }
        }

        if (!use_imu_as_input) {
            bool imu_upda_cov = false;
            effct_feat_num = 0;
            if (time_seq.size() > 0) {
                double pcl_beg_time = meas.lidar_beg_time;
                h_idx = -1;
                for (time_k = 0; time_k < time_seq.size(); time_k++) {
                    PointType &point_body = feats_down_body->points[h_idx + time_seq[time_k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    if (is_first_frame) {
                        if (imu_en) {
                            while (time_current > imu_next.header.stamp.toSec()) {
                                imu_deque.pop_front();
                                if (imu_deque.empty()) break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                            angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }

                    if (imu_en && !imu_deque.empty()) {
                        bool last_imu = imu_next.header.stamp.toSec() == imu_deque.front()->header.stamp.toSec();
                        while (imu_next.header.stamp.toSec() < time_predict_last_const && !imu_deque.empty()) {
                            if (!last_imu) {
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                break;
                            } else {
                                imu_deque.pop_front();
                                if (imu_deque.empty()) break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                        }

                        bool imu_comes = time_current > imu_next.header.stamp.toSec();
                        while (imu_comes) {
                            imu_upda_cov = true;
                            angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            /*** covariance update ***/
                            double dt = imu_next.header.stamp.toSec() - time_predict_last_const;
                            kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = imu_next.header.stamp.toSec();

                            double dt_cov = imu_next.header.stamp.toSec() - time_update_last;
                            if (dt_cov > 0.0) {
                                time_update_last = imu_next.header.stamp.toSec();
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                kf_output.update_iterated_dyn_share_IMU();
                            }

                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_comes = time_current > imu_next.header.stamp.toSec();
                        }
                    }

                    double dt = time_current - time_predict_last_const;
                    if (!prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_output.predict(dt, Q_output, input_in, true, false);
                    time_predict_last_const = time_current;

                    if (feats_down_size < 1) {                        
                        ROS_WARN("No point, skip this scan!\n");
                        h_idx += time_seq[time_k];
                        continue;
                    }
                    if (!kf_output.update_iterated_dyn_share_modified()) {
                        h_idx += time_seq[time_k];
                        continue;
                    }
                    
                    if (publish_odometry_without_downsample) {
                        PublishOdometry(pub_odom_aft_mapped_);
                    }

                    for (int j = 0; j < time_seq[time_k]; ++j) {
                        PointType &point_body_j = feats_down_body->points[h_idx + j + 1];
                        PointType &point_world_j = feats_down_world->points[h_idx + j + 1];
                        PointBodyToWorld(&point_body_j, &point_world_j);
                    }
                    h_idx += time_seq[time_k];
                }
            } else {
                if (!imu_deque.empty()) {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());

                    while (imu_next.header.stamp.toSec() > time_current &&
                           imu_next.header.stamp.toSec() < meas.lidar_beg_time + lidar_time_inte) {
                        if (is_first_frame) {
                            {
                                while (imu_next.header.stamp.toSec() < meas.lidar_beg_time + lidar_time_inte) {
                                    imu_deque.pop_front();
                                    if (imu_deque.empty()) break;
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                }
                            }
                            break;

                            angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                            imu_upda_cov = true;
                            time_update_last = time_current;
                            time_predict_last_const = time_current;

                            is_first_frame = false;
                        }
                        time_current = imu_next.header.stamp.toSec();

                        if (!is_first_frame) {
                            double dt = time_current - time_predict_last_const;
                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0) {
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                time_update_last = time_current;
                            }
                            kf_output.predict(dt, Q_output, input_in, true, false);

                            time_predict_last_const = time_current;

                            angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;
                            kf_output.update_iterated_dyn_share_IMU();
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        } else {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                    }
                }
            }
        } else {
            bool imu_prop_cov = false;
            effct_feat_num = 0;
            if (time_seq.size() > 0) {
                double pcl_beg_time = meas.lidar_beg_time;
                h_idx = -1;
                for (time_k = 0; time_k < time_seq.size(); time_k++) {
                    PointType &point_body = feats_down_body->points[h_idx + time_seq[time_k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                    if (is_first_frame) {
                        while (time_current > imu_next.header.stamp.toSec()) {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                        imu_prop_cov = true;
                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current;
                        input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        input_in.acc = input_in.acc * gravity_norm / acc_norm;
                    }

                    while (time_current > imu_next.header.stamp.toSec()) {
                        imu_deque.pop_front();

                        input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        input_in.acc = input_in.acc * gravity_norm / acc_norm;
                        double dt = imu_last.header.stamp.toSec() - t_last;

                        double dt_cov = imu_last.header.stamp.toSec() - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = imu_last.header.stamp.toSec();
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false);
                        t_last = imu_last.header.stamp.toSec();
                        imu_prop_cov = true;

                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                    }

                    double dt = time_current - t_last;
                    t_last = time_current;
                    if (!prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);

                    if (feats_down_size < 1) {                        
                        ROS_WARN("No point, skip this scan!\n");
                        h_idx += time_seq[time_k];
                        continue;
                    }
                    if (!kf_input.update_iterated_dyn_share_modified()) {
                        h_idx += time_seq[time_k];
                        continue;
                    }

                    if(publish_odometry_without_downsample) {
                        PublishOdometry(pub_odom_aft_mapped_);
                    }

                    for (int j = 0; j < time_seq[time_k]; ++j) {
                        PointType &point_body_j = feats_down_body->points[h_idx + j + 1];
                        PointType &point_world_j = feats_down_world->points[h_idx + j + 1];
                        PointBodyToWorld(&point_body_j, &point_world_j);
                    }
                    h_idx += time_seq[time_k];
                }
            } else {
                if (!imu_deque.empty()) {
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    while (imu_next.header.stamp.toSec() > time_current &&
                           imu_next.header.stamp.toSec() < meas.lidar_beg_time + lidar_time_inte) {
                        if (is_first_frame) {
                            {
                                while (imu_next.header.stamp.toSec() < meas.lidar_beg_time + lidar_time_inte) {
                                    imu_deque.pop_front();
                                    if (imu_deque.empty()) break;
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                }
                            }
                            break;

                            imu_prop_cov = true;

                            t_last = time_current;
                            time_update_last = time_current;
                            input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * gravity_norm / acc_norm;

                            is_first_frame = false;
                        }
                        time_current = imu_next.header.stamp.toSec();

                        if (!is_first_frame) {
                            double dt = time_current - t_last;

                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0) {
                                time_update_last = imu_next.header.stamp.toSec();
                            }

                            t_last = imu_next.header.stamp.toSec();

                            input_in.gyro << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            input_in.acc << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;
                            input_in.acc = input_in.acc * gravity_norm / acc_norm;
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        } else {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                    }
                }
            }
        }

        if (!publish_odometry_without_downsample) { PublishOdometry(pub_odom_aft_mapped_); }
        if (feats_down_size > 4) { MapIncremental(); }
        if (path_en) { PublishPath(pub_path_); }
        if (scan_pub_en || pcd_save_en) { PublishFrameWorld(pub_laser_cloud_full_res_); }
        if (scan_pub_en && scan_body_pub_en) { PublishFrameBody(pub_laser_cloud_full_res_body_); }
    }
}

void LaserMapping::Finish() {
    if (pcl_wait_save->size() > 0 && pcd_save_en) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir = std::string(pcd_save_dir + "PCD/") + file_name;
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
}

bool LaserMapping::SyncPackages(MeasureGroup& meas) {
    if (!imu_en) {
        if (!lidar_buffer.empty()) {
            if (!lidar_pushed) {
                meas.lidar = lidar_buffer.front();
                meas.lidar_beg_time = time_buffer.front();
                lose_lid = false;
                if (meas.lidar->points.size() < 1) {
                    ROS_WARN("lose lidar");
                    lose_lid = true;
                } else {
                    double end_time = meas.lidar->points.back().curvature;
                    for (auto pt: meas.lidar->points) {
                        if (pt.curvature > end_time) { end_time = pt.curvature; }
                    }
                    lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
                    meas.lidar_last_time = lidar_end_time;
                }
                lidar_pushed = true;
            }

            time_buffer.pop_front();
            lidar_buffer.pop_front();
            lidar_pushed = false;
            if (!lose_lid) {
                return true;
            } else {
                return false;
            }
        }
        return false;
    }

    if (lidar_buffer.empty() || imu_deque.empty()) { return false; }

    if (!lidar_pushed) {
        lose_lid = false;
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() < 1) {
            ROS_WARN("lose lidar");
            lose_lid = true;
            // lidar_buffer.pop_front();
            // time_buffer.pop_front();
            // return false;
        } else {
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt: meas.lidar->points) {
                if (pt.curvature > end_time) { end_time = pt.curvature; }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            // cout << "check time lidar:" << end_time << endl;
            meas.lidar_last_time = lidar_end_time;
        }
        lidar_pushed = true;
    }

    if (!lose_lid && (last_timestamp_imu < lidar_end_time)) { return false; }
    if (lose_lid && last_timestamp_imu < meas.lidar_beg_time + lidar_time_inte) { return false; }

    if (!lose_lid && !imu_pushed) {
        /*** push imu data, and pop from imu buffer ***/
        if (p_imu->imu_need_init_) {
            double imu_time = imu_deque.front()->header.stamp.toSec();
            imu_next = *(imu_deque.front());
            meas.imu.shrink_to_fit();
            while (imu_time < lidar_end_time) {
                meas.imu.emplace_back(imu_deque.front());
                imu_last = imu_next;
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_time = imu_deque.front()->header.stamp.toSec();  // can be changed
                imu_next = *(imu_deque.front());
            }
        }
        imu_pushed = true;
    }

    if (lose_lid && !imu_pushed) {
        /*** push imu data, and pop from imu buffer ***/
        if (p_imu->imu_need_init_) {
            double imu_time = imu_deque.front()->header.stamp.toSec();
            meas.imu.shrink_to_fit();
            imu_next = *(imu_deque.front());
            while (imu_time < meas.lidar_beg_time + lidar_time_inte) {
                meas.imu.emplace_back(imu_deque.front());
                imu_last = imu_next;
                imu_deque.pop_front();
                if (imu_deque.empty()) break;
                imu_time = imu_deque.front()->header.stamp.toSec();  // can be changed
                imu_next = *(imu_deque.front());
            }
        }
        imu_pushed = true;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    imu_pushed = false;
    return true;
}

void LaserMapping::MapIncremental()
{
    // 将当前帧下采样后的世界系点增量写入 ivox 地图。
    // 若当前体素内已经存在足够接近的点，则跳过，避免地图过密。
    PointVector points_to_add;
    int cur_pts = feats_down_world->size();
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i) {
        /* decide if need add to map */
        PointType &point_world = feats_down_world->points[i];
        if (!nearest_points[i].empty()) {
            const PointVector &points_near = nearest_points[i];

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


void LaserMapping::StandardPclCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        return;
    }

    last_timestamp_lidar = msg->header.stamp.toSec();

    if ((lidar_type == VELO16 || lidar_type == OUST64 || lidar_type == HESAIxt32) && cut_frame_init) {
        std::deque<PointCloudXYZI::Ptr> ptr;
        std::deque<double> timestamp_lidar;
        p_pre->process_cut_frame_pcl2(msg, ptr, timestamp_lidar, cut_frame_num, scan_count);
        while (!ptr.empty() && !timestamp_lidar.empty()) {
            lidar_buffer.push_back(ptr.front());
            ptr.pop_front();
            time_buffer.push_back(timestamp_lidar.front() / double(1000));
            timestamp_lidar.pop_front();
        }
    } else {
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI(20000, 1));
        p_pre->process(msg, ptr);
        if (con_frame) {
            if (frame_ct == 0) {
                time_con = last_timestamp_lidar;
            }
            if (frame_ct < 10) {
                for (int i = 0; i < ptr->size(); i++) {
                    ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                    ptr_con->push_back(ptr->points[i]);
                }
                frame_ct ++;
            } else {
                PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI(10000, 1));
                *ptr_con_i = *ptr_con;
                lidar_buffer.push_back(ptr_con_i);
                time_buffer.push_back(time_con);
                ptr_con->clear();
                frame_ct = 0;
            }
        } else {
            if (ptr->points.size() > 0) {
                lidar_buffer.emplace_back(ptr);
                time_buffer.emplace_back(msg->header.stamp.toSec());
            }
        }
    }
}

void LaserMapping::LivoxPclCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        return;
    }
    
    last_timestamp_lidar = msg->header.stamp.toSec();    

    if (cut_frame_init) {
        std::deque<PointCloudXYZI::Ptr> ptr;
        std::deque<double> timestamp_lidar;
        p_pre->process_cut_frame_livox(msg, ptr, timestamp_lidar, cut_frame_num, scan_count);

        while (!ptr.empty() && !timestamp_lidar.empty()) {
            lidar_buffer.push_back(ptr.front());
            ptr.pop_front();
            time_buffer.push_back(timestamp_lidar.front() / double(1000));
            timestamp_lidar.pop_front();
        }
    } else {
        PointCloudXYZI::Ptr ptr(new PointCloudXYZI(10000,1));
        p_pre->process(msg, ptr); 
        if (con_frame) {
            if (frame_ct == 0) {
                time_con = last_timestamp_lidar; //msg->header.stamp.toSec();
            }
            if (frame_ct < 10) {
                for (int i = 0; i < ptr->size(); i++) {
                    ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                    ptr_con->push_back(ptr->points[i]);
                }
                frame_ct ++;
            } else {
                PointCloudXYZI::Ptr  ptr_con_i(new PointCloudXYZI(10000,1));
                // cout << "ptr div num:" << ptr_div->size() << endl;
                *ptr_con_i = *ptr_con;
                double time_con_i = time_con;
                lidar_buffer.push_back(ptr_con_i);
                time_buffer.push_back(time_con_i);
                ptr_con->clear();
                frame_ct = 0;
            }

        } else {
            if (ptr->points.size() > 0) {
                lidar_buffer.emplace_back(ptr);
                time_buffer.emplace_back(msg->header.stamp.toSec());
            }
        }
    }
}

void LaserMapping::ImuCallback(const sensor_msgs::Imu::ConstPtr &msg_in) {
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(
        msg->header.stamp.toSec() - timediff_imu_wrt_lidar - time_lag_IMU_wtr_lidar);
    double timestamp = msg->header.stamp.toSec();
    if (timestamp < last_timestamp_imu) {
        ROS_ERROR("imu loop back, clear deque");
        return;
    }

    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
}

void LaserMapping::PublishInitMap(const ros::Publisher &pub_laser_cloud_full_res) {
    // 在初始地图尚未建完之前，发布累计的初始地图点云，方便观察初始化进度。
    int size_init_map = init_feats_world->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*init_feats_world, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pub_laser_cloud_full_res.publish(laserCloudmsg);
}

void LaserMapping::PublishFrameWorld(const ros::Publisher &pub_laser_cloud_full_res) {
    if(scan_pub_en) {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_full_res.publish(laserCloudmsg);
    }
    
    if(pcd_save_en) {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index ++;
            string all_points_dir(string(pcd_save_dir + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            // cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void LaserMapping::PublishFrameBody(const ros::Publisher &pubLaserCloudFull_body)
{
    // 发布 body/IMU 系下的点云，主要用于调试外参和去畸变效果。
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
        PointBodyLidarToImu(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

void LaserMapping::setPoseStamp(geometry_msgs::Pose &pose)
{
    if (!use_imu_as_input) {
        pose.position.x = kf_output.x_.pos(0);
        pose.position.y = kf_output.x_.pos(1);
        pose.position.z = kf_output.x_.pos(2);
        Eigen::Quaterniond q(kf_output.x_.rot);
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
    } else {
        pose.position.x = kf_input.x_.pos(0);
        pose.position.y = kf_input.x_.pos(1);
        pose.position.z = kf_input.x_.pos(2);
        Eigen::Quaterniond q(kf_input.x_.rot);
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();
    }
}

void LaserMapping::PublishOdometry(const ros::Publisher &pubOdomAftMapped)
{
    // 发布 odom 消息并同步 TF，使 RViz 和其他节点可以直接消费当前位姿。
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

void LaserMapping::PublishPath(const ros::Publisher &pubPath)
{
    // 累计历史位姿轨迹，便于在 RViz 中直观看建图轨迹是否稳定。
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

void LaserMapping::PointBodyToWorld(PointType const * const pi, PointType * const po)
{
    // 统一封装“雷达点 -> 世界系点”的坐标变换。
    // 这里会根据当前工作模式和外参估计开关选择不同的状态量。
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global;
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            p_global = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
        } else {
            p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
        }
    } else {
        if (!use_imu_as_input) {
            p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos;
        } else {
            p_global = kf_input.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos;
        }
    }

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyLidarToImu(PointType const * const pi, PointType * const po)
{
    // 发布 body 系点云时，需要把 LiDAR 点先转到 IMU/body 坐标系。
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
        } else {
            p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    } else {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

Eigen::Matrix<double, 24, 24> LaserMapping::process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	return cov;
}

Eigen::Matrix<double, 30, 30> LaserMapping::process_noise_cov_output()
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

void LaserMapping::h_model_input(
    state_input& s,
    Eigen::Matrix3d cov_p,
    Eigen::Matrix3d cov_R,
    esekfom::dyn_share_modified<double>& ekfom_data
) {
    VF(4) pabcd;
    pabcd.setZero();
    normvec->resize(time_seq[time_k]);
    int effect_num_k = 0;
    for (int j = 0; j < time_seq[time_k]; j++) {
        PointType &point_body_j = feats_down_body->points[h_idx + j + 1];
        PointType &point_world_j = feats_down_world->points[h_idx + j + 1];
        PointBodyToWorld(&point_body_j, &point_world_j);
        V3D p_body = pbody_list[h_idx + j + 1];
        double p_norm = p_body.norm();
        {
            auto &points_near = nearest_points[h_idx + j + 1];
            ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);
            if (points_near.size() < NUM_MATCH_POINTS) {
                point_selected_surf[h_idx + j + 1] = false;
            } else {
                point_selected_surf[h_idx + j + 1] = false;
                if (esti_plane(pabcd, points_near, static_cast<float>(plane_thr))) {
                    float pd2 = fabs(
                        pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y
                        + pabcd(2) * point_world_j.z + pabcd(3));
                    if (p_norm > match_s * pd2 * pd2) {
                        point_selected_surf[h_idx + j + 1] = true;
                        normvec->points[j].x = pabcd(0);
                        normvec->points[j].y = pabcd(1);
                        normvec->points[j].z = pabcd(2);
                        normvec->points[j].intensity = pabcd(3);
                        effect_num_k++;
                    }
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

    for (int j = 0; j < time_seq[time_k]; j++) {
        if (point_selected_surf[h_idx + j + 1]) {
            V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

            if (extrinsic_est_en) {
                V3D p_body = pbody_list[h_idx + j + 1];
                M3D p_crossmat, p_imu_crossmat;
                p_crossmat << SKEW_SYM_MATRX(p_body);
                V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
                p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(p_imu_crossmat * C);
                V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
                ekfom_data.h_x.block<1, 12>(m, 0)
                    << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B),
                    VEC_FROM_ARRAY(C);
            } else {
                M3D point_crossmat = crossmat_list[h_idx + j + 1];
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(point_crossmat * C);
                ekfom_data.h_x.block<1, 12>(m, 0)
                    << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }
            ekfom_data.z(m) =
                -norm_vec(0) * feats_down_world->points[h_idx + j + 1].x
                - norm_vec(1) * feats_down_world->points[h_idx + j + 1].y
                - norm_vec(2) * feats_down_world->points[h_idx + j + 1].z
                - normvec->points[j].intensity;

            m++;
        }
    }
    effct_feat_num += effect_num_k;
}

void LaserMapping::h_model_output(
    state_output& s,
    Eigen::Matrix3d cov_p,
    Eigen::Matrix3d cov_R,
    esekfom::dyn_share_modified<double>& ekfom_data
) {
    VF(4) pabcd;
    pabcd.setZero();
    normvec->resize(time_seq[time_k]);
    int effect_num_k = 0;
    for (int j = 0; j < time_seq[time_k]; j++) {
        PointType &point_body_j = feats_down_body->points[h_idx + j + 1];
        PointType &point_world_j = feats_down_world->points[h_idx + j + 1];
        PointBodyToWorld(&point_body_j, &point_world_j);
        V3D p_body = pbody_list[h_idx + j + 1];
        double p_norm = p_body.norm();
        {
            auto &points_near = nearest_points[h_idx + j + 1];
            ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);

            if (points_near.size() < NUM_MATCH_POINTS) {
                point_selected_surf[h_idx + j + 1] = false;
            } else {
                point_selected_surf[h_idx + j + 1] = false;
                if (esti_plane(pabcd, points_near, static_cast<float>(plane_thr))) {
                    float pd2 = fabs(
                        pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y
                        + pabcd(2) * point_world_j.z + pabcd(3));
                    if (p_norm > match_s * pd2 * pd2) {
                        point_selected_surf[h_idx + j + 1] = true;
                        normvec->points[j].x = pabcd(0);
                        normvec->points[j].y = pabcd(1);
                        normvec->points[j].z = pabcd(2);
                        normvec->points[j].intensity = pabcd(3);
                        effect_num_k++;
                    }
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
    for (int j = 0; j < time_seq[time_k]; j++) {
        if (point_selected_surf[h_idx + j + 1]) {
            V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
            if (extrinsic_est_en) {
                V3D p_body = pbody_list[h_idx + j + 1];
                M3D p_crossmat, p_imu_crossmat;
                p_crossmat << SKEW_SYM_MATRX(p_body);
                V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
                p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(p_imu_crossmat * C);
                V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
                ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A),
                    VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            } else {
                M3D point_crossmat = crossmat_list[h_idx + j + 1];
                V3D C(s.rot.transpose() * norm_vec);
                V3D A(point_crossmat * C);
                ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            }

            ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[h_idx + j + 1].x
                            - norm_vec(1) * feats_down_world->points[h_idx + j + 1].y
                            - norm_vec(2) * feats_down_world->points[h_idx + j + 1].z
                            - normvec->points[j].intensity;
            m++;
        }
    }
    effct_feat_num += effect_num_k;
}

void LaserMapping::h_model_IMU_output(state_output& s, esekfom::dyn_share_modified<double>& ekfom_data) {
    std::memset(ekfom_data.satu_check, false, 6);
    ekfom_data.z_IMU.block<3, 1>(0, 0) = angvel_avr - s.omg - s.bg;
    ekfom_data.z_IMU.block<3, 1>(3, 0) = acc_avr * gravity_norm / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov,
        imu_meas_acc_cov, imu_meas_acc_cov;

    if (check_satu) {
        if (fabs(angvel_avr(0)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[0] = true;
            ekfom_data.z_IMU(0) = 0.0;
        }
        if (fabs(angvel_avr(1)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[1] = true;
            ekfom_data.z_IMU(1) = 0.0;
        }
        if (fabs(angvel_avr(2)) >= 0.99 * satu_gyro) {
            ekfom_data.satu_check[2] = true;
            ekfom_data.z_IMU(2) = 0.0;
        }
        if (fabs(acc_avr(0)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[3] = true;
            ekfom_data.z_IMU(3) = 0.0;
        }
        if (fabs(acc_avr(1)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[4] = true;
            ekfom_data.z_IMU(4) = 0.0;
        }
        if (fabs(acc_avr(2)) >= 0.99 * satu_acc) {
            ekfom_data.satu_check[5] = true;
            ekfom_data.z_IMU(5) = 0.0;
        }
    }
}
