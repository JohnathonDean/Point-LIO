#pragma once

#include "parameters.h"
#include <atomic>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>

class LaserMapping {
public:
    static constexpr int kMaxLogSize = 720000;

    explicit LaserMapping(ros::NodeHandle &nh);
    static LaserMapping &activeInstance();
    static void handleSignal(int sig);

    int run();

private:
    static void hModelInput(
        state_input &s,
        Eigen::Matrix3d cov_p,
        Eigen::Matrix3d cov_R,
        esekfom::dyn_share_modified<double> &ekfom_data);
    static void hModelOutput(
        state_output &s,
        Eigen::Matrix3d cov_p,
        Eigen::Matrix3d cov_R,
        esekfom::dyn_share_modified<double> &ekfom_data);
    static void hModelImuOutput(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

    void initialize();
    void cleanup();
    void handleReset();
    void handleFirstScan();
    void prepareMeasurement();
    bool ensureImuInitialized();
    bool initializeMapIfNeeded();
    void prepareStateEstimation();
    void standardPclCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void livoxPclCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void imuCallback(const sensor_msgs::Imu::ConstPtr &msg_in);
    bool syncPackages(MeasureGroup &meas);
    void processOutputStateMode(V3D &euler_cur);
    void processOutputStateWithTimeSequence(V3D &euler_cur);
    void processOutputStateWithoutTimeSequence();
    void processInputStateMode(V3D &euler_cur);
    void processInputStateWithTimeSequence(V3D &euler_cur);
    void processInputStateWithoutTimeSequence();
    void publishAndLog(
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
        V3D &euler_cur);
    void pointBodyToWorld(PointType const * const pi, PointType * const po);
    void dumpLioStateToLog(FILE *fp);
    void pointBodyLidarToImu(PointType const * const pi, PointType * const po);
    void mapIncremental();
    void publishInitMap(const ros::Publisher &pub_laser_cloud_full_res);
    void publishFrameWorld(const ros::Publisher &pub_laser_cloud_full_res);
    void publishFrameBody(const ros::Publisher &pub_laser_cloud_full_body);
    template<typename T>
    void setPoseStamp(T &out);
    void publishOdometry(const ros::Publisher &pub_odom_aft_mapped);
    void publishPath(const ros::Publisher &pub_path);

    ros::NodeHandle &nh_;
    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Publisher pub_laser_cloud_full_res_;
    ros::Publisher pub_laser_cloud_full_res_body_;
    ros::Publisher pub_laser_cloud_map_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;

    pcl::VoxelGrid<PointType> down_size_filter_surf_;
    pcl::VoxelGrid<PointType> down_size_filter_map_;

    Eigen::Matrix<double, 24, 24> p_init_;
    Eigen::Matrix<double, 24, 24> q_input_;
    Eigen::Matrix<double, 30, 30> p_init_output_;
    Eigen::Matrix<double, 30, 30> q_output_;
    std::string root_dir_{ROOT_DIR};
    std::atomic<bool> exit_requested_{false};
    int time_log_counter_ = 0;
    bool init_map_ = false;
    bool first_scan_ = true;
    bool reset_requested_ = false;
    double match_time_ = 0.0;
    double solve_time_ = 0.0;
    double propag_time_ = 0.0;
    double update_time_ = 0.0;
    PointCloudXYZI::Ptr feats_undistort_{new PointCloudXYZI()};
    PointCloudXYZI::Ptr init_feats_world_{new PointCloudXYZI()};
    PointCloudXYZI::Ptr pcl_wait_save_{new PointCloudXYZI()};
    nav_msgs::Path path_;
    nav_msgs::Odometry odom_aft_mapped_;
    geometry_msgs::PoseStamped msg_body_pose_;
    PointCloudXYZI::Ptr normvec_{new PointCloudXYZI(100000, 1)};
    std::vector<int> time_seq_;
    PointCloudXYZI::Ptr feats_down_body_{new PointCloudXYZI(10000, 1)};
    PointCloudXYZI::Ptr feats_down_world_{new PointCloudXYZI(10000, 1)};
    std::vector<V3D> pbody_list_;
    std::vector<PointVector> nearest_points_;
    std::shared_ptr<IVoxType> ivox_;
    std::vector<float> point_search_sq_dis_{NUM_MATCH_POINTS};
    bool point_selected_surf_[100000] = {0};
    std::vector<M3D> crossmat_list_;
    int effct_feat_num_ = 0;
    int k_ = 0;
    int idx_ = -1;
    esekfom::esekf<state_input, 24, input_ikfom> kf_input_;
    esekfom::esekf<state_output, 30, input_ikfom> kf_output_;
    input_ikfom input_in_;
    V3D angvel_avr_;
    V3D acc_avr_;
    V3D acc_avr_norm_;
    int feats_down_size_ = 0;
    V3D lidar_t_wrt_imu_{Zero3d};
    M3D lidar_r_wrt_imu_{Eye3d};
    double g_m_s2_ = 9.81;

    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    int scan_count_ = 0;
    int frame_ct_ = 0;
    int wait_num_ = 0;
    bool lose_lid_ = false;
    bool lidar_pushed_ = false;
    bool imu_pushed_ = false;
    double time_lag_imu_wrt_lidar_ = 0.0;
    double timediff_imu_wrt_lidar_ = 0.0;
    sensor_msgs::Imu imu_last_;
    sensor_msgs::Imu imu_next_;
    PointCloudXYZI::Ptr ptr_con_{new PointCloudXYZI()};
    std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
    std::deque<double> time_buffer_;
    std::deque<sensor_msgs::Imu::Ptr> imu_deque_;
    std::vector<double> t1_log_{kMaxLogSize, 0.0};
    std::vector<double> s_plot_{kMaxLogSize, 0.0};
    std::vector<double> s_plot2_{kMaxLogSize, 0.0};
    std::vector<double> s_plot3_{kMaxLogSize, 0.0};
    std::vector<double> s_plot11_{kMaxLogSize, 0.0};

    static LaserMapping *active_instance_;

    FILE *fp_ = nullptr;
};

template<typename T>
void LaserMapping::setPoseStamp(T &out)
{
    if (!use_imu_as_input) {
        out.position.x = kf_output_.x_.pos(0);
        out.position.y = kf_output_.x_.pos(1);
        out.position.z = kf_output_.x_.pos(2);
        Eigen::Quaterniond q(kf_output_.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    } else {
        out.position.x = kf_input_.x_.pos(0);
        out.position.y = kf_input_.x_.pos(1);
        out.position.z = kf_input_.x_.pos(2);
        Eigen::Quaterniond q(kf_input_.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
}
