#ifndef LASER_MAPPING_H
#define LASER_MAPPING_H


#include <malloc.h>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <livox_ros_driver/CustomMsg.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "ivox/ivox3d.h"
#include "common_lib.h"
#include "IMU_Processing.h"


// #define IVOX_NODE_TYPE_PHC
#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::DEFAULT, PointType>;
#endif


// LaserMapping 负责整个在线建图流程的编排：
// 1. 读取参数、初始化 EKF / 地图 / ROS 通信对象。
// 2. 接收 IMU 与 LiDAR 回调，并在内部做时序对齐。
// 3. 驱动状态传播、点面匹配、地图增量更新与结果发布。
class LaserMapping {
public:
    LaserMapping();
    ~LaserMapping();

    void InitROS(ros::NodeHandle &nh);
    void Run();
    void Finish();

    void StandardPclCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void LivoxPclCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void ImuCallback(const sensor_msgs::Imu::ConstPtr &msg_in);


private:
    bool LoadParams(ros::NodeHandle &nh);
    void SubAndPubToROS(ros::NodeHandle &nh);
    
    bool SyncPackages(MeasureGroup &meas);
    void MapIncremental();
    
    void PublishInitMap(const ros::Publisher &pub_laser_cloud_full_res);
    void PublishFrameWorld(const ros::Publisher &pub_laser_cloud_full_res);
    void PublishFrameBody(const ros::Publisher &pub_laser_cloud_full_body);
    void PublishOdometry(const ros::Publisher &pub_odom_aft_mapped);
    void PublishPath(const ros::Publisher &pub_path);
    
    void PointBodyToWorld(PointType const * const pi, PointType * const po);
    void PointBodyLidarToImu(PointType const * const pi, PointType * const po);

    Eigen::Matrix<double, 24, 24> process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> process_noise_cov_output();

private:
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_;

    std::shared_ptr<ImuProcess> p_imu;
    std::shared_ptr<Preprocess> p_pre;

    // ROS 通信对象：订阅原始传感器数据，发布建图结果。
    ros::Subscriber sub_pcl_;
    ros::Subscriber sub_imu_;
    ros::Publisher pub_laser_cloud_full_res_;
    ros::Publisher pub_laser_cloud_full_res_body_;
    ros::Publisher pub_laser_cloud_map_;
    ros::Publisher pub_odom_aft_mapped_;
    ros::Publisher pub_path_;

    nav_msgs::Path path_;
    nav_msgs::Odometry odom_aft_mapped_;
    geometry_msgs::PoseStamped msg_body_pose_;

    // 话题回调接收的雷达数据和IMU数据
    std::deque<PointCloudXYZI::Ptr>  lidar_buffer;
    std::deque<double>               time_buffer;
    std::deque<sensor_msgs::Imu::Ptr> imu_deque;

    double last_timestamp_lidar = -1.0;
    double lidar_end_time = 0.0;
    int scan_count = 0;
    bool cut_frame_init = false; // true;
    int frame_ct = 0;
    double time_con = 0.0;
    PointCloudXYZI::Ptr ptr_con(new PointCloudXYZI());

    double last_timestamp_imu = -1.0;
    double timediff_imu_wrt_lidar = 0.0;
    double time_lag_IMU_wtr_lidar = 0.0;

    bool lidar_pushed = false;
    bool imu_pushed = false;
    
    // 降采样滤波器对象。
    pcl::VoxelGrid<PointType> down_size_filter_surf;
    pcl::VoxelGrid<PointType> down_size_filter_map;

    // 点云与发布缓存。
    PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
    PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
    PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
    PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());
    bool init_map = false;

    // EKF 初始化矩阵、过程噪声、运行期状态与配套缓存。
    esekfom::esekf<state_input, 24, input_ikfom> kf_input;
    esekfom::esekf<state_output, 30, input_ikfom> kf_output;
    Eigen::Matrix<double, 30, 30> Q_output;
    Eigen::Matrix<double, 24, 24> Q_input;


private:
    // 通过 ROS 参数服务器加载配置项
    int lidar_type;
    std::string lid_topic;
    std::string imu_topic;
    bool cut_frame = false;
    bool con_frame = false;
    int cut_frame_num = 1;

    int  init_map_size = 10;
    bool use_imu_as_input = false;
    bool extrinsic_est_en = true;
    bool imu_en = true;
    bool space_down_sample = true;
    double filter_size_surf_min = 0.5;
    double filter_size_map_min = 0.5;
    std::vector<double> extrinT{3, 0.0};
    std::vector<double> extrinR{9, 0.0};
    V3D Lidar_T_wrt_IMU;
    M3D Lidar_R_wrt_IMU;
    std::vector<double> gravity_init;
    std::vector<double> gravity;

	double gyr_cov_input;
	double acc_cov_input;
	double b_gyr_cov;
	double b_acc_cov;
	double vel_cov;
	double gyr_cov_output;
	double acc_cov_output;

    bool publish_odometry_without_downsample;
    bool path_en;
    bool scan_pub_en;
    bool scan_body_pub_en;

};

#endif // LASER_MAPPING_H
