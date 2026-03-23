#include "LaserMapping.h"

#include "ros/ros.h"
#include <csignal>
#include <unistd.h>

bool LIO_FLAG_EXIT = false;
void SigHandle(int sig) {
    LIO_FLAG_EXIT = true;
    ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh("~");
    ros::AsyncSpinner spinner(0);
    spinner.start();

    auto laser_mapping = std::make_shared<LaserMapping>();
    laser_mapping->InitROS(nh);
    
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    while (ros::ok()) {
        if (LIO_FLAG_EXIT) {
            break;
        }
        ros::spinOnce();
        laser_mapping->Run();
        rate.sleep();
    }

    ROS_INFO("finishing mapping");
    laser_mapping->Finish();

    return 0;
}
