#include "LaserMapping.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh("~");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    LaserMapping laser_mapping(nh);
    return laser_mapping.run();
}
