#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>

int main(int argc, char *argv[])
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr input_clouds (new pcl::PointCloud<pcl::PointXYZ>);
  	pcl::io::loadPLYFile ("/home/zju-taku/catkin_ws/src/test_point_cloud/src/04.ply", *input_clouds);
	
	ros::init(argc, argv, "publish_raw_point_clouds");
	ros::NodeHandle nh;
	ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2> ("raw_point_clouds", 1);

	sensor_msgs::PointCloud2 output_clouds;
	pcl::toROSMsg(*input_clouds, output_clouds);

	ros::Rate loop_rate(0.05);

	while (nh.ok())
	{
		pub.publish (output_clouds);
		ros::spinOnce ();
		loop_rate.sleep ();
	}
	return 0;
}

