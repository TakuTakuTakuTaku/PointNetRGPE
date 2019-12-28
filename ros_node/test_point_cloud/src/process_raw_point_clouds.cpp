#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/parse.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>

ros::Publisher pub;

void process_points_clouds(const sensor_msgs::PointCloud2ConstPtr& input_clouds)
{
	// Read in the cloud data
  	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  	
	pcl::fromROSMsg (*input_clouds, *cloud); 

  	// Create the filtering object: downsample the dataset using a leaf size of 0.5mm
  	pcl::VoxelGrid<pcl::PointXYZ> vg;
  	vg.setInputCloud (cloud);
  	vg.setLeafSize (2.0f, 2.0f, 2.0f); //this value can be change
  	vg.filter (*cloud_filtered);
  	// std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; 

  	// Create the segmentation object for the planar model and set all the parameters
  	pcl::SACSegmentation<pcl::PointXYZ> seg;
  	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

  	seg.setOptimizeCoefficients (true);
  	seg.setModelType (pcl::SACMODEL_PLANE);
  	seg.setMethodType (pcl::SAC_RANSAC);
  	seg.setMaxIterations (5);
  	seg.setDistanceThreshold (3.0);
  	seg.setInputCloud (cloud_filtered);
  	seg.segment (*inliers, *coefficients);

  	pcl::ExtractIndices<pcl::PointXYZ> extract;
  	extract.setInputCloud (cloud_filtered);
  	extract.setIndices (inliers);
  
  	// Remove the planar inliers, extract the rest
  	extract.setNegative (true);
  	extract.filter (*cloud_filtered);
  
  	// Remove outliers using a StatisticalOutlierRemoval filter
  	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  	sor.setInputCloud (cloud_filtered);
  	sor.setMeanK (50);
  	sor.setStddevMulThresh (1.0);
  	sor.filter (*cloud_filtered);

	sensor_msgs::PointCloud2 output_clouds;
	pcl::toROSMsg(*cloud_filtered, output_clouds);

  	// Publish the data.
  	pub.publish(output_clouds);
	
}

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "process_raw_points_clouds");

	ros::NodeHandle n;
	ros::Subscriber sub = n.subscribe("raw_point_clouds", 2, process_points_clouds);
  	pub = n.advertise<sensor_msgs::PointCloud2> ("point_clouds", 1);

	ros::spin();

	return 0;
}