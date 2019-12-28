#!/usr/bin/env python
'''obtain_6d_pose ROS Node'''
import rospy
import numpy as np
import tensorflow as tf
import random
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import tf_util
from test_point_cloud.msg import PointCloudCls
from test_point_cloud.msg import Trajectory

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True

NUM_POINT = 500  # Number of point (can be change)
CLASS_SIZE = 2  # Class size (can be change)
BATCH_SIZE = 1  # Batch size

def get_data(data):

	assert isinstance(data.cloud, PointCloud2)
	cloud = point_cloud2.read_points(data.cloud, field_names=("x", "y", "z"), skip_nans=True)
	point_cloud = np.asarray(list(cloud))
	real_point_cloud = np.asarray(random.sample(list(point_cloud), NUM_POINT), dtype='float32')
	real_point_cloud = real_point_cloud[np.newaxis, :]
	real_class_number = [data.data]

	return real_point_cloud, real_class_number

def point_net_model(point_cloud, is_training, bn_decay=None):
	""" Classification PointNet, input is BxNx3, output Bx40 """
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	feature_dim = point_cloud.get_shape()[2].value
	input_image = tf.expand_dims(point_cloud, -1)

	# Point functions (MLP implemented as conv2d)
	net = tf_util.conv2d(input_image, 64, [1, feature_dim],
						padding='VALID', stride=[1, 1],
						is_training=is_training,
						scope='conv1', bn_decay=bn_decay)

	net = tf_util.conv2d(net, 64, [1, 1],
						padding='VALID', stride=[1, 1],
						is_training=is_training,
						scope='conv2', bn_decay=bn_decay)

	net = tf_util.conv2d(net, 64, [1, 1],
						padding='VALID', stride=[1, 1],
						is_training=is_training,
						scope='conv3', bn_decay=bn_decay)

	net = tf_util.conv2d(net, 128, [1, 1],
						padding='VALID', stride=[1, 1],
						is_training=is_training,
						scope='conv4', bn_decay=bn_decay)

	net = tf_util.conv2d(net, 1024, [1, 1],
						padding='VALID', stride=[1, 1],
						is_training=is_training,
						scope='conv5', bn_decay=bn_decay) # check the maximum along point dimension

	net = tf_util.max_pool2d(net, [num_point, 1],
							padding='VALID', scope='maxpool')

	# MLP on global point cloud vector
	net = tf.reshape(net, [batch_size, -1])
	net, _, _ = tf_util.fully_connected(net, 512, is_training=is_training,
										scope='fc1', bn_decay=bn_decay)
	net, _, _ = tf_util.fully_connected(net, 256, is_training=is_training,
										scope='fc2', bn_decay=bn_decay)
	return net

def Generator_Trans(point_cloud, class_number, is_training=tf.cast(False, tf.bool), bn_decay=None):

	element_mean = tf.reduce_mean(point_cloud, axis=1)
	point_cloud_normalized = point_cloud - tf.expand_dims(element_mean, 1)
	point_cloud_normalized = point_cloud_normalized/1000.0

	cls_gt_onehot = tf.one_hot(indices=class_number, depth=CLASS_SIZE)
	cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
	cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

	with tf.variable_scope('Generator_Trans'):
		trans_pred_res = point_net_model(tf.concat([point_cloud_normalized, cls_gt_onehot_tile], 2), 
										is_training, bn_decay=bn_decay)

		trans_pred_res, _, _ = tf_util.fully_connected(trans_pred_res, 3, activation_fn=None, scope='output')
		
		trans_pred = trans_pred_res*1000 + element_mean

	return trans_pred

def Generator_Rot(point_cloud, class_number, is_training=tf.cast(False, tf.bool), bn_decay=None):

	element_mean = tf.reduce_mean(point_cloud, axis=1)
	point_cloud_normalized = point_cloud - tf.expand_dims(element_mean, 1)
	point_cloud_normalized = point_cloud_normalized/1000.0
	
	cls_gt_onehot = tf.one_hot(indices=class_number, depth=CLASS_SIZE)
	cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
	cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

	with tf.variable_scope('Generator_Rot'):
		
		rot_net = point_net_model(tf.concat([point_cloud_normalized, cls_gt_onehot_tile], 2), 
									is_training, bn_decay=bn_decay)

		net1, _, _ = tf_util.fully_connected(rot_net, 2, activation_fn=tf.tanh, scope='output1_rot')
		net2, _, _ = tf_util.fully_connected(rot_net, 1, activation_fn=tf.sigmoid, scope='output2_rot')
	
		rot_pred = tf.concat([net1 * 90.0 * 1.05, net2 * 180.0 * 1.05], 1)

	return rot_pred

def Generator_Rot_Plus(point_cloud, class_number, is_training=tf.cast(False, tf.bool), bn_decay=None):

	element_mean = tf.reduce_mean(point_cloud, axis=1)
	point_cloud_normalized = point_cloud - tf.expand_dims(element_mean, 1)
	point_cloud_normalized = point_cloud_normalized/1000.0

	cls_gt_onehot = tf.one_hot(indices=class_number, depth=CLASS_SIZE)
	cls_gt_onehot_expand = tf.expand_dims(cls_gt_onehot, axis=1)
	cls_gt_onehot_tile = tf.tile(cls_gt_onehot_expand, [1, NUM_POINT, 1])

	with tf.variable_scope('Generator_Rot_Plus'):
		
		rot_plus_net = point_net_model(tf.concat([point_cloud_normalized, cls_gt_onehot_tile], 2), 
										is_training, bn_decay=bn_decay)

		rot_plus_pred, _, _ = tf_util.fully_connected(rot_plus_net, 2, activation_fn=tf.nn.softmax, scope='output_rot_plus')

	return rot_plus_pred

def restore():
	saver = tf.train.Saver()
	saver.restore(session, '/home/zju-taku/catkin_ws/src/test_point_cloud/6d_pose_params/params.ckpt')

def obtain_pose(data):
	'''obtain_6d_pose Callback Function'''
	
	_rpc, _rcn = get_data(data)
	grt, grr, grrp = session.run([gen_robot_trans, gen_robot_rot, tf.argmax(gen_robot_rot_plus, 1)], 
								feed_dict={point_cloud: _rpc, class_number:_rcn})
	
	if grrp[0] == 1: grr[0, -1] = grr[0, -1] * -1.
	trajectory = np.append(grt[0], grr[0])
	pub.publish(trajectory)

def main():
	'''obtain_6d_pose Subscriber'''
	session.run(tf.global_variables_initializer())
	restore()

	rospy.init_node('obtain_6d_pose', anonymous=True)
	rospy.Subscriber('point_clouds_cls', PointCloudCls,  obtain_pose)
	rospy.spin()

if __name__ == '__main__':
	
	session = tf.Session(config=config)
	point_cloud = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
	robot_trajectory = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 6))
	robot_trajectory_rot = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2))
	class_number = tf.placeholder(tf.int32, shape=BATCH_SIZE)

	gen_robot_trans = Generator_Trans(point_cloud, class_number)
	gen_robot_rot = Generator_Rot(point_cloud, class_number)
	gen_robot_rot_plus = Generator_Rot_Plus(point_cloud, class_number)

	pub = rospy.Publisher('robot_trajectory', Trajectory, queue_size=2)
	main()