#!/usr/bin/env python
'''classfication ROS Node'''
import rospy
import numpy as np
import tensorflow as tf
import random
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import tf_util
from test_point_cloud.msg import PointCloudCls

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True

NUM_POINT = 300  # Number of point (can be change)
BATCH_SIZE = 1  # Batch size
NUM_CLASS = 3 # can be change

def standardization(data):
	data = np.array(data)
	mu = np.mean(data, axis=0)
	sigma = np.std(data, axis=0)
	return (data - mu) / sigma

def get_cls_data(data):
	
	assert isinstance(data, PointCloud2)
	cloud = point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
	point_cloud = np.asarray(list(cloud))
	real_point_cloud = np.asarray(random.sample(list(point_cloud), NUM_POINT), dtype='float32')
	real_point_cloud = standardization(real_point_cloud)
	real_point_cloud = real_point_cloud[np.newaxis, :]

	return real_point_cloud, data

def get_cls_model(point_cloud, is_training=tf.cast(False, tf.bool), bn_decay=None):
	""" Classification PointNet, input is BxNx3, output Bx40 """
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	input_image = tf.expand_dims(point_cloud, -1)

	# Point functions (MLP implemented as conv2d)
	net = tf_util.conv2d(input_image, 64, [1,3],
						 padding='SAME', stride=[1,1],
						 bn=True, is_training=is_training,
						 scope='conv1', bn_decay=bn_decay)

	net = tf_util.dropout(net, keep_prob=0.8, is_training=is_training,
						  scope='dp1')

	net = tf_util.conv2d(net, 64, [1,1],
						 padding='SAME', stride=[1,1],
						 bn=True, is_training=is_training,
						 scope='conv2', bn_decay=bn_decay)

	net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
						  scope='dp2')

	net = tf_util.conv2d(net, 64, [1,1],
						 padding='SAME', stride=[1,1],
						 bn=True, is_training=is_training,
						 scope='conv3', bn_decay=bn_decay)

	net = tf_util.dropout(net, keep_prob=0.8, is_training=is_training,
						  scope='dp3')

	net = tf_util.conv2d(net, 128, [1,1],
						 padding='SAME', stride=[1,1],
						 bn=True, is_training=is_training,
						 scope='conv4', bn_decay=bn_decay)

	net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
						  scope='dp4')

	net = tf_util.conv2d(net, 1024, [1,1],
						 padding='SAME', stride=[1,1],
						 bn=True, is_training=is_training,
						 scope='conv5', bn_decay=bn_decay)

	# Symmetric function: max pooling
	net = tf_util.max_pool2d(net, [num_point,1],
							 padding='VALID', scope='maxpool')
	
	# MLP on global point cloud vector
	net = tf.reshape(net, [batch_size, -1])

	net, _, _ = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
								  scope='fc1', bn_decay=bn_decay)
	net, _, _ = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
								  scope='fc2', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.8, is_training=is_training,
						  scope='dp5')
	net, _, _ = tf_util.fully_connected(net, NUM_CLASS, activation_fn=tf.nn.softmax, scope='fc3')

	return net

def restore():
	saver = tf.train.Saver()
	saver.restore(session, '/home/zju-taku/catkin_ws/src/test_point_cloud/cls_params/cls_model.ckpt')

def cls_point_cloud(data):

	cloud, point_cloud_data = get_cls_data(data)
	predict_cls = session.run(tf.argmax(predict, 1), feed_dict={point_cloud: cloud})

	pub.publish(point_cloud_data, predict_cls[0])

def main():
	'''classfication Subscriber'''
	session.run(tf.global_variables_initializer())
	restore()

	rospy.init_node('classfication', anonymous=True)
	rospy.Subscriber("point_clouds", PointCloud2, cls_point_cloud)
	rospy.spin()

if __name__ == '__main__':

	session = tf.Session(config=config)
	point_cloud = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
	predict = get_cls_model(point_cloud)
	pub = rospy.Publisher('point_clouds_cls', PointCloudCls, queue_size=2)
	main()