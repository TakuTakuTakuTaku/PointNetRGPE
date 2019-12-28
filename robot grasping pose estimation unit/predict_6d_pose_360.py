# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from open3d import *
import tensorflow as tf
import random
import tf_util
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True

DATA_ROOT = "testing_data_3/"
NUM_POINT = 500  # Number of point
CLASS_SIZE = 3  # Class size
BATCH_SIZE = 90  # Batch size

real_class_number = [2] * BATCH_SIZE

def get_data():

	real_robot_trajectory = []
	real_point_cloud = []

	for i in range(BATCH_SIZE):
		
		if (i+1 < 10):
			file_name = "{}{}".format("0", i+1)
		else:
			file_name = "{}".format(i+1)

		pcd = read_point_cloud(DATA_ROOT + file_name + '.pcd')
		point_cloud = np.asarray(pcd.points)
		real_point_cloud_temp = np.asarray(random.sample(list(point_cloud), NUM_POINT), dtype='float32')

		
		Data_Points = open(DATA_ROOT + file_name + '.txt', 'r')
		robot_trajectory = np.zeros([6], dtype='float32')
		c = 0
		for line in Data_Points:
			robot_trajectory[c] = line
			c += 1
		Data_Points.close()
		real_robot_trajectory_temp = np.asarray(robot_trajectory, dtype='float32')

		real_point_cloud.append(real_point_cloud_temp)
		real_robot_trajectory.append(real_robot_trajectory_temp)

	return real_point_cloud, real_robot_trajectory

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
	
def Generator_Trans_Mean(point_cloud, class_number, is_training=tf.cast(False, tf.bool), bn_decay=None):
	
	element_mean = tf.reduce_mean(point_cloud, axis=1)
	cls_gt_onehot = tf.one_hot(indices=class_number, depth=CLASS_SIZE)
	
	with tf.variable_scope('Generator_Trans_Mean', reuse=tf.AUTO_REUSE):
		element_mean, _, _ = tf_util.fully_connected(tf.concat([element_mean, cls_gt_onehot], 1), 6, is_training=is_training, activation_fn=None, scope='output1')
		element_mean, _, _ = tf_util.fully_connected(element_mean, 3, is_training=is_training, activation_fn=None, scope='output2')

	return element_mean

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

		trans_pred_res, _, _ = tf_util.fully_connected(trans_pred_res, 3, is_training=is_training, activation_fn=None, scope='output')

	return trans_pred_res

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
	saver.restore(session, 'params/params.ckpt')

def get_trans_loss(robot_trajectory, robot_trans):
	trans_loss = np.sqrt(np.sum(np.square(robot_trajectory-robot_trans), axis=1))
	return trans_loss

def get_rot_loss(robot_trajectory, robot_rot):
	rot_loss = robot_trajectory - robot_rot

	for i in range(BATCH_SIZE):
		loss = np.abs(rot_loss[i,-1])
		loss = np.minimum(np.abs(360.0 - loss), loss)
		rot_loss[i,-1] = loss

	rot_loss = np.sqrt(np.sum(np.square(rot_loss), axis=1))
	return rot_loss

point_cloud = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
robot_trajectory = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 6))
robot_trajectory_rot = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2))
class_number = tf.placeholder(tf.int32, shape=BATCH_SIZE)

gen_robot_trans_res = Generator_Trans(point_cloud, class_number)
gen_robot_trans_mean = Generator_Trans_Mean(point_cloud, class_number)

gen_robot_trans = gen_robot_trans_res*1000 + gen_robot_trans_mean
gen_robot_rot = Generator_Rot(point_cloud, class_number)
gen_robot_rot_plus = Generator_Rot_Plus(point_cloud, class_number)

# Train loop
with tf.Session(config=config) as session:
	session.run(tf.global_variables_initializer())
	restore()

	_rpc, _rrt = get_data()
	_rcn = real_class_number

	grt, grr, grrp = session.run([gen_robot_trans, gen_robot_rot, tf.argmax(gen_robot_rot_plus, 1)], 
								feed_dict={point_cloud: _rpc, class_number:_rcn})


	for i in range(BATCH_SIZE):
		if grrp[i] == 1: grr[i, -1] = grr[i, -1] * -1.

	_rrt = np.asarray(_rrt)
	trans_loss = get_trans_loss(_rrt[:, :3], grt)
	rot_loss = get_rot_loss(_rrt[:, 3:], grr)
	trans_loss_mean = np.mean(trans_loss)
	rot_loss_mean = np.mean(rot_loss)
	x = np.linspace(-180, 176, 90, endpoint=True)
	y = rot_loss

	data = pd.DataFrame(np.random.randn(90,2), columns=['trans_loss','rot_loss'])
	data['trans_loss'] = trans_loss
	data['rot_loss'] = rot_loss
	data.to_csv('loss_360.csv')

	print('360 degrees trans_loss_mean:', trans_loss_mean)
	print('360 degrees rot_loss_mean:', rot_loss_mean)

	fig=plt.figure()
	plt.subplot(211)
	plt.plot(trans_loss,'b',lw = 1.5) # 蓝色的线
	plt.plot(trans_loss,'ro') #离散的点
	plt.grid(True)
	plt.ylim(0, 10)
	plt.xlabel('number')
	plt.ylabel('trans_loss')
	plt.title('360 degrees trans_loss')
	plt.subplot(212)
	plt.plot(x,y,'b',lw = 1.5) # 蓝色的线
	plt.plot(x,y,'ro') #离散的点
	plt.grid(True)
	plt.ylim(0, 185)
	plt.xlabel('degree')
	plt.ylabel('rot_loss')
	plt.title('360 degrees rot_loss')
	plt.show()