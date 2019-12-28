# -*- coding: utf-8 -*-
import numpy as np
from open3d import *
import tensorflow as tf
import random
import tf_util

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True

DATA_ROOT = "trainning_data/"
NUM_POINT = 500  # Number of point
CLASS_SIZE = 3  # Class size
BATCH_SIZE = 5 * CLASS_SIZE  # Batch size
ITERS = 60000   # How many generator iterations to train for

Learning_Rate_Tran = 5e-4  
decay_rate_tran = 0.85  
decay_steps_tran = 3000

Learning_Rate_Rot = 5e-4  
decay_rate_rot = 0.85  
decay_steps_rot = 3000

def get_real_data():

	real_point_cloud = []
	real_robot_trajectory = []
	real_robot_rot = []
	real_class_number = [] 

	for i in range(BATCH_SIZE):
		batch = random.randint(1, 49)
		
		if batch <= 12: 
			class_number = 0.
		elif batch <= 26:
			class_number = 1.
		else:
			class_number = 2.

		if (batch < 10):
			batch = "{}{}".format("0", batch)
		else:
			batch = "{}".format(batch)

		pcd = read_point_cloud(DATA_ROOT + batch + '.pcd')
		point_cloud = np.asarray(pcd.points)
		real_point_cloud_temp = np.asarray(random.sample(list(point_cloud), NUM_POINT), dtype='float32')

		Data_Points = open(DATA_ROOT + batch + '.txt', 'r')
		robot_trajectory = np.zeros([6], dtype='float32')
		robot_rot = np.zeros([2], dtype='float32')
		c = 0
		for line in Data_Points:
			robot_trajectory[c] = line
			c += 1
		Data_Points.close()
		real_robot_trajectory_temp = np.asarray(robot_trajectory, dtype='float32')

		if real_robot_trajectory_temp[-1] < 0:
			robot_rot[1] = 1.
		else:
			robot_rot[0] = 1.

		real_robot_trajectory_temp[-1] = np.abs(real_robot_trajectory_temp[-1])

		real_point_cloud.append(real_point_cloud_temp)
		real_robot_trajectory.append(real_robot_trajectory_temp)
		real_robot_rot.append(robot_rot)
		real_class_number.append(class_number)

	return real_point_cloud, real_robot_trajectory, real_robot_rot, real_class_number

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

def Generator_Trans_Mean(point_cloud, class_number, is_training=tf.cast(True, tf.bool), bn_decay=None):
	
	element_mean = tf.reduce_mean(point_cloud, axis=1)
	cls_gt_onehot = tf.one_hot(indices=class_number, depth=CLASS_SIZE)
	
	with tf.variable_scope('Generator_Trans_Mean', reuse=tf.AUTO_REUSE):
		element_mean, _, _ = tf_util.fully_connected(tf.concat([element_mean, cls_gt_onehot], 1), 6, is_training=is_training, activation_fn=None, scope='output1')
		element_mean, _, _ = tf_util.fully_connected(element_mean, 3, is_training=is_training, activation_fn=None, scope='output2')

	return element_mean

def Generator_Trans(point_cloud, class_number, is_training=tf.cast(True, tf.bool), bn_decay=None):

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

def Generator_Rot(point_cloud, class_number, is_training=tf.cast(True, tf.bool), bn_decay=None):

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

def Generator_Rot_Plus(point_cloud, class_number, is_training=tf.cast(True, tf.bool), bn_decay=None):

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

def save():
	saver = tf.train.Saver()
	saver.save(session, 'params/params.ckpt', write_meta_graph=False)

def restore():
	saver = tf.train.Saver()
	saver.restore(session, 'params/params.ckpt')

def get_trans_loss(robot_trajectory, robot_trans):
	trans_loss = tf.sqrt(tf.reduce_sum(tf.square(robot_trajectory - robot_trans), axis=1))
	return tf.reduce_mean(trans_loss)

def get_rot_loss(robot_trajectory, robot_rot):
	output = robot_trajectory - robot_rot
	output = tf.square(output)
	rot_loss = tf.sqrt(tf.reduce_sum(output, axis=1))

	return tf.reduce_mean(rot_loss)

point_cloud = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
robot_trajectory = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 6))
robot_trajectory_rot = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2))
class_number = tf.placeholder(tf.int32, shape=BATCH_SIZE)

gen_robot_trans_res = Generator_Trans(point_cloud, class_number)
gen_robot_trans_mean = Generator_Trans_Mean(point_cloud, class_number)

gen_robot_trans = gen_robot_trans_res*1000 + gen_robot_trans_mean
gen_robot_rot = Generator_Rot(point_cloud, class_number)
gen_robot_rot_plus = Generator_Rot_Plus(point_cloud, class_number)

trans_mean_loss = get_trans_loss(robot_trajectory[:, :3], gen_robot_trans_mean)
trans_loss = get_trans_loss(robot_trajectory[:, :3], gen_robot_trans)
rot_loss = get_rot_loss(robot_trajectory[:, 3:], gen_robot_rot)
rot_plus_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
								logits=gen_robot_rot_plus, labels=robot_trajectory_rot))

global_step_tran = tf.Variable(tf.constant(0))
learning_rate_tran = tf.train.exponential_decay(Learning_Rate_Tran, global_step_tran, decay_steps_tran, decay_rate_tran, staircase=False)

global_step_rot = tf.Variable(tf.constant(0))
learning_rate_rot = tf.train.exponential_decay(Learning_Rate_Rot, global_step_rot, decay_steps_rot, decay_rate_rot, staircase=False)

gen_trans_mean_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0., beta2=0.9).minimize(trans_mean_loss)
gen_trans_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_tran, beta1=0., beta2=0.9).minimize(trans_loss, global_step=global_step_tran)
gen_rot_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_rot, beta1=0., beta2=0.9).minimize(rot_loss, global_step=global_step_rot)
gen_rot_plus_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_rot, beta1=0., beta2=0.9).minimize(rot_plus_loss, global_step=global_step_rot)

# Train loop
with tf.Session(config=config) as session:
	session.run(tf.global_variables_initializer())
	#restore()

	for iteration in range(5000):
		_rpc, _rrt, _rrr, _rcn = get_real_data()
		session.run([gen_trans_mean_train_op], 
					feed_dict={point_cloud: _rpc, robot_trajectory: _rrt, class_number:_rcn})
		
		if iteration % 1000 == 999:
			print('iteration = {}'.format(iteration))

	for iteration in range(ITERS):
		_rpc, _rrt, _rrr, _rcn = get_real_data()
		session.run([gen_trans_train_op, gen_rot_train_op, gen_rot_plus_train_op], 
					feed_dict={point_cloud: _rpc, robot_trajectory: _rrt, robot_trajectory_rot: _rrr, class_number:_rcn})
		
		if iteration % 1000 == 999:
			save()
			print('iteration = {}'.format(iteration))