# -*- coding: utf-8 -*-
import numpy as np
from open3d import *
import tensorflow as tf
import random
import tf_util

DATA_ROOT = "trainning_data/"
NUM_POINT = 368  # Number of point
BATCH_SIZE = 5  # Batch size
NUM_CLASS = 3
ITERS = 30000  # How many generator iterations to train for

Learning_Rate = 1e-3
decay_rate = 0.85  
decay_steps = 3000 

def rotate_point (point, rotation_angle):
	point = np.array(point)
	cos_theta = np.cos(rotation_angle)
	sin_theta = np.sin(rotation_angle)
	rotation_matrix = np.array([[cos_theta, sin_theta, 0],
								[-sin_theta, cos_theta, 0],
								[0, 0, 1]])
	rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
	return rotated_point

def jitter_point(point, sigma=0.001, clip=0.05):
	assert(clip > 0)
	point = np.array(point)
	point = point.reshape(-1,3)
	Row, Col = point.shape
	jittered_point = np.clip(sigma * np.random.randn(Row, Col), -1*clip, clip)
	jittered_point += point
	return jittered_point

def standardization(data):
	data = np.array(data)
	mu = np.mean(data, axis=0)
	sigma = np.std(data, axis=0)
	return (data - mu) / sigma

def get_cls_data():

	real_point_cloud = []
	real_label = []

	for i in range(BATCH_SIZE):
		label = np.zeros([NUM_CLASS], dtype='float32')
		batch = random.randint(1, 30)
		
		if batch <= 10: 
			label[0] = 1.
		elif batch <= 20:
			label[1] = 1.
		else:
			label[2] = 1.

		if (batch < 10):
			batch = "{}{}".format("0", batch)
		else:
			batch = "{}".format(batch)

		pcd = read_point_cloud(DATA_ROOT + batch + '.pcd')
		point_cloud = np.asarray(pcd.points)
		point_cloud = np.asarray(random.sample(list(point_cloud), NUM_POINT), dtype='float32')
		point_cloud = jitter_point(rotate_point(standardization(point_cloud), 2 * np.pi * np.random.random()))

		real_point_cloud.append(point_cloud)
		real_label.append(label)

	return real_point_cloud, real_label

def get_cls_model(point_cloud, is_training=tf.cast(True, tf.bool), bn_decay=None):
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

def save(iteration):
	saver = tf.train.Saver()
	saver.save(session, 'params/cls_model.ckpt'.format(iteration), write_meta_graph=False)

def restore():
	saver = tf.train.Saver()
	saver.restore(session, 'params/cls_model.ckpt')


point_cloud = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
label = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASS))

predict = get_cls_model(point_cloud)

classify_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
								logits=predict, labels=label))

global_step = tf.Variable(tf.constant(0))
learning_rate = tf.train.exponential_decay(Learning_Rate, global_step, decay_steps, decay_rate, staircase=False)

cls_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(classify_loss, global_step=global_step)

# train_x, train_y = get_cls_data()

# Train loop
with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	#restore()

	for iteration in range(ITERS):
		train_x, train_y = get_cls_data()

		session.run(cls_op, feed_dict={point_cloud: train_x, label: train_y})
				
		if iteration % 1000 == 999:
			save()
			print('iteration = {}'.format(iteration))