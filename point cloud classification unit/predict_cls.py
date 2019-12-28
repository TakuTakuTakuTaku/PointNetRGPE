# -*- coding: utf-8 -*-
import numpy as np
from open3d import *
import tensorflow as tf
import random
import tf_util
import os

DATA_ROOT = "testing_data/"
# File_Name = '01.pcd'
NUM_POINT = 368  # Number of point
BATCH_SIZE = 30  # Batch size
NUM_CLASS = 3

def standardization(data):
	data = np.array(data)
	mu = np.mean(data, axis=0)
	sigma = np.std(data, axis=0)
	return (data - mu) / sigma

def get_cls_data():

	real_point_cloud = []
	real_label = []
	l = os.listdir(DATA_ROOT)
	for i in  l:
		pcd = read_point_cloud(DATA_ROOT + str(i))
		point_cloud = np.asarray(pcd.points)
		point_cloud = np.asarray(random.sample(list(point_cloud), NUM_POINT), dtype='float32')
		point_cloud = standardization(point_cloud)
		real_point_cloud.append(point_cloud)
		idx = int(i.split('.')[0]) 
		if idx <= 10:
			label = 0
		elif idx <= 20:
			label = 1
		else:
			label = 2
		real_label.append(label)
	return real_point_cloud, real_label

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
	saver.restore(session, 'params/cls_model.ckpt')

point_cloud = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))

predict = get_cls_model(point_cloud)

# Train loop
with tf.Session() as session:

	session.run(tf.global_variables_initializer())
	restore()

	test_x, test_y = get_cls_data()

	predict_cls = session.run(tf.argmax(predict, 1), feed_dict={point_cloud: test_x})
	print(list(predict_cls))
	print(test_y)
	predict_cls = np.array(predict_cls)
	test_y = np.array(test_y)
	print('test acc is ', (test_y == predict_cls).sum()/len(predict_cls))