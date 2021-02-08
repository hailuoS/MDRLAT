from __future__ import print_function
from __future__ import division
from Gazebo_BND import GazeboWorld

import tensorflow as tf
import random
import numpy as np
import time
import rospy
import copy
from collections import deque

GAME = 'GazeboWorld'
ACTIONS = 10 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.99 # decay rate of past observations
OBSERVE = -1. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.00 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 8 # size of minibatch
MAX_EPISODE = 50
MAX_T = 320000
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TAU = 0.001 # Rate to update target network toward primary network
H_SIZE = 8*10*32
LASER_H_SIZE = 32*32
IMAGE_HIST = 4
LASER_BEAM = 512
depthsize = 128
lasersize = 64
table = [[0.4, np.pi/12],[0.4, 0],[0.4, -np.pi/12],[0.2, np.pi/6],[0.2, -np.pi/6],[0.2, np.pi/12],[0.2, -np.pi/12],[0.2, 0],[0, np.pi/3],[0, -np.pi/3]]

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)
		
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial, name="weights")

def bias_variable(shape):
	initial = tf.constant(0., shape = shape)
	return tf.Variable(initial, name="bias")

def conv1d(x,W,stride):
	return tf.nn.conv1d(x,W,stride = stride, padding = "SAME")

def conv2d(x, W, stride_h, stride_w):
	return tf.nn.conv2d(x, W, strides = [1, stride_h, stride_w, 1], padding = "SAME")

def conv2d_transpose(x, W, stride_h, stride_w, output_shape):
	return tf.nn.conv2d_transpose(x, W, strides = [1, stride_h, stride_w, 1],output_shape = output_shape, padding = "SAME")



class QNetwork(object):
	"""docstring for ClassName"""
	def __init__(self, sess):
		# # laser_scan 256x1
		# with tf.name_scope("scan_conv1"):
			# scan_W_conv1 = weight_variable([LASER_BEAM*IMAGE_HIST, 256])
			# scan_b_conv1 = bias_variable([256])
		# with tf.name_scope("scan_conv2"):
			# scan_W_conv2 = weight_variable([256, 512])
			# scan_b_conv2 = bias_variable([512])
		with tf.name_scope("scan_conv1"):
			scan_W_conv1 = weight_variable([5, IMAGE_HIST, 16])
			scan_b_conv1 = bias_variable([16])
		with tf.name_scope("scan_conv2"):
			scan_W_conv2 = weight_variable([3, 16, 32])
			scan_b_conv2 = bias_variable([32])
		with tf.name_scope("scan_conv3"):
			scan_W_conv3 = weight_variable([3, 32, 32])
			scan_b_conv3 = bias_variable([32])
		with tf.name_scope("scan_fc1"):
			scan_W_fc1 = weight_variable([LASER_H_SIZE, lasersize])
			scan_b_fc1 = bias_variable([lasersize])
			
		# # depth input 128x160x1
		with tf.name_scope("Conv1"):
			W_conv1 = weight_variable([10, 14, IMAGE_HIST, 16])
			b_conv1 = bias_variable([16])
		# 16x20x32
		with tf.name_scope("Conv2"):
			W_conv2 = weight_variable([4, 4, 16, 32])
			b_conv2 = bias_variable([32])
		# 4x5x64
		with tf.name_scope("Conv3"):
			W_conv3 = weight_variable([3, 3, 32, 32])
			b_conv3 = bias_variable([32])
		with tf.name_scope("depth_fc1"):
			depth_W_fc1 = weight_variable([H_SIZE, depthsize])
			depth_b_fc1 = bias_variable([depthsize])


		with tf.name_scope("afterTF_fc"):
			afterTF_W = weight_variable([depthsize*lasersize, 512])
			afterTF_b = bias_variable([512])
			
		# # ego_predict
		with tf.name_scope("FC1"):
			W_FC1 = weight_variable([H_SIZE + LASER_H_SIZE + 512, 512])
			b_FC1 = bias_variable([512])
		with tf.name_scope("FC2"):
			W_FC2 = weight_variable([512, 6])
			b_FC2 = bias_variable([6])
		# FC ob value layer
		with tf.name_scope("FCValue"):
			W_value = weight_variable([H_SIZE + LASER_H_SIZE + 512, 512])
			b_value = bias_variable([512])

		with tf.name_scope("FCAdv"):
			W_adv = weight_variable([H_SIZE + LASER_H_SIZE + 512, 512])
			b_adv = bias_variable([512])

		with tf.name_scope("FCValueOut"):
			W_value_out = weight_variable([512, 1])
			b_value_out = bias_variable([1])

		with tf.name_scope("FCAdvOut"):
			W_adv_out = weight_variable([512, ACTIONS])
			b_adv_out = bias_variable([ACTIONS])


		# input layer
		self.state_depth = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, IMAGE_HIST])
		self.state_laser = tf.placeholder("float",[None,LASER_BEAM,IMAGE_HIST])
		self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
		self.egomotion = tf.placeholder(tf.float32, [None, 6])
		# laser conv1 layer
		scan_conv1 = tf.nn.relu(conv1d(self.state_laser, scan_W_conv1, 4) + scan_b_conv1)
		# laser conv2 layer
		scan_conv2 = tf.nn.relu(conv1d(scan_conv1,scan_W_conv2, 2) + scan_b_conv2)
		# laser conv3 layer
		scan_conv3 = tf.nn.relu(conv1d(scan_conv2,scan_W_conv3, 2) + scan_b_conv3)
		scan_conv3_flat = tf.reshape(scan_conv3,[-1, LASER_H_SIZE])
		scan_fc1 = tf.matmul(scan_conv3_flat, scan_W_fc1) + scan_b_fc1
		# depth Conv1 layer
		h_conv1 = tf.nn.relu(conv2d(self.state_depth, W_conv1, 8, 8) + b_conv1)
		# depth Conv2 layer
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2, 2) + b_conv2)
		# depth Conv3 layer
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1, 1) + b_conv3)
		h_conv3_flat = tf.reshape(h_conv3, [-1, H_SIZE])
		# # depth predict egomotion
		# h_conv4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_conv4) + b_conv4)
		# depth_pre_ego = tf.matmul(h_conv4, W_conv5) + b_conv5
		# # laser predict egomotion
		# h_conv6 = tf.nn.relu(tf.matmul(scan_conv3_flat, W_conv6) + b_conv6)
		# laser_pre_ego = tf.matmul(h_conv6, W_conv7) + b_conv7
		h_fc1 = tf.matmul(h_conv3_flat, depth_W_fc1) + depth_b_fc1
		tensorFusion =  tf.multiply(tf.expand_dims(scan_fc1, 1), tf.expand_dims(h_fc1, -1))
		h_stateBF = tf.reshape(tensorFusion, [-1, depthsize*lasersize])
		h_stateAF = tf.matmul(h_stateBF, afterTF_W) + afterTF_b
		# concat
		h_state = tf.concat([h_conv3_flat, h_stateAF, scan_conv3_flat], axis = 1)
		h_fc = tf.nn.relu(tf.matmul(h_state, W_FC1) + b_FC1)
		pre_ego = tf.matmul(h_fc, W_FC2) + b_FC2
		# FC ob value layer
		h_fc_value = tf.nn.relu(tf.matmul(h_state, W_value) + b_value)
		value = tf.matmul(h_fc_value, W_value_out) + b_value_out

		# FC ob adv layer
		h_fc_adv = tf.nn.relu(tf.matmul(h_state, W_adv) + b_adv)		
		advantage = tf.matmul(h_fc_adv, W_adv_out) + b_adv_out

		# Q = value + (adv - advAvg)
		advAvg = tf.expand_dims(tf.reduce_mean(advantage, axis=1), axis=1)
		advIdentifiable = tf.subtract(advantage, advAvg)
		self.readout = tf.add(value, advIdentifiable)

		# define the ob cost function
		self.predict_loss = tf.reduce_mean(tf.square(pre_ego - self.egomotion))
		self.a = tf.placeholder("float", [None, ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), axis=1)
		self.td_error = tf.square(self.y - self.readout_action)
		self.cost = tf.reduce_mean(self.td_error)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)
		self.train_depth_predict = tf.train.AdamOptimizer(1e-4).minimize(self.predict_loss)

def TestNetwork():
	sess = tf.InteractiveSession()
	print("multiTF_aux")
	# define q network
	with tf.device("/cpu:0"):
		with tf.name_scope("OnlineNetwork"):
			online_net = QNetwork(sess)
		with tf.name_scope("TargetNetwork"):
			target_net = QNetwork(sess)
	rospy.sleep(1.)

	# reward_var = tf.Variable(0., trainable=False)
	# reward_epi = tf.summary.scalar('reward', reward_var)
	# # define summary
	# merged_summary = tf.summary.merge_all()
	# summary_writer = tf.summary.FileWriter('./logs', sess.graph)

	# Initialize the World and variables
	env = GazeboWorld()
	print('Environment initialized')
	# init_op = tf.global_variables_initializer()
	# sess.run(init_op)

	# get the first state 
	depth_img_t1 = env.GetDepthImageObservation()
	depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=2)
	egomotion_t1 = env.Getegomotion()
	egomotions_t1 = np.stack((egomotion_t1, egomotion_t1, egomotion_t1), axis=1)
	s1 = env.GetLaserObservation()
	s_1 = np.stack((s1, s1, s1, s1), axis=1)
	terminal = False

	# saving and loading Q networks
	episode = 0
	q_net_params = []

	# Find variables of q network
	for variable in tf.trainable_variables():
		variable_name = variable.name
		if variable_name.find('OnlineNetwork') >= 0:
			q_net_params.append(variable)
	for variable in tf.trainable_variables():
		variable_name = variable.name
		if variable_name.find('TargetNetwork') >= 0:
			q_net_params.append(variable)
	# print("  [*] printing QNetwork variables")
	# for idx, v in enumerate(q_net_params):
		# print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
	q_net_saver = tf.train.Saver(q_net_params, max_to_keep=1)
	path = "/media/user/disk01/shl/D3QN_base/512beams/saved_networks/multiTF_aux/GazeboWorld-dqn-"
	for model_index in range(0, 10):
		checkpoint = path + str(400000-model_index*10000)
		# checkpoint = tf.train.get_checkpoint_state("/media/user/disk01/shl/D3QN_base/saved_networks/multiTF_aux/")
		# print('checkpoint:', checkpoint)
		# if checkpoint and checkpoint.model_checkpoint_path:
			# q_net_saver.restore(sess, checkpoint.model_checkpoint_path)
			# print("Q network successfully loaded:", checkpoint.model_checkpoint_path)
		# else:
			# print("Could not find old Q network weights")
		q_net_saver.restore(sess, checkpoint)
		print("Q network successfully loaded:", checkpoint)
		# start training
		epsilon = INITIAL_EPSILON
		r_epi = 0.
		r_cache = []
		random_flag = False
		t_max = 0
		t = 0
		rate = rospy.Rate(5)
		loop_time = time.time()
		last_loop_time = loop_time
		success = 0
		episode = 0
		while episode < MAX_EPISODE and not rospy.is_shutdown():
			episode+=1
			env.ResetWorld(episode % 4)
			#env.GenerateTargetPoint()
			#print (env.target_point[0], env.target_point[1])
			#target_distance = copy.deepcopy(env.distance)
			t = 0
			flag = 0
			r_epi = 0.
			terminal = False
			reset = False
			loop_time_buf = []
			action_index = 0
			w_t1 = 0
			while not reset and not rospy.is_shutdown():
				#depth_image
				depth_img_t1 = env.GetDepthImageObservation()
				depth_img_t1 = np.reshape(depth_img_t1, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
				depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:, :, :(IMAGE_HIST - 1)], axis=2)
				#laser_t
				s1 = env.GetLaserObservation()
				s_1 = np.append(np.reshape(s1, (LASER_BEAM, 1)), s_1[:, :(IMAGE_HIST - 1)], axis=1)
				#s__1 = np.reshape(s_1, (LASER_BEAM * IMAGE_HIST))
				lasers_t1 = s_1
				#egomotions
				egomotion_t1 = env.Getegomotion()
				egomotion_t1 = np.reshape(egomotion_t1, (2, 1))
				egomotions_t1 = np.append(egomotion_t1, egomotions_t1[:,:(IMAGE_HIST - 2)], axis=1)
				egos_t1 = np.reshape(egomotions_t1, 6)
				reward_t, terminal, reset = env.GetRewardAndTerminate(t, w_t1)
				depth_imgs_t = depth_imgs_t1
				lasers_t = lasers_t1
				w_t1 = env.GetSelfOdomeSpeed()[1]
				#target = target1
				egomotions_t = egomotions_t1
				egos_t = egos_t1
				# choose an action epsilon greedily
				a = sess.run(online_net.readout, feed_dict = {online_net.state_depth : [depth_imgs_t],online_net.state_laser : [lasers_t],online_net.egomotion : [egos_t],online_net.batch_size : 1})
				readout_t = a[0]
				a_t = np.zeros([ACTIONS])
				if episode <= OBSERVE:
					action_index = random.randrange(ACTIONS)
					a_t[action_index] = 1
				else:
					if random.random() <= epsilon:
						print("----------Random Action----------")
						action_index = random.randrange(ACTIONS)
						a_t[action_index] = 1
					else:
						action_index = np.argmax(readout_t)
						a_t[action_index] = 1
				# Control the agent
				env.Control(action_index)
				#print('action:', action_index)

				t += 1
				if t > 301:
					success += 1
					flag = 1
				last_loop_time = loop_time
				loop_time = time.time()
				rate.sleep()
#			if flag==1:
#				print(str(episode)+' '+'success')
#			else:
#				print(str(episode)+' '+'crash')
		success_rate = success / MAX_EPISODE
		print('Sucesse Rate: %.2f' % success_rate)


def main():
	TestNetwork()

if __name__ == "__main__":
	main()
