import pickle
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
import sklearn
import tensorflow as tf

def run_activations_fcn(MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=False):
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! the q0 is now the chasis pos. needs to be fixed
	"""
	this function runs the predicted activations generatred from running
	the inverse map on the desired task kinematics
	"""

	MuJoCo_model = load_model_from_path("./assets/"+MuJoCo_model_name)
	sim = MjSim(MuJoCo_model)
	if Mj_render:
		viewer = MjViewer(sim)
		# to move it to the mounted camera
		viewer.cam.fixedcamid += 1
		viewer.cam.type = const.CAMERA_FIXED
		# # to record the video
		# viewer._record_video = True
		# # recording path
		# viewer._video_path = "~/Documents/"+str(time.localtime()[3])+str(time.localtime()[4])+str(time.localtime()[5])
	sim_state = sim.get_state()
	control_vector_length=sim.data.ctrl.__len__()
	number_of_DoFs = sim.data.qpos.__len__()
	number_of_task_samples=est_activations.shape[0]
	real_attempt_positions = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_velocities = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_accelerations = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_activations = np.zeros((number_of_task_samples,control_vector_length))
	chassis_pos=np.zeros(number_of_task_samples,)
	sim.set_state(sim_state)
	for ii in range(number_of_task_samples):
	    sim.data.ctrl[:] = est_activations[ii,:]
	    sim.step()
	    # collecting kinematics (pos, vel, and acc) from all joints
	    current_positions_array = sim.data.qpos
	    current_velocity_array = sim.data.qvel
	    current_acceleration_array = sim.data.qacc
	    real_attempt_positions[ii,:] = current_positions_array
	    real_attempt_velocities[ii,:] = current_velocity_array
	    real_attempt_accelerations[ii,:] = current_acceleration_array
	    real_attempt_activations[ii,:] = sim.data.ctrl
	    if Mj_render:
	    	viewer.render()
	real_attempt_kinematics = np.concatenate((real_attempt_positions, real_attempt_velocities, real_attempt_accelerations),axis=1)
	return real_attempt_kinematics, real_attempt_activations


def copy_model_fcn(original_model):
	config=original_model.get_config()
	new_model=tf.keras.Sequential.from_config(config)
	new_model.set_weights(original_model.get_weights())
	new_model.compile(optimizer=tf.train.AdamOptimizer(0.01),loss='mse',metrics=['mse'])  # mean squared error
	return new_model

def inverse_mapping_fcn(kinematics, activations, log_address=None, early_stopping=False, **kwargs):
	"""
	this function used the babbling data to create an inverse mapping using a
	MLP NN
	"""
	x = kinematics
	y = activations
	x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
	
	logdir = log_address
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
	
	if ("prior_model" in kwargs):
		model = kwargs["prior_model"]
		history = \
		model.fit(
		x_train,
		y_train,
		epochs=5,
		validation_data=(x_valid, y_valid),
		callbacks=[tensorboard_callback])
		with open(logdir+'/trainHistoryDict.pickle', 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
	else:
		model = tf.keras.Sequential()
		# Adds a densely-connected layer with 15 units to the model:
		model.add(tf.keras.layers.Dense(24, activation='linear'))
		# Add a softmax layer with 3 output units:
		model.add(tf.keras.layers.Dense(8, activation='linear'))
		model.compile(optimizer=tf.train.AdamOptimizer(0.01),
	              loss='mse',       # mean squared error
	              metrics=['mse'])  # mean squared error
		#training the model
		history = \
		model.fit(
		x_train,
		y_train,
		epochs=100,
		validation_data=(x_valid, y_valid),
		callbacks=[tensorboard_callback])
		with open(logdir+'/trainHistoryDict.pickle', 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
		#tf.keras.utils.plot_model(model, to_file='model.png')
	
	# running the model
	#est_activations=model.predict(kinematics)
	return model

#import pdb; pdb.set_trace()
