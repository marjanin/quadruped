import pickle
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
import sklearn
import tensorflow as tf

def babbling_input_gen_fcn(number_of_signals, signal_duration_in_seconds, pass_chance, max_in, min_in, dt):
	"""
	babbling_input_gen_fcn generates the babbling input to be fed to the plant

	number_of_signals(int): number of the inputs of the plant
	signal_duration_in_seconds(int): duration of babbling in seconds
	pass_chance(float): the chance of the babbling signal to change at each step
	max_in and min_in (float and float): uper and lower bound for the babbling inputs
	dt(flaot): time step
	"""
	number_of_samples = int(np.round(signal_duration_in_seconds/dt))
	babbling_signals = np.zeros([number_of_samples,number_of_signals])
	gen_input = np.zeros(number_of_samples,)
	for jj in range(0, number_of_signals):
		for ii in range(0, number_of_samples):
			pass_rand = np.random.uniform(0,1,1)
			if pass_rand < pass_chance:
				gen_input[ii] = ((max_in-min_in)*np.random.uniform(0,1,1)) + min_in
			else:
				gen_input[ii] = gen_input[ii-1]
		babbling_signals[:,jj]=gen_input
	return babbling_signals

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
	    joint_names = ["rbthigh", "rbshin", "rfthigh", "rfshin", "lbthigh", "lbshin", "lfthigh", "lfshin"]
	    current_positions_array = np.zeros([len(joint_names),])
	    current_velocity_array = np.zeros([len(joint_names),])
	    current_acceleration_array = np.zeros([len(joint_names),])
	    for joint_name , joint_index in zip(joint_names, range(len(joint_names))):
	    	current_positions_array[joint_index] = sim.data.get_joint_qpos(joint_name)
	    	current_velocity_array[joint_index] = sim.data.get_joint_qvel(joint_name)
	    	current_acceleration_array = sim.data.qacc[-8:]
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

	hidden_layer_nodes = 24
	output_layer_nodes = activations.shape[1]
	# input_layer_nodes = determined from the input data


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
		model.add(tf.keras.layers.Dense(hidden_layer_nodes, activation='linear'))
		# Add a softmax layer with 3 output units:
		model.add(tf.keras.layers.Dense(output_layer_nodes, activation='linear'))
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
def sinusoidal_CPG_fcn(w = 1, phi = 0, lower_band = -1, upper_band = 1, attempt_length = 5 , timestep = 0.01):
	number_of_attempt_samples = int(np.round(attempt_length/timestep))
	q0 = np.zeros(number_of_attempt_samples)
	for ii in range(number_of_attempt_samples):
		q0[ii]=np.sin((2*np.pi*w*ii/(number_of_attempt_samples/attempt_length))+phi)
	q0 = (q0+1)/2 # normalize 0-1
	q0 = q0 * (upper_band-lower_band)
	q0 = q0 + lower_band
	return q0

def create_cyclical_movements_fcn(timestep = 0.01):
	omega = 2.5
	q0a = sinusoidal_CPG_fcn(w = omega, phi = 0, lower_band = -1, upper_band = .7, attempt_length = 5 , timestep = 0.01)
	q1a = sinusoidal_CPG_fcn(w = omega, phi = np.pi/2, lower_band = -1.2, upper_band = .87, attempt_length = 5 , timestep = 0.01)

	q0b = sinusoidal_CPG_fcn(w = omega, phi = np.pi, lower_band = -1, upper_band = .7, attempt_length = 5 , timestep = 0.01)
	q1b = sinusoidal_CPG_fcn(w = omega, phi = -np.pi, lower_band = -1.2, upper_band = .87, attempt_length = 5 , timestep = 0.01)

	attempt_kinematics_RB = positions_to_kinematics_fcn(q0a, q1a, timestep)
	attempt_kinematics_RF = positions_to_kinematics_fcn(q0b, q1b, timestep)
	attempt_kinematics_LB = positions_to_kinematics_fcn(q0b, q1b, timestep)
	attempt_kinematics_LF = positions_to_kinematics_fcn(q0a, q1a, timestep)
	attempt_kinematics = combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF)
	return attempt_kinematics

def create_sin_cos_kinematics_fcn(attempt_length = 5 , number_of_cycles = 4, timestep = 0.01):
	"""
	this function creates desired task kinematics and their corresponding 
	actuation values predicted using the inverse mapping
	"""
	#attempt_length=5 # in seconds
	number_of_attempt_samples=int(np.round(attempt_length/timestep))

	q0=np.zeros(number_of_attempt_samples)
	q1=np.zeros(number_of_attempt_samples)

	for ii in range(number_of_attempt_samples):
		q0[ii]=(np.pi/3)*np.sin(number_of_cycles*(2*np.pi*ii/number_of_attempt_samples))
		q1[ii]=-1*(np.pi/2)*((-1*np.cos(number_of_cycles*(2*np.pi*ii/number_of_attempt_samples))+1)/2)
	#import pdb; pdb.set_trace()
	attempt_kinematics_RB = positions_to_kinematics_fcn(q0, q1, timestep)
	attempt_kinematics_RF = positions_to_kinematics_fcn(q0, q1, timestep)
	attempt_kinematics_LB = positions_to_kinematics_fcn(q0, q1, timestep)
	attempt_kinematics_LF = positions_to_kinematics_fcn(q0, q1, timestep)
	attempt_kinematics = combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF)
	#np.save("attempt_kinematics",attempt_kinematics)
	#np.save("est_task_activations",est_attempt_activations)
	#import pdb; pdb.set_trace()
	return attempt_kinematics


def positions_to_kinematics_fcn(q0, q1, timestep = 0.01):
	kinematics=np.transpose(
	np.concatenate(
		(
			[[q0],
			[q1],
			[np.gradient(q0)/timestep],
			[np.gradient(q1)/timestep],
			[np.gradient(np.gradient(q0)/timestep)/timestep],
			[np.gradient(np.gradient(q1)/timestep)/timestep]]),
		axis=0
		)
	)
	return kinematics

def combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF):
	attempt_kinematics = np.concatenate(
		(
			attempt_kinematics_RB[:,0:2], attempt_kinematics_RF[:,0:2], attempt_kinematics_LB[:,0:2], attempt_kinematics_LF[:,0:2],
			attempt_kinematics_RB[:,2:4], attempt_kinematics_RF[:,2:4], attempt_kinematics_LB[:,2:4], attempt_kinematics_LF[:,2:4],
			attempt_kinematics_RB[:,4:6], attempt_kinematics_RF[:,4:6], attempt_kinematics_LB[:,4:6], attempt_kinematics_LF[:,4:6]
		),
		axis=1
	)
	return attempt_kinematics

#import pdb; pdb.set_trace()