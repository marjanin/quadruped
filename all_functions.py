import pickle
import numpy as np
from scipy import signal
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const
import sklearn
import sklearn.model_selection
import tensorflow as tf
import matplotlib.pyplot as plt

## executive functions
def babble_and_refine(MuJoCo_model_name, experiment_ID, run_no, kinematics_all, sensory_all, activations_all, number_of_refinements, random_seed, use_sensory=True, use_feedback=False, normalize=True, task_type="cyclical", ANN_structure="M", actuation_type="JD",use_acc=False, dt=.01):
	location_folder='./log/{}/'.format(experiment_ID.split('/',1)[0])
	[kinematics_norm_stndrd_vector_global, sensory_norm_stndrd_coef_global]=load_norm_stndrd_coefficients_fcn(location_folder)
	babbling = True
	number_of_legs = 4
	if ANN_structure == "M":
		number_of_ANNs = number_of_legs
	elif ANN_structure == "S":
		number_of_ANNs = 1
	else:
		ValueError("unacceptable ANN_structure")
	#import pdb; pdb.set_trace()
	if actuation_type == "TD":
		number_of_signals = 12
		min_in=0
	elif actuation_type == "JD":
		number_of_signals = 8
		min_in=-1
	else:
		ValueError("unacceptable actuation_type")

	ANNs = number_of_ANNs*[None]
	babbling_signal_duration_in_seconds= 1*60
	refinement_duration_in_seconds = 10
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=number_of_signals,
		signal_duration_in_seconds=babbling_signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=min_in, #### needs to be 0 for TD
		dt=dt)
	est_activations = babbling_signals
	if task_type == "cyclical":
		attempt_kinematics = create_cyclical_movements_fcn(omega = 1.5, attempt_length = refinement_duration_in_seconds, dt=dt)
	elif task_type == "p2p":
		filtfilt_N = 4
		attempt_kinematics = create_p2p_movements_fcn(random_seed=random_seed, number_of_steps = 10, attempt_length = refinement_duration_in_seconds, dt=dt, filtfilt_N=filtfilt_N)
	else:
		ValueError("unacceptable task")
	[babbling_kinematics, babbling_sensorreads, babbling_activations] = run_activations_ws_ol_fcn(
	MuJoCo_model_name, est_activations, Mj_render=False) # this should be ol
	errors = []
	if kinematics_all == []: # initialization
		kinematics_all = babbling_kinematics
		sensory_all = babbling_sensorreads
		activations_all = babbling_activations
		use_prior_model_babbling = False
	else:
		# asses the before babbling error
		[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_varANNs_fcn(
			MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), dt=dt, ANN_structure = ANN_structure, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, Mj_render=False, actuation_type=actuation_type, use_acc=use_acc) # this should be cl
		RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8])))) # RMSE on the last half of the trial
		print("Pre-babbling error>")
		print("RMSE:", RMSE)
		errors.append(RMSE)
		# concatinating all data
		kinematics_all = np.concatenate((kinematics_all,babbling_kinematics),axis=0)
		sensory_all = np.concatenate((sensory_all,babbling_sensorreads),axis=0)
		activations_all = np.concatenate((activations_all,babbling_activations),axis=0)
		use_prior_model_babbling = True

	Inverse_ANN_models = inverse_mapping_ws_varANNs_fcn(
	kinematics_all, sensory_all, activations_all, ANN_structure=ANN_structure, epochs=25, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, normalize=normalize, use_prior_model=use_prior_model_babbling, actuation_type=actuation_type, use_acc=use_acc) #
	
	
	for ii in range(number_of_refinements):
		[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_varANNs_fcn(
			MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), dt=dt, ANN_structure = ANN_structure, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, Mj_render=False, actuation_type=actuation_type, use_acc=use_acc) # this should be cl
		RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8])))) # RMSE on the last half of the trial
		print("Run #:", ii+1)
		print("RMSE:", RMSE)
		errors.append(RMSE)

		kinematics_all = np.concatenate((kinematics_all,returned_kinematics),axis=0)
		sensory_all = np.concatenate((sensory_all,returned_sensorreads),axis=0)
		activations_all = np.concatenate((activations_all,returned_est_activations),axis=0)

		Inverse_ANN_models = inverse_mapping_ws_varANNs_fcn(
		kinematics_all, sensory_all, activations_all, ANN_structure=ANN_structure, epochs=5, log_address="./log/{}/{}/".format(experiment_ID,run_no), use_sensory=use_sensory, normalize=normalize, use_prior_model=True, actuation_type=actuation_type, use_acc=use_acc) #
	# test_run (no-training or storing the data)
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_varANNs_fcn(
		MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), dt=dt, ANN_structure = ANN_structure, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, Mj_render=False, actuation_type=actuation_type, use_acc=use_acc) # this should be cl
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8]))))
	print("Run #:", number_of_refinements+1)
	print("RMSE:", RMSE)
	errors.append(RMSE)
	return errors, kinematics_all, sensory_all, activations_all

def test_a_task(MuJoCo_model_name, experiment_ID, run_no, random_seed, use_sensory=True, use_feedback=False, normalize=True, Mj_render=False, plot_position_curves=False, task_type = "cyclical", ANN_structure="M", dt=0.01, actuation_type="JD", use_acc=False):
	refinement_duration_in_seconds = 10
	if task_type == "cyclical":
		attempt_kinematics = create_cyclical_movements_fcn(omega = 1.5, attempt_length = refinement_duration_in_seconds, dt=dt)
	elif task_type == "p2p":
		filtfilt_N=4
		attempt_kinematics = create_p2p_movements_fcn(random_seed=random_seed, number_of_steps = 10, attempt_length = refinement_duration_in_seconds, dt=dt, filtfilt_N=filtfilt_N)
	else:
		ValueError("unacceptable task")
	[returned_kinematics, returned_sensorreads, returned_est_activations ] = run_activations_ws_cl_varANNs_fcn(
			MuJoCo_model_name, attempt_kinematics, log_address="./log/{}/{}/".format(experiment_ID,run_no), dt=dt, ANN_structure = ANN_structure, use_sensory=use_sensory, use_feedback=use_feedback, normalize=normalize, Mj_render=Mj_render, actuation_type=actuation_type, use_acc=use_acc) # this should be cl
	RMSE = np.sqrt(np.mean(np.square((returned_kinematics[int(returned_kinematics.shape[0]/2):,:8]-attempt_kinematics[int(attempt_kinematics.shape[0]/2):,:8])))) # RMSE on the last half of the trial
	if plot_position_curves:
		# import pdb; pdb.set_trace()
		fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4.2))
		axes[0].plot(np.arange(1000),returned_kinematics[int(returned_kinematics.shape[0]/2):,6], np.arange(1000), attempt_kinematics[int(attempt_kinematics.shape[0]/2):,6])
		axes[0].set_title('proximal')
		axes[1].plot(np.arange(1000), returned_kinematics[int(returned_kinematics.shape[0]/2):,7], np.arange(1000), attempt_kinematics[int(attempt_kinematics.shape[0]/2):,7])
		axes[1].set_title('distal')
		plt.show(block=True)

	return RMSE
## lower level functions
def babbling_input_gen_fcn(number_of_signals, signal_duration_in_seconds, pass_chance, max_in, min_in, dt=.01):
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
		
def run_activations_ws_ol_fcn(MuJoCo_model_name, est_activations, Mj_render=False):
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
	sensor_vector_length=sim.data.sensordata.__len__()
	number_of_legs = 4
	number_of_DoFs = number_of_legs*2#sim.data.qpos.__len__()
	number_of_task_samples=est_activations.shape[0]
	real_attempt_positions = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_velocities = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_accelerations = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_activations = np.zeros((number_of_task_samples,control_vector_length))
	real_attempt_sensorreads = np.zeros((number_of_task_samples,sensor_vector_length))
	chassis_pos=np.zeros(number_of_task_samples,)
	sim.set_state(sim_state)
	for ii in range(number_of_task_samples):
		sim.data.ctrl[:] = est_activations[ii,:]
		sim.step()
		# collecting kinematics (pos, vel, and acc) from all joints
		# joint_names = ["rbthigh", "rbshin"]
		current_positions_array = sim.data.qpos[-number_of_DoFs:]#current_positions_array[joint_index] = sim.data.get_joint_qpos(joint_name)
		current_velocity_array = sim.data.qvel[-number_of_DoFs:]#current_velocity_array[joint_index] = sim.data.get_joint_qvel(joint_name)
		current_acceleration_array = sim.data.qacc[-number_of_DoFs:]
		real_attempt_positions[ii,:] = current_positions_array
		real_attempt_velocities[ii,:] = current_velocity_array
		real_attempt_accelerations[ii,:] = current_acceleration_array
		real_attempt_activations[ii,:] = sim.data.ctrl
		real_attempt_sensorreads[ii,:] = sim.data.sensordata
		if Mj_render:
			viewer.render()
	real_attempt_kinematics = np.concatenate((real_attempt_positions, real_attempt_velocities, real_attempt_accelerations),axis=1)
	return real_attempt_kinematics, real_attempt_sensorreads, real_attempt_activations

def inverse_mapping_ws_varANNs_fcn(kinematics, sensorydata, activations, ANN_structure="M", epochs=25, log_address=None, use_sensory=True, normalize=True, use_prior_model=False, actuation_type = "JD", use_acc=False):
	"""
	this function used the babbling data to create an inverse mapping using a
	MLP NN
	"""
	# import pdb; pdb.set_trace()
	file_location='/'.join(log_address.split('/',3)[:3])+"/"
	[kinematics_norm_stndrd_vector_global, sensory_norm_stndrd_coef_global]=load_norm_stndrd_coefficients_fcn(file_location)
	# global kinematics_norm_stndrd_vector_global, sensory_norm_stndrd_coef_global
	number_of_legs = 4
	if ANN_structure == "M":
		number_of_ANNs = number_of_legs
		hidden_layer_nodes = 6
	elif ANN_structure == "S":
		number_of_ANNs = 1
		hidden_layer_nodes = 6*number_of_legs
	else:
		ValueError("unacceptable ANN_structure")
	output_layer_nodes = activations.shape[1]/number_of_ANNs
	ANNs = number_of_ANNs*[None]
	
	if normalize:
		normalization_vector = kinematics_norm_stndrd_vector_global
		sensory_normalization_coef = sensory_norm_stndrd_coef_global
	else:
		normalization_vector = 1
		sensory_normalization_coef = 1
	kinematics_normalized=np.squeeze(kinematics/normalization_vector)
	# input_layer_nodes = determined from the input data
	if ANN_structure == "M":
		for leg_number in range(number_of_legs):
			# normalization
			if use_acc:
				leg_kinematics = kinematics_normalized[:,[0+leg_number*2, 1+leg_number*2, 8+leg_number*2, 9+leg_number*2, 16+leg_number*2,17+leg_number*2]]
			else:
				leg_kinematics = kinematics_normalized[:,[0+leg_number*2, 1+leg_number*2, 8+leg_number*2, 9+leg_number*2]]
			if use_sensory:
				sensorydata_delayed = np.zeros(sensorydata.shape)
				sensorydata_delayed[1:,:] = sensorydata[:-1,:]/sensory_normalization_coef#/400 # this is needed since we use observed sensory when inputing desired kinematics
				x = np.concatenate((leg_kinematics, np.transpose(np.array([sensorydata_delayed[:,leg_number]]))),axis=1)   
			else:
				x = leg_kinematics
			if actuation_type == "TD":
				y = activations[:,0+leg_number*3:3+leg_number*3]
			elif actuation_type == "JD":
				y = activations[:,0+leg_number*2:2+leg_number*2]
			else:
				ValueError("unacceptable actuation_type")

			x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
			logdir = log_address+"leg_{}/".format(leg_number)
			tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
			earlystopping_callback = tf.keras.callbacks.EarlyStopping(
			monitor='val_loss', patience=5, verbose=1)
			checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
				logdir+"model", monitor='val_loss', verbose=1, save_best_only=True)
			if use_prior_model:
				model = tf.keras.models.load_model(logdir+"model",compile=False)
				model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
				loss='mse',	   # mean squared error
				metrics=['mse'])  # mean squared error
				history = \
				model.fit(
				x_train,
				y_train,
				epochs=epochs,
				validation_data=(x_valid, y_valid),
				callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
				# with open(logdir+'/trainHistoryDict.pickle', 'wb') as file_pi:
				# 	pickle.dump(history.history, file_pi)
			else:
				model = tf.keras.Sequential()
				# Adds a densely-connected layer with 15 units to the model:
				model.add(tf.keras.layers.Dense(hidden_layer_nodes, activation='linear', input_shape= x_train.shape[1:]))
				# Add a softmax layer with 3 output units:
				model.add(tf.keras.layers.Dense(output_layer_nodes, activation='linear'))
				model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
			              loss='mse',       # mean squared error
			              metrics=['mse'])  # mean squared error
				#training the model
				history = \
				model.fit(
				x_train,
				y_train,
				epochs=epochs,
				validation_data=(x_valid, y_valid),
				callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
				# with open(logdir+'/trainHistoryDict.pickle', 'wb') as file_pi:
				# 	pickle.dump(history.history, file_pi)
				#tf.keras.utils.plot_model(model, to_file='model.png')
			# running the model
			#est_activations=model.predict(kinematics)
			tf.keras.backend.clear_session() 
			model=tf.keras.models.load_model(logdir+"model",compile=False)
			ANNs[leg_number] = model
		return ANNs
	else: # ANN_structure == "S"
		if use_acc:
			kinematics_to_use = kinematics_normalized
		else:
			kinematics_to_use = kinematics_normalized[:,:-8]
		if use_sensory:
			sensorydata_delayed = np.zeros(sensorydata.shape)
			sensorydata_delayed[1:,:] = sensorydata[:-1,:]/sensory_normalization_coef # this is needed since we use observed sensory when inputing desired kinematics
			x = np.concatenate((kinematics_to_use, sensorydata_delayed), axis=1)   
		else:
			x = kinematics_to_use
		y = activations
		x_train_all, x_valid_all, y_train_all, y_valid_all = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
		# import pdb; pdb.set_trace()
		downsampling_factor = 10
		x_train=x_train_all[::downsampling_factor,:]
		x_valid=x_valid_all[::downsampling_factor,:]
		y_train=y_train_all[::downsampling_factor,:]
		y_valid=y_valid_all[::downsampling_factor,:]

		logdir = log_address+"compound/"
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
		earlystopping_callback = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss', patience=5, verbose=1)
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			logdir+"model", monitor='val_loss', verbose=1, save_best_only=True)
		if use_prior_model:
			model = tf.keras.models.load_model(logdir+"model",compile=False)
			model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
			loss='mse',	   # mean squared error
			metrics=['mse'])  # mean squared error
			history = \
			model.fit(
			x_train,
			y_train,
			epochs=epochs,
			validation_data=(x_valid, y_valid),
			callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
			# with open(logdir+'/trainHistoryDict.pickle', 'wb') as file_pi:
			# 	pickle.dump(history.history, file_pi)
		else:
			model = tf.keras.Sequential()
			# Adds a densely-connected layer with 15 units to the model:
			#model.add(tf.keras.layers.BatchNormalization(input_shape= x_train.shape[1:]))
			model.add(tf.keras.layers.Dense(hidden_layer_nodes, activation='linear', input_shape= x_train.shape[1:]))
			# Add a softmax layer with 3 output units:
			model.add(tf.keras.layers.Dense(output_layer_nodes, activation='linear'))
			model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
		              loss='mse',       # mean squared error
		              metrics=['mse'])  # mean squared error
			#training the model
			history = \
			model.fit(
			x_train,
			y_train,
			epochs=epochs,
			validation_data=(x_valid, y_valid),
			callbacks=[tensorboard_callback, earlystopping_callback, checkpoint_callback])
			# with open(logdir+'/trainHistoryDict.pickle', 'wb') as file_pi:
			# 	pickle.dump(history.history, file_pi)
			#tf.keras.utils.plot_model(model, to_file='model.png')
		
		# running the model
		#est_activations=model.predict(kinematics)
		tf.keras.backend.clear_session() 
		ANNs=tf.keras.models.load_model(logdir+"model",compile=False)
	return ANNs

def run_activations_ws_cl_varANNs_fcn(MuJoCo_model_name, attempt_kinematics, log_address, dt, ANN_structure = "M", use_sensory=True, use_feedback=False, normalize=True, Mj_render=False, actuation_type="JD", use_acc=False):
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! the q0 is now the chasis pos. needs to be fixed
	"""
	this function runs the predicted activations generatred from running
	the inverse map on the desired task kinematics
	"""
	# import pdb; pdb.set_trace()
	file_location='/'.join(log_address.split('/',3)[:3])+"/"
	[kinematics_norm_stndrd_vector_global, sensory_norm_stndrd_coef_global]=load_norm_stndrd_coefficients_fcn(file_location)
	# global kinematics_norm_stndrd_vector_global, sensory_norm_stndrd_coef_global
	if normalize:
		kinematics_norm_stndrd_vector = kinematics_norm_stndrd_vector_global
		sensory_norm_stndrd_coef = sensory_norm_stndrd_coef_global
	else:
		kinematics_norm_stndrd_vector = 1
		sensory_norm_stndrd_coef = 1

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
	sensor_vector_length=sim.data.sensordata.__len__()


	number_of_legs = 4
	number_of_DoFs = number_of_legs*2#sim.data.qpos.__len__()
	number_of_task_samples=attempt_kinematics.shape[0]
	real_attempt_positions = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_velocities = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_accelerations = np.zeros((number_of_task_samples,number_of_DoFs))
	real_attempt_activations = np.zeros((number_of_task_samples,control_vector_length))
	real_attempt_sensorreads = np.zeros((number_of_task_samples,sensor_vector_length))
	chassis_pos=np.zeros(number_of_task_samples,)
	sim.set_state(sim_state)
	tf.keras.backend.clear_session() 

	if ANN_structure=="M":
		Inverse_ANN_models = number_of_legs*[None]
		for leg_number in range(number_of_legs):
			logdir = log_address+"leg_{}/".format(leg_number)
			Inverse_ANN_models[leg_number] = tf.keras.models.load_model(logdir+"model",compile=False)
		for ii in range(number_of_task_samples):
			if ii == 0:
				last_sensorydata = np.array([0, 0, 0, 0])
				current_control_kinematics = attempt_kinematics[ii,:]
				p_vec_error = 0
				p_vec_error_integ = 0
			else:
				last_sensorydata = sim.data.sensordata/sensory_norm_stndrd_coef
				if use_feedback:
					[current_control_kinematics, p_vec_error, p_vec_error_integ] = \
					create_control_kinematics_fcn(attempt_kinematics[ii,:], sim.data, number_of_DoFs, p_vec_error, p_vec_error_integ, dt=dt)
				else:
					current_control_kinematics = attempt_kinematics[ii,:]
			current_control_kinematics_normalized = np.squeeze(current_control_kinematics/kinematics_norm_stndrd_vector)
			#import pdb; pdb.set_trace()
			for leg_number in range(number_of_legs):
				Inverse_ANN_model = Inverse_ANN_models[leg_number]
				if use_sensory:
					if use_acc:
						input_data = np.append(current_control_kinematics_normalized[[0+leg_number*2, 1+leg_number*2, 8+leg_number*2, 9+leg_number*2, 16+leg_number*2,17+leg_number*2]],last_sensorydata[leg_number])
					else:
						input_data = np.append(current_control_kinematics_normalized[[0+leg_number*2, 1+leg_number*2, 8+leg_number*2, 9+leg_number*2]],last_sensorydata[leg_number])
				else:
					if use_acc:
						input_data = current_control_kinematics_normalized[[0+leg_number*2, 1+leg_number*2, 8+leg_number*2, 9+leg_number*2, 16+leg_number*2,17+leg_number*2]]
					else:
						input_data = current_control_kinematics_normalized[[0+leg_number*2, 1+leg_number*2, 8+leg_number*2, 9+leg_number*2]]
				if actuation_type == "TD":
					sim.data.ctrl[0+3*leg_number:3+3*leg_number] = Inverse_ANN_model.predict(np.expand_dims(input_data, axis=0))
				elif actuation_type == "JD":
					sim.data.ctrl[0+2*leg_number:2+2*leg_number] = Inverse_ANN_model.predict(np.expand_dims(input_data, axis=0))
				else:
					ValueError("unacceptable actuation_type")
			sim.step()
			joint_names = ["rbthigh", "rbshin"]
			current_positions_array = np.zeros([len(joint_names),])
			current_velocity_array = np.zeros([len(joint_names),])
			current_acceleration_array = np.zeros([len(joint_names),])
			current_positions_array = sim.data.qpos[-number_of_DoFs:]#current_positions_array[joint_index] = sim.data.get_joint_qpos(joint_name)
			current_velocity_array = sim.data.qvel[-number_of_DoFs:]#current_velocity_array[joint_index] = sim.data.get_joint_qvel(joint_name)
			current_acceleration_array = sim.data.qacc[-number_of_DoFs:]
			real_attempt_positions[ii,:] = current_positions_array
			real_attempt_velocities[ii,:] = current_velocity_array
			real_attempt_accelerations[ii,:] = current_acceleration_array
			real_attempt_activations[ii,:] = sim.data.ctrl
			real_attempt_sensorreads[ii,:] = sim.data.sensordata
			if Mj_render:
				viewer.render()

	elif ANN_structure=="S":
		logdir = log_address+"compound/"
		Inverse_ANN_models = tf.keras.models.load_model(logdir+"model",compile=False)
		for ii in range(number_of_task_samples):
			if ii == 0:
				last_sensorydata = np.array([0, 0, 0, 0])
				current_control_kinematics = attempt_kinematics[ii,:]
				p_vec_error = 0
				p_vec_error_integ = 0
			else:
				last_sensorydata = sim.data.sensordata/sensory_norm_stndrd_coef
				if use_feedback == True:
					[current_control_kinematics, p_vec_error, p_vec_error_integ] = \
					create_control_kinematics_fcn(attempt_kinematics[ii,:], sim.data, number_of_DoFs, p_vec_error, p_vec_error_integ, dt=dt)
				else:
					current_control_kinematics = attempt_kinematics[ii,:]
			current_control_kinematics_normalized = np.squeeze(current_control_kinematics/kinematics_norm_stndrd_vector)
			if use_acc:
				current_control_kinematics_to_use = current_control_kinematics_normalized
			else:
				current_control_kinematics_to_use = current_control_kinematics_normalized[:-8]
			if use_sensory:
				sim.data.ctrl[:] = Inverse_ANN_models.predict(np.array([np.concatenate((current_control_kinematics_to_use, last_sensorydata))]))
			else:
				sim.data.ctrl[:] = Inverse_ANN_models.predict(np.array([current_control_kinematics_to_use]))
			sim.step()
			# collecting kinematics (pos, vel, and acc) from all joints
			joint_names = ["rbthigh", "rbshin"]
			current_positions_array = np.zeros([len(joint_names),])
			current_velocity_array = np.zeros([len(joint_names),])
			current_acceleration_array = np.zeros([len(joint_names),]) ###
			#for joint_name , joint_index in zip(joint_names, range(len(joint_names))):
			current_positions_array = sim.data.qpos[-number_of_DoFs:]#current_positions_array[joint_index] = sim.data.get_joint_qpos(joint_name)
			current_velocity_array = sim.data.qvel[-number_of_DoFs:]#current_velocity_array[joint_index] = sim.data.get_joint_qvel(joint_name)
			current_acceleration_array = sim.data.qacc[-number_of_DoFs:]
			real_attempt_positions[ii,:] = current_positions_array
			real_attempt_velocities[ii,:] = current_velocity_array
			real_attempt_accelerations[ii,:] = current_acceleration_array
			real_attempt_activations[ii,:] = sim.data.ctrl
			real_attempt_sensorreads[ii,:] = sim.data.sensordata
			if Mj_render:
				viewer.render()
	else:
		ValueError("unacceptable ANN_structure")

	real_attempt_kinematics = np.concatenate((real_attempt_positions, real_attempt_velocities, real_attempt_accelerations),axis=1)
	return real_attempt_kinematics, real_attempt_sensorreads, real_attempt_activations

def sinusoidal_CPG_fcn(w = 1, phi = 0, lower_band = -1, upper_band = 1, attempt_length = 5 , dt=0.01):
	number_of_attempt_samples = int(np.round(attempt_length/dt))
	q0 = np.zeros(number_of_attempt_samples)
	for ii in range(number_of_attempt_samples):
		q0[ii]=np.sin((2*np.pi*w*ii/(number_of_attempt_samples/attempt_length))+phi)
	q0 = (q0+1)/2 # normalize 0-1
	q0 = q0 * (upper_band-lower_band)
	q0 = q0 + lower_band
	return q0

def p2p_positions_gen_fcn(lower_band, upper_band, number_of_positions, duration_of_each_position, random_seed, dt=.01):
	np.random.seed(random_seed)
	sample_no_of_each_position = duration_of_each_position / dt
	random_array = np.zeros(int(np.round(number_of_positions*sample_no_of_each_position)),)
	for ii in range(number_of_positions):
		random_value = ((upper_band-lower_band)*(np.random.rand(1)[0])) + lower_band
		random_array_1position = np.repeat(random_value,sample_no_of_each_position)
		random_array[int(ii*sample_no_of_each_position):int((ii+1)*sample_no_of_each_position)] = random_array_1position
	return random_array

def create_cyclical_movements_fcn(omega=1.5, attempt_length=10, dt=0.01):
	distance_from_limits=0.00
	q0a = sinusoidal_CPG_fcn(w = omega, phi = 0, lower_band = -.8+distance_from_limits, upper_band = -.3-distance_from_limits, attempt_length = attempt_length , dt=dt)
	q1a = sinusoidal_CPG_fcn(w = omega, phi = np.pi/2, lower_band = 0+distance_from_limits, upper_band = .45-distance_from_limits, attempt_length = attempt_length , dt=dt)

	q0b = sinusoidal_CPG_fcn(w = omega, phi = np.pi, lower_band = -.8+distance_from_limits, upper_band = -.3-distance_from_limits, attempt_length = attempt_length , dt=dt)
	q1b = sinusoidal_CPG_fcn(w = omega, phi = -np.pi/2, lower_band = 0+distance_from_limits, upper_band = .45-distance_from_limits, attempt_length = attempt_length , dt=dt)
	attempt_kinematics_RB = positions_to_kinematics_fcn(q0a, q1a, dt)
	# # plotting
	# plt.plot(q0a[0:100])
	# plt.title('Proximal')
	# plt.ylabel('Angle (rads)')
	# plt.xlabel('Sample #')
	# plt.figure()
	# plt.plot(q1a[0:100])
	# plt.title('Distal')
	# plt.ylabel('Angle (rads)')
	# plt.xlabel('Sample #')
	# plt.show(block=True)

	attempt_kinematics_RF = positions_to_kinematics_fcn(q0b, q1b, dt)
	attempt_kinematics_LB = positions_to_kinematics_fcn(q0b, q1b, dt)
	attempt_kinematics_LF = positions_to_kinematics_fcn(q0a, q1a, dt)
	attempt_kinematics = combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF)
	return attempt_kinematics

def create_p2p_movements_fcn(random_seed, number_of_steps = 10, attempt_length = 10, dt=0.01, filtfilt_N=1):
	step_duration = attempt_length/number_of_steps
	distance_from_limits=0.00
	random_seed_1=random_seed
	random_seed_2=np.random.randint(1000)
	q0a = p2p_positions_gen_fcn(lower_band =  -.8+distance_from_limits, upper_band = -.3-distance_from_limits, number_of_positions = number_of_steps, duration_of_each_position = step_duration, random_seed=random_seed_1, dt=dt)
	q1a = p2p_positions_gen_fcn(lower_band = 0+distance_from_limits, upper_band = .45-distance_from_limits, number_of_positions = number_of_steps, duration_of_each_position = step_duration, random_seed=random_seed_2, dt=dt)

	# q0b = p2p_positions_gen_fcn(lower_band = -.8+distance_from_limits, upper_band = -.3-distance_from_limits, number_of_positions = number_of_steps, duration_of_each_position = step_duration, random_seed=random_seed, dt=dt) # for second independent synergies
	# q1b = p2p_positions_gen_fcn(lower_band = 0+distance_from_limits, upper_band = .45-distance_from_limits, number_of_positions = number_of_steps, duration_of_each_position = step_duration, random_seed=random_seed, dt=dt) #  for second independent synergies
	
	if filtfilt_N>1:
		b=np.ones(filtfilt_N)/filtfilt_N
		q0a = signal.filtfilt(b,1,q0a)
		q1a = signal.filtfilt(b,1,q1a)
		q0b = signal.filtfilt(b,1,q0a)# all joints on the same angle
		q1b = signal.filtfilt(b,1,q1a)# all joints on the same angle

	attempt_kinematics_RB = positions_to_kinematics_fcn(q0a, q1a, dt)
	attempt_kinematics_RF = positions_to_kinematics_fcn(q0b, q1b, dt)
	attempt_kinematics_LB = positions_to_kinematics_fcn(q0b, q1b, dt)
	attempt_kinematics_LF = positions_to_kinematics_fcn(q0a, q1a, dt)
	attempt_kinematics = combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF)
	return attempt_kinematics


def positions_to_kinematics_fcn(q0, q1, dt=0.01):
	kinematics=np.transpose(
	np.concatenate(
		(
			[[q0],
			[q1],
			[np.gradient(q0)/dt],
			[np.gradient(q1)/dt],
			[np.gradient(np.gradient(q0)/dt)/dt],
			[np.gradient(np.gradient(q1)/dt)/dt]]),
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

def create_control_kinematics_fcn(current_desired_kinematics, current_sim_data, number_of_DoFs, prev_p_vec_error, prev_p_vec_error_int, dt, P=4, I=5, D=0.05):
#	current_sim_kinematics = pva2kin_fcn(current_sim_data.qpos, current_sim_data.qvel, current_sim_data.qacc, number_of_DoFs)
	[current_desired_p, current_desired_v, current_desired_a] = kin2pva_fcn(current_desired_kinematics, number_of_DoFs)
	p_vec_error = current_desired_p - current_sim_data.qpos[-number_of_DoFs:]
	p_vec_error_integ = prev_p_vec_error_int + p_vec_error*dt
	p_vec_error_der = (p_vec_error-prev_p_vec_error)/dt
	current_control_v = current_desired_v + P*p_vec_error + I*p_vec_error_integ + D*p_vec_error_der
	current_control_kinematics = pva2kin_fcn(current_desired_p, current_control_v, current_desired_a, number_of_DoFs)
	#import pdb; pdb.set_trace()
	return current_control_kinematics, p_vec_error, p_vec_error_integ
	

def pva2kin_fcn(p_vec,v_vec,a_vec,number_of_DoFs):
	kinematics = np.concatenate((p_vec[-number_of_DoFs:], v_vec[-number_of_DoFs:], a_vec[-number_of_DoFs:]))
	return kinematics

def kin2pva_fcn(kinematics, number_of_DoFs):
	each_vec_length=int(len(kinematics)/3)
	p_vec_all = kinematics[:each_vec_length]
	v_vec_all = kinematics[each_vec_length:2*each_vec_length]
	a_vec_all = kinematics[-each_vec_length:]
	p_vec = p_vec_all[-number_of_DoFs:]
	v_vec = v_vec_all[-number_of_DoFs:]
	a_vec = a_vec_all[-number_of_DoFs:]
	return p_vec, v_vec, a_vec

def calculate_norm_stndrd_coefficients_fcn(
	MuJoCo_model_name,
	number_of_signals,
	signal_duration_in_seconds,
	pass_chance,
	max_in=1,
	min_in=0, #### needs to be 0 for TD
	dt=0.01):
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=number_of_signals,
		signal_duration_in_seconds=signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=min_in, #### needs to be 0 for TD
		dt=dt)
	est_activations = babbling_signals
	[babbling_kinematics, babbling_sensorreads, babbling_activations] = run_activations_ws_ol_fcn(
		MuJoCo_model_name, est_activations, Mj_render=False)

	kinematics_max_amplitudes=np.abs(babbling_kinematics).max(0)
	kinematics_stds=babbling_kinematics.std(0)
	sensory_max_amplitures=abs(babbling_sensorreads).max(0)
	sensory_max_ampliture=sensory_max_amplitures.max()
	sensory_stds=babbling_sensorreads.std(0)
	sensory_avg_std=sensory_stds.mean()
	norm_stndrd_coefficients={"kinematics_stds":kinematics_stds, "sensory_avg_std":sensory_avg_std}
	return norm_stndrd_coefficients

def load_norm_stndrd_coefficients_fcn(location_folder):
	file_locaiton=location_folder+"norm_stndrd_coefficients.npy"
	norm_stndrd_coefficients = np.load(file_locaiton,allow_pickle=True).item()
	kinematics_norm_stndrd_vector_global=norm_stndrd_coefficients["kinematics_stds"]
	sensory_norm_stndrd_coef_global=norm_stndrd_coefficients["sensory_avg_std"]
	if sensory_norm_stndrd_coef_global==0:
		sensory_norm_stndrd_coef_global=1
	return kinematics_norm_stndrd_vector_global, sensory_norm_stndrd_coef_global

#import pdb; pdb.set_trace()