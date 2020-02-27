import sklearn
import tensorflow as tf
import numpy as np
import pickle
from scipy import signal

def LP_filt(filter_length, x):
	"""
	Finite Impulse Response (FIR) Moving Average (MA) Low-Pass Filter
	"""
	b=np.ones(filter_length,)/(filter_length) #Finite Impulse Response (FIR) Moving Average (MA) filter with one second filter length
	a=1
	y = signal.filtfilt(b, a, x)
	return y

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

def inverse_mapping_fcn(kinematics, activations, log_address='./log', early_stopping=False, **kwargs):
	"""
	this function used the babbling data to create an inverse mapping using an MLP NN
	"""
	# initialization
	x = kinematics
	y = activations
	x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
	log_address
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_address)
	
	if ("prior_model" in kwargs):
		model = kwargs["prior_model"]
		history = \
		model.fit(
		x_train,
		y_train,
		epochs=5,
		validation_data=(x_valid, y_valid),
		callbacks=[tensorboard_callback])
		with open(log_address+'/trainHistoryDict.pickle', 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
	else:
		model = tf.keras.Sequential()
		# Adds a densely-connected layer with 15 units to the model:
		model.add(tf.keras.layers.Dense(15, activation='relu'))
		# Add a softmax layer with 3 output units:
		model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
		model.compile(optimizer=tf.train.AdamOptimizer(0.01),
	              loss='mse',       # mean squared error
	              metrics=['mse'])  # mean squared error
		#training the model
		history = \
		model.fit(
		x_train,
		y_train,
		epochs=20,
		validation_data=(x_valid, y_valid),
		callbacks=[tensorboard_callback])
		with open(log_address+'/trainHistoryDict.pickle', 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
		tf.keras.utils.plot_model(model, to_file=log_address+'/model.png')
	# running the model
	#est_activations=model.predict(kinematics)
	return model

	#import pdb; pdb.set_trace()
