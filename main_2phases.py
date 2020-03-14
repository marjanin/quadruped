import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
from all_functions import *

## initialization
dt=0.01 # time step
#.65 0-4
#.37   4-4
#.46 0-8
#.62   4-0
babbling = False
experiment_ID = "0001"
if babbling:
	#phase 1 - babbling on air
	MuJoCo_model_name = "tendon_quadruped_ws_onair.xml"
	babbling_signal_duration_in_seconds=4*60 # babbling duration
	np.random.seed(0) # setting the seed for numpy's random number generator
	## generating babbling data
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=8,
		signal_duration_in_seconds=babbling_signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=-1,
		dt=dt)
	## running the babbling data through the plant
	est_activations = babbling_signals
	[babbling_kinematics_p1, real_attempt_sensorreads_p1, babbling_activations_p1] = run_activations_ws_ol_fcn(
	MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=False) # this should be ol
	# phase 2 - babblin on floor
	MuJoCo_model_name = "tendon_quadruped_ws_onfloor.xml"
	babbling_signal_duration_in_seconds=4*60 # babbling duration
	np.random.seed(1) # setting the seed for numpy's random number generator
	## generating babbling data
	babbling_signals = babbling_input_gen_fcn(
		number_of_signals=8,
		signal_duration_in_seconds=babbling_signal_duration_in_seconds,
		pass_chance=dt,
		max_in=1,
		min_in=-1,
		dt=dt)
	## running the babbling data through the plant
	est_activations = babbling_signals
	#import pdb; pdb.set_trace()
	[babbling_kinematics_p2, real_attempt_sensorreads_p2, babbling_activations_p2] = run_activations_ws_ol_fcn(
	MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=True) # this should be ol
	# concatinating babbling data from two phases
	babbling_kinematics = np.concatenate((babbling_kinematics_p1, babbling_kinematics_p2),axis=0)
	real_attempt_sensorreads = np.concatenate((real_attempt_sensorreads_p1, real_attempt_sensorreads_p2),axis=0)
	babbling_activations = np.concatenate((babbling_activations_p1, babbling_activations_p2),axis=0)
	# training the neural network
	#	real_attempt_sensorreads = []
	
	Inverse_ANN_model = inverse_mapping_ws_fcn(
		babbling_kinematics, real_attempt_sensorreads, babbling_activations, log_address="./log/save/{}/".format(experiment_ID), early_stopping=False)
	#saving the Inverse_ANN
	os.makedirs("./models/{}".format(experiment_ID), exist_ok=True)
	Inverse_ANN_model.save("./models/{}/Inverse_ANN_model".format(experiment_ID))
else:
	#loading the Inverse_ANN
	Inverse_ANN_model = tf.keras.models.load_model("./models/{}/Inverse_ANN_model".format(experiment_ID))
# creating the cyclical movement kinematics

MuJoCo_model_name = "tendon_quadruped_ws_onfloor.xml"
attempt_kinematics = create_cyclical_movements_fcn(omega = -3, attempt_length = 10, timestep = 0.01)
#import pdb; pdb.set_trace()
#kinematics to activations
#est_activations=Inverse_ANN_model.predict(np.concatenate((attempt_kinematics, 000*np.ones((1000,1))),axis=1))
#[returned_kinematics, real_attempt_sensorreads, returned_est_activations ] = run_activations_ws_ol_fcn(MuJoCo_model_name, est_activations, timestep=0.01, Mj_render=True) # this should be cl
[returned_kinematics, real_attempt_sensorreads, returned_est_activations ] = run_activations_ws_cl_fcn(
MuJoCo_model_name, Inverse_ANN_model, attempt_kinematics, timestep=0.01, Mj_render=True) # this should be cl

# running the activations created for the cyclical movements
#MuJoCo_model_name = "single_leg_ws_onfloor.xml"
# calculating the RSME0
RMSE = np.sqrt(np.mean(np.square((returned_kinematics[:,:2]-attempt_kinematics[:,:2]))))
print("RMSE:", RMSE)

#import pdb; pdb.set_trace()
