import os
import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from all_functions import *

experiment_ID = "test_run_"
MuJoCo_model_name = "tendon_quadruped_ws_inair.xml"
run_no = 0
kinematics_all = sensory_all = activations_all = []
number_of_refinements = 2
use_sensory = True
task_type = "cyclical"
# train
babble_and_refine(
	MuJoCo_model_name,
	experiment_ID,
	run_no,
	kinematics_all,
	sensory_all,
	activations_all,
	number_of_refinements,
	use_sensory=use_sensory,
	ANN_structure="S",
	task_type=task_type)
# test
# _ = test_a_task(MuJoCo_model_name, experiment_ID, run_no, use_sensory=use_sensory,  Mj_render=True, task_type = task_type)

#import pdb; pdb.set_trace()
