import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from ch5_visu_functions import *
#from all_functions import *
experiment_ID_base = 'cur4_xmlVer11_TD_V1_testANNstruc'#mc1 error -> replaced with MC0 copy



number_of_refinements = 6+1
number_of_all_runs = 32

curricula = ["_E2H", "_H2E"]
ANN_structures = ["S","M"]
task_types = ["cyclical", "p2p"]

task_type = task_types[1]
curriculum = curricula[0]
ANN_structure = ANN_structures[0]
use_feedback = 1
use_sensory = 1

comparison_name="task_{}_cur_{}_stru_{}_fb_{}_sensory_{}".format(
	task_type,curriculum,"var",use_feedback,use_sensory)

labels=["sing.", "mult."]

[learning_errors_all_1, task_errors_all_1] =\
loading_plotting_data_fcn(
	experiment_ID_base=experiment_ID_base,
	use_sensory=use_sensory,
	use_feedback=use_feedback,
	curriculum=curriculum,
	task_type=task_type,
	ANN_structure=ANN_structure,
	number_of_refinements=number_of_refinements,
	number_of_all_runs=number_of_all_runs)
[learning_errors_all_2, task_errors_all_2] =\
loading_plotting_data_fcn(
	experiment_ID_base=experiment_ID_base,
	use_sensory=use_sensory,
	use_feedback=use_feedback,
	curriculum=curriculum,
	task_type=task_type,
	ANN_structure="M",
	number_of_refinements=number_of_refinements,
	number_of_all_runs=number_of_all_runs)
# import pdb; pdb.set_trace()
fig1=compare_learning_error_plots_fcn(learning_errors_all_1, learning_errors_all_2, labels, curriculum="_E2H")
fig2=compare_task_error_plots_fcn(task_errors_all_1, task_errors_all_2, labels)
#import pdb; pdb.set_trace()
save_figures = True
if save_figures:
	dpi = 600
	# fig1.subplots_adjust(left=.06, bottom=.12, right=.96, top=.92, wspace=.30, hspace=.20)
	fig1.savefig("./results/{}/{}_figure1.png".format(experiment_ID_base,comparison_name), dpi=dpi)
	#fig2.subplots_adjust(bottom=.12, top=.92)
	fig2.savefig("./results/{}/{}_figure2.png".format(experiment_ID_base,comparison_name), dpi=dpi)
plt.show(block=1)

# #import pdb; pdb.set_trace()

