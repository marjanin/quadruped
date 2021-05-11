import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from ch5_visu_functions import *
#from all_functions import *

experiment_ID_base = 'cur4_xmlVer11_TD_V1'#mc1 error -> replaced with MC0 copy


curriculums = ["_E2H", "_H2E"]
ANN_structures = ["S","M"]
task_types = ["cyclical", "p2p"]

all_sensory_cases = [True, False]
use_feedback = False
use_acc = True
normalize = True


show_video = False
random_seed = 0

task_type = task_types[0]
curriculum = curriculums[0]
ANN_structure = ANN_structures[0]
number_of_refinements = 6+1
number_of_all_runs = 60
use_sensory=True

[learning_errors_all_1, task_errors_all_1] =\
loading_plotting_data_fcn(
	experiment_ID_base='cur4_xmlVer11_TD_V1',
	use_sensory=False,
	use_feedback=False,
	curriculum="_E2H",
	task_type="cyclical",
	ANN_structure="M",
	number_of_refinements=6+1,
	number_of_all_runs=60)
[learning_errors_all_2, task_errors_all_2] =\
loading_plotting_data_fcn(
	experiment_ID_base='cur4_xmlVer11_TD_V1',
	use_sensory=False,
	use_feedback=False,
	curriculum="_E2H",
	task_type="cyclical",
	ANN_structure="M",
	number_of_refinements=6+1,
	number_of_all_runs=60)
# import pdb; pdb.set_trace()
labels=["S", "M"]
fig1=compare_learning_error_plots_fcn(learning_errors_all_1, learning_errors_all_2, labels, curriculum="_E2H")
fig2=compare_task_error_plots_fcn(task_errors_all_1, task_errors_all_2, labels)
#import pdb; pdb.set_trace()
comparison_name="test"
save_figures = True
if save_figures:
	dpi = 600
	# fig1.subplots_adjust(left=.06, bottom=.12, right=.96, top=.92, wspace=.30, hspace=.20)
	fig1.savefig("./results/{}/{}_figure1.png".format(experiment_ID_base,comparison_name), dpi=dpi)
	#fig2.subplots_adjust(bottom=.12, top=.92)
	fig2.savefig("./results/{}/{}_figure2.png".format(experiment_ID_base,comparison_name), dpi=dpi)
plt.show(block=True)

# #import pdb; pdb.set_trace()

