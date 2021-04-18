# adapted from https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
# https://github.com/miloharper/visualise-neural-network

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
import tensorflow as tf
from scipy import stats

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()

def weight_threshold(weights_matrix, threshold = .7):
    W_abs=np.abs(weights_matrix)
    W_max = np.max(W_abs)
    W_abs_thresholded = W_abs
    num_of_zeros = 0
    for col in range(W_abs.shape[1]):
        for row in range(W_abs.shape[0]):
            if W_abs_thresholded[row,col] < threshold*W_max:
                W_abs_thresholded[row,col] = 0
                num_of_zeros+=1
    return W_abs_thresholded, num_of_zeros

def weight_threshold_colwise(weights_matrix, threshold = .7):
    W_abs=np.abs(weights_matrix)
    W_abs_thresholded = W_abs
    for col in range(W_abs.shape[1]):
        W_col = W_abs[:,col]
        W_col_max = np.max(W_col)
        for row in range(W_abs.shape[0]):
            if W_col[row] < threshold*W_col_max:
                W_abs_thresholded[row,col] = 0
    return W_abs_thresholded

def kin_normalization_fcn(weights_all_0_kinnormalized, num_joints = 8):
    kinematic_vec_length = 3
    for kin_order in range (kinematic_vec_length):
        sub_matrix = weights_all_0_kinnormalized[kin_order*num_joints:(kin_order+1)*num_joints][:]
        sub_max = np.max(sub_matrix)
        weights_all_0_kinnormalized[kin_order*num_joints:(kin_order+1)*num_joints][:] = sub_matrix / sub_max
    # for touch sensors
    kin_order = kinematic_vec_length
    #import pdb; pdb.set_trace()
    # if sensory exists
    # sub_matrix = weights_all_0_kinnormalized[kin_order*num_joints:][:]
    # sub_max = np.max(sub_matrix)
    # weights_all_0_kinnormalized[kin_order*num_joints:][:] = sub_matrix / sub_max
    return weights_all_0_kinnormalized

def sparsity_calculation_fcn(logdir, threshold, draw_ANN=False, num_joints = 8):
    Inverse_ANN_models = tf.keras.models.load_model(logdir+"model",compile=False)
    weights_all = Inverse_ANN_models.get_weights()
    tf.keras.backend.clear_session()
    num_touch_sensors = num_joints/2
    #import pdb; pdb.set_trace()
    weights_all_0_kinnormalized = kin_normalization_fcn(weights_all[0], num_joints = num_joints)
    [W_thresh_2, num_zeros_2] = weight_threshold(weights_all[2], threshold = threshold)
    [W_thresh_0, num_zeros_0] = weight_threshold(weights_all_0_kinnormalized[:][:], threshold = threshold) # no need to -1 if using no sensory data
    #W_thresh_0 : 28*24 --- W_thresh_2 : 23*8
    total_num_elements = np.prod(W_thresh_2.shape)+np.prod(W_thresh_0.shape)
    sparsity_ratio = (num_zeros_2+num_zeros_0)/total_num_elements
    # drawing the ANN
    if draw_ANN:
        network = NeuralNetwork()
        network.add_layer(W_thresh_2.shape[1], W_thresh_2/np.max(W_thresh_2)) # output layer
        network.add_layer(W_thresh_0.shape[1], W_thresh_0/np.max(W_thresh_0)) # hidden layer
        network.add_layer(W_thresh_0.shape[0]) # input layer
        network.draw()
    return sparsity_ratio

if __name__ == "__main__":
    experiment_ID_base = "cur3_V4_TD_fulltest1"
    ANN_structures = ["S","M"]
    cur = "_E2H"
    task_type="cyclical"
    just_plot = True

    thresholds = np.arange(0,1.1,.05)
    if not just_plot:
        for ANN_structure in ANN_structures:
            experiment_ID = "wo_Sensory_"+ANN_structure+"_ANN_"+task_type+cur
            save_log_path = experiment_ID_base+"/"+experiment_ID#
            total_run_no=50        
            means=np.zeros(len(thresholds))
            stds=np.zeros(len(thresholds))
            sparsity_ratios_all = np.zeros((total_run_no,len(thresholds)))
            for threshold, threshold_num in zip(thresholds, range(len(thresholds))):
                if ANN_structure=="M":
                    number_of_legs = 4
                    num_joints = 2
                    sparsity_ratios = np.zeros((number_of_legs, total_run_no))
                    std_sparsity_ratios = np.zeros((number_of_legs, total_run_no))
                    for leg in range(number_of_legs):
                        for run_no in range(total_run_no):
                            print("Threshold: {}, Leg: {}, MC_no: {}".format(threshold, leg, run_no))
                            log_address="./log/{}/{}/".format(save_log_path,run_no)
                            logdir = log_address+"leg_{}/".format(leg)
                            sparsity_ratios[leg, run_no] = sparsity_calculation_fcn(logdir, threshold, draw_ANN=False)
                            sparsity_ratios_all[run_no,threshold_num] = sparsity_calculation_fcn(logdir, threshold, draw_ANN=False)
                    mean_sparsity_ratios = sparsity_ratios.mean()
                    std_sparsity_ratios = sparsity_ratios.std(axis=1).mean() # calculated across mc runs and averaged across legs
                elif ANN_structure=="S":
                    num_joints = 8
                    sparsity_ratios = np.zeros(total_run_no)
                    std_sparsity_ratios = np.zeros(total_run_no)
                    for run_no in range(total_run_no):
                        print("Threshold: {}, MC_no: {}".format( threshold,run_no))
                        log_address="./log/{}/{}/".format(save_log_path,run_no)
                        logdir = log_address+"compound/"
                        sparsity_ratios[run_no] = sparsity_calculation_fcn(logdir, threshold, draw_ANN=False)
                        sparsity_ratios_all[run_no,threshold_num] = sparsity_calculation_fcn(logdir, threshold, draw_ANN=False)
                    mean_sparsity_ratios = sparsity_ratios.mean()
                    std_sparsity_ratios = sparsity_ratios.std()
                else:
                    ValueError("unacceptable ANN_structure")
                means[threshold_num] = mean_sparsity_ratios
                stds[threshold_num] = std_sparsity_ratios
            np.save("results/"+experiment_ID_base+"/sparsity_all_means_{}.npy".format(ANN_structure), means)
            np.save("results/"+experiment_ID_base+"/sparsity_all_stds_{}.npy".format(ANN_structure), stds)
            np.save("results/"+experiment_ID_base+"/sparsity_all_sparsities_{}.npy".format(ANN_structure), sparsity_ratios_all)

    means_S = np.load("results/"+experiment_ID_base+"/sparsity_all_means_S.npy")
    stds_S = np.load("results/"+experiment_ID_base+"/sparsity_all_stds_S.npy")
    means_M = np.load("results/"+experiment_ID_base+"/sparsity_all_means_M.npy")
    stds_M = np.load("results/"+experiment_ID_base+"/sparsity_all_stds_M.npy")
    sparsities_all_S = np.load("results/"+experiment_ID_base+"/sparsity_all_sparsities_S.npy")
    sparsities_all_M = np.load("results/"+experiment_ID_base+"/sparsity_all_sparsities_M.npy")
    
    p_values = np.zeros(len(thresholds))
    F_values = np.zeros(len(thresholds))

    pyplot.figure(figsize = (7,5))
    pyplot.errorbar(thresholds[3:-2],1-means_S[3:-2], yerr=stds_S[3:-2], capsize=3)
    pyplot.errorbar(thresholds[3:-2],1-means_M[3:-2], yerr=stds_M[3:-2], capsize=3)
    pyplot.legend(["single ANN","multiple ANNs"])
    # pyplot.errorbar(thresholds[3:-2],(1-means_M[3:-2])/(1-means_S[3:-2]), capsize=3)

    # pyplot.figure()
    for threshold_num in range(3, len(thresholds)-2):
        [F_values[threshold_num], p_values[threshold_num]] = stats.f_oneway(sparsities_all_S[:,threshold_num],sparsities_all_M[:,threshold_num])
        if (0.01 <= p_values[threshold_num] < 0.05):
            pyplot.plot(thresholds[threshold_num],1-means_M[threshold_num]+stds_M[threshold_num]+.02,"k*", markersize='4')
        elif (p_values[threshold_num] < 0.01):
            pyplot.plot(thresholds[threshold_num],1-means_M[threshold_num]+stds_M[threshold_num]+.02,"k*", markersize='4')
            pyplot.plot(thresholds[threshold_num],1-means_M[threshold_num]+stds_M[threshold_num]+.04,"k*", markersize='4')
    # pyplot.xticks(np.arange(thresholds[3], thresholds[-2], 0.05))
    # pyplot.plot(thresholds[4:-1],p_values[4:-1])
    pyplot.xlabel("threshold")
    pyplot.ylabel("density ratio")
    pyplot.title("ANN densities vs. weight reject threshold")
    pyplot.show(block=True)

    draw_ANN = True
    if draw_ANN:
        vertical_distance_between_layers = 6
        horizontal_distance_between_neurons = 2
        neuron_radius = 0.5
        number_of_neurons_in_widest_layer = 28

        threshold = .75
        run_no = 1
        experiment_ID = "wo_Sensory_"+ANN_structures[0]+"_ANN_"+task_type+cur
        save_log_path = experiment_ID_base+"/"+experiment_ID#
        log_address="./log/{}/{}/".format(save_log_path,run_no)
        logdir = log_address+"compound/"
        sparsity = sparsity_calculation_fcn(logdir, threshold, draw_ANN=draw_ANN)
        print("sparsity: ", sparsity)

        experiment_ID = "wo_Sensory_"+ANN_structures[1]+"_ANN_"+task_type+cur
        save_log_path = experiment_ID_base+"/"+experiment_ID#
        log_address="./log/{}/{}/".format(save_log_path,run_no)
        leg = 0
        logdir = log_address+"leg_{}/".format(leg)
        sparsity = sparsity_calculation_fcn(logdir, threshold, draw_ANN=draw_ANN, num_joints = 2)
        print("sparsity: ", sparsity)

    
    # import pdb; pdb.set_trace()
