import tensorflow as tf

experiment_ID = "test_run_2"
MuJoCo_model_name = "tendon_quadruped_ws_inair.xml"
run_no = 0

log_address="./log/{}/{}/".format(experiment_ID,run_no)
logdir = log_address+"compound/"
Inverse_ANN_models = tf.keras.models.load_model(logdir+"model",compile=False)
weights_all = Inverse_ANN_models.get_weights()

weights_all[0]
import pdb; pdb.set_trace()