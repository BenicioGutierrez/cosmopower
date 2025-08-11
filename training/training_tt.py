
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display, clear_output

# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print('using', device, 'device \n')

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

# training parameters
training_parameters = np.load('./nn_params_retry1_mod.npz')  # CHANGE FILE
print(training_parameters.files)

print(training_parameters['omega_b'])
print('number of training samples: ', len(training_parameters['omega_b'])) # same for all of the other parameters

# training features (= log-spectra, in this case)
training_features = np.load('./TT_1_shifted.npz')
print(training_features.files)

print(training_features['modes'])
print('number of multipoles: ', len(training_features['modes']))

training_log_spectra = training_features['features']
print('(number of training samples, number of ell modes): ', training_log_spectra.shape)


#These premodifications are not necessary but might improve the accuracy
processing_vectors = {'mean':np.mean(training_log_spectra,axis=0), 'sigma':np.std(training_log_spectra,axis=0)}

def preprocessing(features,processing_vectors):
    return (features-processing_vectors['mean'])/processing_vectors['sigma']

def postprocessing_np(features,processing_vectors):
    return features*processing_vectors['sigma'] + processing_vectors['mean']

def postprocessing_tf(features,processing_vectors):
    return tf.add(tf.multiply(features, processing_vectors['sigma']), processing_vectors['mean'])

# NN Instantiation

# list of parameter names, in arbitrary order
model_parameters = ['h',
                    'tau',
                    'omega_b',
                    'n_s',
                    'ln10^{10}A_s',
                    'omega_cdm',
                    'log(omega_gdm)',
                    'w1',
                    'w2',
                    'w3']

ell_range = training_features['modes']
print('ell range: ', ell_range)

from cosmopower import cosmopower_NN

# instantiate NN class
cp_nn = cosmopower_NN(parameters=model_parameters,
                      modes=ell_range,
                      n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

print("Parameters in model:", cp_nn.parameters)
print("Number of input parameters:", cp_nn.n_parameters)
print("Architecture (input/hidden/output sizes):", cp_nn.architecture)

# Training code!!
with tf.device(device):
    # train
    cp_nn.train(training_parameters=training_parameters,
                training_features=training_log_spectra,
                filename_saved_model='NN_TT_log_1',
                #preprocessing = preprocessing,
                #postprocessing_np = postprocessing_np,
                #postprocessing_tf = postprocessing_tf,
                #processing_vectors = processing_vectors,
                # cooling schedule
                validation_split=0.2,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                batch_sizes=[1024, 1024, 1024, 500, 500, 500],
                gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,50,30,30,20],
                max_epochs = [1000,2000,1000,1000,1000,1000],
                )
