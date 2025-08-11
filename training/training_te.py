import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display, clear_output

# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print('using', device, 'device \n')

training_parameters = np.load('./nn_params_retry1_mod.npz')


# Tests
print(training_parameters.files)

print(training_parameters['omega_b'])
print('number of training samples: ', len(training_parameters['omega_b'])) # same for all of the other parameters

# training features (= log-spectra, in this case)
training_features = np.load('./TE_1_shifted.npz', mmap_mode='r')
print(training_features.files)


print(training_features['modes'])
print('number of multipoles: ', len(training_features['modes']))

training_log_spectra = training_features['features']
#print('(number of training samples, number of ell modes): ', training_log_spectra.shape)

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


#These premodifications are not necessary but might improve the accuracy
processing_vectors = {'mean':np.mean(training_log_spectra,axis=0), 'sigma':np.std(training_log_spectra,axis=0)}

def preprocessing(features,processing_vectors):
    return (features-processing_vectors['mean'])/processing_vectors['sigma']

def postprocessing_np(features,processing_vectors):
    return features*processing_vectors['sigma'] + processing_vectors['mean']

def postprocessing_tf(features,processing_vectors):
    return tf.add(tf.multiply(features, processing_vectors['sigma']), processing_vectors['mean'])

from cosmopower import cosmopower_PCA
n_pcas = 512


cp_pca = cosmopower_PCA(parameters=model_parameters,
                        modes=ell_range,
                        n_pcas=n_pcas,
                        parameters_filenames=["./nn_params_retry1_mod"],
                        features_filenames=["./TE_1_shifted"],
                        verbose=True, # useful to follow the various steps
                        )


cp_pca.transform_and_stack_training_data()

from cosmopower import cosmopower_PCAplusNN

cp_pca_nn = cosmopower_PCAplusNN(cp_pca=cp_pca,
                                 n_hidden=[512,512,512,512,512], # 4 hidden layers, each with 512 nodes
                                 verbose=True, # useful to understand the different steps in initialisation and training
                                 )

with tf.device(device):
    # train
    cp_pca_nn.train(filename_saved_model='NN_TE_1',
                    # cooling schedule
                    validation_split=0.2,
                    learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                    batch_sizes=[1024, 1024, 1024, 500, 100, 100],
                    gradient_accumulation_steps = [1, 1, 1, 1, 1, 1],
                    # early stopping set up
                    patience_values = [100, 100, 100, 50, 20, 20],
                    max_epochs = [1000,1000,1000,1000,1000,1000],
                    )
