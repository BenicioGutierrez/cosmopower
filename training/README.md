Code for generating data and training CMB power spectra emulators.

params_gen.py - generates Latin hypercube of parameters, and generates spectra output folder. 

params_rescale.py - changes omega_gdm to log scale

prep_spectra.py - convert output spectra folders to npz files

shift_spectra - convert Dl to Cl, remove l(l+1)/(2 * np.pi), for log spectra

shift_te - convert Dl to Cl for TE spectra

training_tt.py - train TT emulator

training_ee.py - train EE emulator

training_te.py - train TE emulator
