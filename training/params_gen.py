#!/usr/bin/env python3
import os
import stat
import shutil
import numpy as np
import subprocess
import pyDOE
import glob


# =======================
# User-specified settings
# =======================

# CLASS paths (update these to your actual CLASS installation)
class_exec = "/users/benicio/class_public/class"  # actual CLASS executable
class_dir  = "/users/benicio/class_public"          # CLASS working directory

# Output directory for .ini files and CLASS outputs
output_dir = "./class_spectra_retry1"
os.makedirs(output_dir, exist_ok=True)
# Set permissions so that the folder is writable by you (user and group)
os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG)


# =====================
# Helper functions
# =====================

def run_class(ini_file):
    abs_ini_file = os.path.abspath(ini_file)
    try:
        # Run the CLASS program
        subprocess.run([class_exec, abs_ini_file], cwd=class_dir, check=True)
        return True, None  # Success, no error message
    except subprocess.CalledProcessError as e:
        # Capture the error message from the exception
        error_message = str(e)
        print(f"CLASS execution failed for {abs_ini_file}: {error_message}")
        return False, error_message  # Failure, return the error message

def generate_ini_file(params, i, output_dir):
    """
    Write an .ini file for CLASS using the parameter dictionary.
    Returns the full path of the generated file.
    """
    ini_content = f"""
# Output and precision settings
output = tCl,pCl,lCl

modes = s
ic = ad
gauge = synchronous

lensing = yes
lcmb_rescale = 1
lcmb_tilt = 0
lcmb_pivot = 0.1

non_linear =
P_k_max_h/Mpc = 1.

# Cosmological parameters
omega_b = {params['omega_b']}
omega_cdm = {params['omega_cdm']}
h = {params['h']}

YHe = BBN
recombination = HyRec
reio_parametrization = reio_camb
n_s = {params['n_s']}
ln_A_s_1e10 = {params['ln10^{10}A_s']}
tau_reio = {params['tau']}
N_ncdm = 0
N_eff = 3.046
"""

    # Include GDM-specific parameters only if has_gdm is True
    if (1==1):
        ini_content += f"""
# GDM-specific parameters
gdm_log10a_vals = -14.0, -13.50, -13.0, -12.0, -11.0, -10.0, -9.0, -8.5, -8.25, -8.0, -7.5, -7.0, -6.5, -6.25, -6.0, -5.75, -5.5, -5.25, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.75, -1.5, -0.75, -0.5, 0
"""
        # For w, we want: w_ini, then w1,...,w10, then w_fin
        w_list = [0.33333] + [0.33333] +  [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] + [0.33333] $        w_str = ", ".join(map(str, w_list))
        ini_content += f"gdm_w_vals = {w_str}\n"
        ini_content += f"Omega_gdm = {params['omega_gdm']}\n"
        ini_content += "gdm_w_interpolation_method = 3\n"
        ini_content += "nap = n\n"

    # Additional CLASS output settings (optional)
    ini_content += f"""
#k_output_values = 0.01, 0.1, 0.0001

# 1.a.1) If root is specified, do you want to keep overwriting the file,
#      or do you want to create files numbered as '<root>N_'.
#      Can be set to anything starting with 'y' or 'n' (default: no)
overwrite_root = y

# 1.b) Do you want headers at the beginning of each output file (giving
#      precisions on the output units/ format) ? Can be set to anything
#      starting with 'y' or 'n' (default: yes)
headers = yes

# 1.c) In all output files, do you want columns to be normalized and ordered
#      with the default CLASS definitions or with the CAMB definitions (often
#      idential to the CMBFAST one) ? Set 'format' to either 'class', 'CLASS',
#      'camb' or 'CAMB' (default: 'class')
format = class

# 1.d) Do you want to write a table of background quantitites in a file? This
#      will include H, densities, Omegas, various cosmological distances, sound
#      horizon, etc., as a function of conformal time, proper time, scale
#      factor. Can be set to anything starting with 'y' or 'no' (default: no)
#write_background = yes

# 1.e) Do you want to write a table of thermodynamics quantitites in a file?
#      Can be set to anything starting with 'y' or 'n'. (default: no)
write_thermodynamics = no

# 1.f) Do you want to write a table of perturbations to files for certain
#      wavenumbers k? Dimension of k is 1/Mpc. The actual wave numbers are
#      chosen such that they are as close as possible to the requested k-values. (default: none)
#k_output_values = 0.01, 0.1, 0.0001

# 1.g) Do you want to write the primordial scalar(/tensor) spectrum in a file,
#      with columns k [1/Mpc], P_s(k) [dimensionless], ( P_t(k)
#      [dimensionless])? Can be set to anything starting with 'y' or 'n'. (default: no)
write_primordial = no

# 1.h) Do you want to write the exotic energy injection function in a file,
#     with columns z [dimensionless], dE/dz_inj, dE/dz_dep [J/(m^3 s)]?
# 1.i) Do you want to write also the non-injected photon heating?
#     File created if 'write_exotic_injection' or
#     'write_noninjection' set to something containing the letter
#     'y' or 'Y', file written, otherwise not written (default: no)
write_exotic_injection = no
#write_noninjection = no

# 1.k) Do you want to write the spectral distortions in a file,
#     with columns x [dimensionless], DI(x) [dimensionless]?
#     File created if 'write_distortions' set to something containing the letter
#     'y' or 'Y', file written, otherwise not written (default: no)
write_distortions = no

# 1.l) Do you want to have all input/precision parameters which have been read
#      written in file '<root>parameters.ini', and those not written in file
#      '<root>unused_parameters' ? Can be set to anything starting with 'y'
#      or 'n'. (default: yes)
write_parameters = no

# 1.m) Do you want a warning written in the standard output when an input
#      parameter or value could not be interpreted ? Can be set to anything starting
#      with 'y' or 'n' (default: no)
#write_warnings = yes

# 2) Amount of information sent to standard output: Increase integer values
#    to make each module more talkative (default: all set to 0)
input_verbose = 1
background_verbose = 1
thermodynamics_verbose = 1
perturbations_verbose = 1
transfer_verbose = 1
primordial_verbose = 1
harmonic_verbose = 1
fourier_verbose = 1
lensing_verbose = 1
distortions_verbose = 1
output_verbose = 1

"""


    # Specify output directory for CLASS files
    ini_content += f"root = {os.path.abspath(output_dir)}/params_set_{i}\n"

    ini_path = os.path.join(output_dir, f"params_set_{i}.ini")
    try:
        with open(ini_path, "w") as f:
            f.write(ini_content)
        os.chmod(ini_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except PermissionError:
        print(f"Permission denied when writing {ini_path}")

    return ini_path


# =====================
# Main loop: Run CLASS for each combination from AllCombinations
# =====================np.random.seed(42)  # For reproducibility

n_params_lhs = 10

n_samples_orig = 150000

n_params_lhs = 10

n_samples_orig = 150000

# Latin Hypercube Sampling for selected parameters
lhd = pyDOE.lhs(n_params_lhs, samples=n_samples_orig, criterion=None)  # Only for first 10 params
idx = (lhd * n_samples_orig).astype(int)

# Define parameter ranges for LHS sampling
obh2       = np.linspace(0.01861, 0.02611, n_samples_orig)[idx[:, 0]]
omch2      = np.linspace(0.0852, 0.1552, n_samples_orig)[idx[:, 1]]
h          = np.linspace(0.5227, 0.8227, n_samples_orig)[idx[:, 2]]
ns         = np.linspace(0.80, 1.5, n_samples_orig)[idx[:, 3]]    #
ln10_10A_s = np.linspace(2.60, 3.4, n_samples_orig)[idx[:, 4]]
tau_reio   = np.linspace(0.0053, 0.1226, n_samples_orig)[idx[:, 5]]

omega_gdm = np.logspace(-15, -1, n_samples_orig)[idx[:, 6]]

w1         = np.linspace(-0.6, 0.8, n_samples_orig)[idx[:, 7]]
w2         = np.linspace(-0.6, 0.8, n_samples_orig)[idx[:, 8]]
w3         = np.linspace(-0.6, 0.8, n_samples_orig)[idx[:, 9]]


w_ini      = 0.33333
w_fin = w3  # Ensuring w_fin is identical to w3

# Stack all parameters into an array
AllCombinations = np.vstack([
    obh2, omch2, h, ns, ln10_10A_s, tau_reio,
    omega_gdm, w1, w2, w3
]).T  # Transpose to get correct shape

# Combine all parameters into one array
AllParams = np.vstack([
    obh2, omch2, h, ns, ln10_10A_s, tau_reio,
    omega_gdm, w1, w2, w3
])

successful_params = []      # Store successful parameter sets
successful_ini_files = []   # Store successful .ini file paths

param_names = ['omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s', 'tau',
               'omega_gdm', 'w1', 'w2', 'w3']


spectra_folder = "class_spectra_retry1"                                 ###########################
file_pattern = os.path.join(spectra_folder, "params_set_*_cl.dat")      ###########################
spectra_list = []                                                       ##################

import os

# (Assume earlier definitions of n_samples_orig, param_names, etc. remain unchanged)
# CLASS paths (update these to your actual CLASS installation)
class_exec = "/users/benicio/class_public/class"  # actual CLASS executable
class_dir  = "/users/benicio/class_public"          # CLASS working directory

# Output directory for .ini files and CLASS outputs
output_dir = "./class_spectra_retry1"
os.makedirs(output_dir, exist_ok=True)
os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG)

spectra_folder = output_dir  # assuming spectra files are in the same output folder


successful_params = []      # Store successful parameter sets
successful_ini_files = []   # Store successful .ini file paths

for i in range(n_samples_orig):  # Loop over all Latin Hypercube samples
    num_attempt = 1

    print(f"\n=== Testing combination {i} ===")
    while True:
        # Extract parameters from AllCombinations
        if num_attempt == 1:
            params = dict(zip(param_names, AllCombinations[i]))
            print("Attempt num ", i)
            print(params['omega_gdm'])

        ini_file = generate_ini_file(params, i, output_dir)
        # Try running CLASS
        success, error_message = run_class(ini_file)
        if success:
            print(f"Combination {i} succeeded with ini file {ini_file}")
            successful_params.append(params)
            successful_ini_files.append(ini_file)

            break  # move on to the next combination
        else:
            print(f"Combination {i} failed, generating a new set of parameters and retrying.")


            #if "=>thermodynamics_init(L:342)" in error_message:
            #    print("Error in thermodynamics_init detected. Adjusting omega_gdm.")
            #    params['omega_gdm'] = np.random.uniform(0, 0.01*params['omega_gdm'])  # Modify omega_gdm randomly

            if os.path.exists(ini_file):
                os.remove(ini_file)
            break

# ----- End of loop over n_samples_orig -----

# Compile parameters from all successful runs into params_compiled
params_compiled = {
    'omega_b': np.array([param['omega_b'] for param in successful_params]),
    'omega_cdm': np.array([param['omega_cdm'] for param in successful_params]),
    'h': np.array([param['h'] for param in successful_params]),
    'n_s': np.array([param['n_s'] for param in successful_params]),
    'ln10^{10}A_s': np.array([param['ln10^{10}A_s'] for param in successful_params]),
    'tau': np.array([param['tau'] for param in successful_params]),
    'omega_gdm': np.array([param['omega_gdm'] for param in successful_params]),
    'w1': np.array([param['w1'] for param in successful_params]),
    'w2': np.array([param['w2'] for param in successful_params]),
    'w3': np.array([param['w3'] for param in successful_params]),
}

# ===== NEW: Append the new parameters to nn_parameters_final.npz =====
params_file = "nn_params_retry1.npz"
if os.path.exists(params_file):
    loaded_params = np.load(params_file)
    updated_params = {}
    for key in params_compiled:
        # If the key already exists, concatenate the old and new arrays.
        if key in loaded_params:
            updated_params[key] = np.concatenate([loaded_params[key], params_compiled[key]])
        else:
            updated_params[key] = params_compiled[key]
else:
    updated_params = params_compiled

np.savez(params_file, **updated_params)
print("Parameters appended to nn_params_retry1.npz")
# ===== END NEW parameters saving =====

