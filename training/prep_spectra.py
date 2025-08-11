import numpy as np
import glob
import os
import re

# Define the folder containing the spectra files
spectra_folder = "class_spectra_test"
file_pattern = os.path.join(spectra_folder, "params_set_*_cl.dat")

def extract_index(filename):
    match = re.search(r'params_set_(\d+)_cl\.dat', filename)
    return int(match.group(1)) if match else -1

# Numerically sort files
spectra_files = sorted(glob.glob(file_pattern), key=extract_index)

# Initialize
tt_list, ee_list, te_list = [], [], []

for file in spectra_files:
    data = np.loadtxt(file)

    if data.shape[0] < 2:
        print(f"Skipping {file}, insufficient data.")
        continue

    ell = data[:, 0]

    Cl_tt = np.log10(np.abs(data[:, 1]))
    Cl_ee = np.log10(np.abs(data[:, 2]))
    Cl_te = data[:, 3]

    # Handle inf/nan
    Cl_tt[np.isinf(Cl_tt) | np.isnan(Cl_tt)] = -999
    Cl_ee[np.isinf(Cl_ee) | np.isnan(Cl_ee)] = -999

    tt_list.append(Cl_tt)
    ee_list.append(Cl_ee)
    te_list.append(Cl_te)

# Convert to arrays
tt_array = np.array(tt_list)
ee_array = np.array(ee_list)
te_array = np.array(te_list)

# Save
np.savez('TT_test.npz', modes=ell, features=tt_array)
np.savez('EE_test.npz', modes=ell, features=ee_array)
np.savez('TE_test.npz', modes=ell, features=te_array)
