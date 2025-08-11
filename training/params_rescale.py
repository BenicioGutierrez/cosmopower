import numpy as np

# Load the original .npz file
data = np.load('nn_params_test.npz')

# Convert to a dictionary (this keeps all arrays)
new_data = dict(data)

# Extract and transform 'omega_gdm'
omega_gdm_scaled = np.log10(new_data['omega_gdm'])

# Remove the original 'omega_gdm'
del new_data['omega_gdm']

# Add the new scaled version
new_data['log(omega_gdm)'] = omega_gdm_scaled

# Save to a new file (or overwrite the original if you prefer)
np.savez('nn_params_test_mod.npz', **new_data)
