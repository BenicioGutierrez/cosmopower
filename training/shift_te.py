import numpy as np

# Load original data
data = np.load('TE_1.npz')
features = data['features']  # shape: (n_samples, ell)
ell = data['modes']          # shape: (ell,)

ell = ell.reshape(1, -1)

# Apply the transformation
transformed = (features) * (2 * np.pi) / (ell * (ell + 1))


# Save new .npz
np.savez('TE_1_shifted.npz', features=transformed, modes=ell.flatten())
