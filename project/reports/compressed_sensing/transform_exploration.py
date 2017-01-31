import utils
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
dataset_name = 'standard'
patch_size = (32, 32)  # (patch_height, patch_width)
compression_percent = 60

# Data acquisition
#   Generate original images
utils.generate_original_images(dataset_name)

#   Generate training set
utils.generate_train_images(dataset_name)

#   Generate test set
utils.generate_test_images(dataset_name)

# Pre-process data
#   Load train and test reference sets
data_paths = utils.get_data_paths(dataset_name)
train_image_list, train_name_list = utils.load_images(data_paths['train'])
test_image_list, test_name_list = utils.load_images(data_paths['test'])

#   Split in non-overlapping patches and vectorize
test_set_ref = utils.generate_vec_set(test_image_list, patch_size)
full_train_set_ref = utils.generate_vec_set(train_image_list, patch_size)
train_set_ref, val_set_ref \
    = utils.generate_cross_validation_sets(full_train_set_ref, fold_number=5, fold_combination=5)

# Transform exploration without applying the measurement model (i.e. the compression)
#   Parameters
transform_name = ['dirac', 'wavelet', 'wavelet', 'wavelet', 'wavelet', 'wavelet']
dec_type = [None, 'db1', 'db2', 'db4', 'db4', 'db4']
level = [0, 1, 2, 1, 2, 2]
mode = [None, 'symmetric', 'symmetric', 'symmetric', 'symmetric', 'periodization']

#   Generate the transform concatenation
transform_list = utils.generate_transform_list(patch_size, transform_name, dec_type, level, mode)

#   Choose a patch of the training set
i = 0
# patch_vec = full_train_set_ref[:, i] / 255
patch_vec = full_train_set_ref[:, i] / 255

#   Decomposition
dec_coeff_list, bk_mat_list = utils.multiple_transform_decomposition(patch_vec, transform_list)

#   Reconstruction
patch_vec_rec = utils.multiple_transform_reconstruction(dec_coeff_list, bk_mat_list, transform_list)

#   Reshape for plot
patch = utils.reshape_vec_in_patch(patch_vec, patch_size)
patch_rec = utils.reshape_vec_in_patch(patch_vec_rec, patch_size)
rel_diff = np.abs(patch - patch_rec) / np.linalg.norm(patch)

#   Figure 1: difference between the reference patch and the reconstruction
fig = plt.figure(num=1, figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
utils.plot_image_with_cbar(patch, ax=ax1, title='Reference patch {}'.format(patch_size))

ax2 = fig.add_subplot(1, 3, 2)
utils.plot_image_with_cbar(patch, ax=ax2, title='Reconstructed patch {}'.format(patch_size))

ax3 = fig.add_subplot(1, 3, 3)
utils.plot_image_with_cbar(rel_diff, ax=ax3, title='Relative difference')

#   Typical threshold that would be applied
theta = []  # MUST always be a list even if there is only one decomposition basis
threshold_fact = 0.02
for cv in dec_coeff_list:
    theta.append(threshold_fact*np.linalg.norm(cv, ord=np.inf))

#   Figure 2:
sub_plt_n_w = 3
sub_plt_n_h = int(np.ceil(len(dec_coeff_list) / sub_plt_n_w))
ax = []
cv_lim = [(cv.min(), cv.max()) for cv in dec_coeff_list]
ylim = (np.array(cv_lim).min(), np.array(cv_lim).max())
xlim = (0, max([tl['coeff_number'] for tl in transform_list]))
fig = plt.figure(num=2, figsize=(12, 8))
for i, (tl, cv) in enumerate(zip(transform_list, dec_coeff_list)):
    title = '{}\n {}, level={}, mode={}'.format(tl.get('name'), tl.get('wavelet_type'), tl.get('level'), tl.get('mode'))
    ax.append(fig.add_subplot(sub_plt_n_h, sub_plt_n_w, i + 1))
    # ax[i].set_title(title, fontsize=12)
    utils.plot_decomposition_coeffs(cv, ax=ax[i], title=title, theta=theta[i])
    ax[i].set_xlim(xlim)
    ax[i].set_ylim(ylim)

# Case 2: with compression
#   Mix and compress the patch
mm_type = 'gaussian-rip'  # or 'bernoulli-rip'
M = utils.create_measurement_model(mm_type, patch_size, compression_percent)
patch_comp = np.dot(M, patch_vec)

#   Decomposition
patch_vec = np.dot(M.transpose(), patch_comp)
dec_coeff_list, bk_mat_list = utils.multiple_transform_decomposition(patch_vec, transform_list)

#   Reconstruction
patch_vec_rec = utils.multiple_transform_reconstruction(dec_coeff_list, bk_mat_list, transform_list)

#   Reshape for plot
patch = utils.reshape_vec_in_patch(patch_vec, patch_size)
patch_rec = utils.reshape_vec_in_patch(patch_vec_rec, patch_size)
rel_diff = np.abs(patch - patch_rec) / np.linalg.norm(patch)

#   Figure 3: difference between the reference patch and the reconstruction
fig = plt.figure(num=3, figsize=(15, 5))
ax1 = fig.add_subplot(1, 3, 1)
utils.plot_image_with_cbar(patch, ax=ax1, title='Reference patch {}'.format(patch_size))

ax2 = fig.add_subplot(1, 3, 2)
utils.plot_image_with_cbar(patch, ax=ax2, title='Reconstructed patch {}'.format(patch_size))

ax3 = fig.add_subplot(1, 3, 3)
utils.plot_image_with_cbar(rel_diff, ax=ax3, title='Relative difference')

#   Typical threshold that would be applied
theta = []  # MUST always be a list even if there is only one decomposition basis
threshold_fact = 0.02
for cv in dec_coeff_list:
    theta.append(threshold_fact*np.linalg.norm(cv, ord=np.inf))

#   Figure 4:
sub_plt_n_w = 3
sub_plt_n_h = int(np.ceil(len(dec_coeff_list) / sub_plt_n_w))
ax = []
cv_lim = [(cv.min(), cv.max()) for cv in dec_coeff_list]
ylim = (np.array(cv_lim).min(), np.array(cv_lim).max())
xlim = (0, max([tl['coeff_number'] for tl in transform_list]))
fig = plt.figure(num=4, figsize=(12, 8))
for i, (tl, cv) in enumerate(zip(transform_list, dec_coeff_list)):
    title = '{}\n {}, level={}, mode={}'.format(tl.get('name'), tl.get('wavelet_type'), tl.get('level'), tl.get('mode'))
    ax.append(fig.add_subplot(sub_plt_n_h, sub_plt_n_w, i + 1))
    utils.plot_decomposition_coeffs(cv, ax=ax[i], title=title, theta=theta[i])
    ax[i].set_xlim(xlim)
    ax[i].set_ylim(ylim)

plt.show()

