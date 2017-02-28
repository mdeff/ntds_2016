import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import utils
import scipy.sparse
import tensorflow as tf

# Global parameters
dataset_name = 'standard'
patch_size = (32, 32)  # (patch_height, patch_width)
compression_percent = 60

# Data acquisition
#   Generate original images
utils.generate_original_images(dataset_name)

#   Generate training set
train_name_list = ['airplane']  # optional, default is in generate_train_images()
utils.generate_train_images(dataset_name, train_name_list)
# utils.generate_train_images(dataset_name)

#   Generate test set
# test_name_list = ['fruits', 'frymire']
# utils.generate_test_images(dataset_name, test_name_list)
utils.generate_test_images(dataset_name)

# Pre-process data
#   Load train and test sets
data_paths = utils.get_data_paths(dataset_name)
train_image_list, train_name_list = utils.load_images(data_paths['train'], file_ext='.png')
test_image_list, test_name_list = utils.load_images(data_paths['test'], file_ext='.png')

#   Split in non-overlapping patches and vectorize
test_set_ref = utils.generate_vec_set(test_image_list, patch_size)

full_train_set_ref = utils.generate_vec_set(train_image_list, patch_size)
train_set_ref, val_set_ref \
    = utils.generate_cross_validation_sets(full_train_set_ref, fold_number=5, fold_combination=5)

#   Mix and compress train and test sets
mm_type = 'gaussian-rip'  # or 'bernoulli-rip'
M = utils.create_measurement_model(mm_type, patch_size, compression_percent)

train_set = np.matmul(M, train_set_ref)
val_set = np.matmul(M, val_set_ref)
test_set = np.matmul(M, test_set_ref)  # in this case identical to np.dot()

# Wavelet parameters
patch_vec = train_set_ref[:, 0]
patch_vec = np.dot(M.transpose(), train_set[:,0])
patch_mat = utils.reshape_vec_in_patch(patch_vec, patch_size)

# TODO: configuration of the network
algorithm = 'ISTA'  # -> to create the network
transform_name = ['wavelet', 'wavelet', 'wavelet', 'wavelet']
wavelet_type = ['db1', 'db2', 'db4', 'db4']
level = [1, 2, 2, 1]
mode = ['symmetric', 'symmetric', 'symmetric', 'periodization']

transform_list = utils.generate_transform_list(patch_size, transform_name, wavelet_type, level, mode)

# Parameters set up (ISTA)
lmdb = 1e-2
#   L >= 2 x max eigevalue of M
L, _ = scipy.sparse.linalg.eigsh(np.dot(M.transpose(), M), k=1, which='LM')
#   Filter matrix
We = 1/L*M.transpose()
#   Mutual inhibition matrix
S = np.eye(M.shape[1]) - 1/L*np.dot(M.transpose(), M)
# TODO: bornes pour theta?
theta = lmdb/L
# theta = lmdb/(2*L)?

# Wavelet decomposition
coeffs_vec = []
bk_mat = []
scale_factor = np.sqrt(len(transform_list))
# scale factor can be applied either to the patch_vec or to the coeffs (i.e. cv)
for transform in transform_list:
    cv, bk = utils.wavelet_decomposition(patch_vec/scale_factor, transform)
    coeffs_vec.append(cv)
    bk_mat.append(bk)

# THRESHOLDING
#   Theoretical threshold
theta = []  # MUST always be a list even if there is only one decomposition basis
threshold_fact = 0.4
for cv in coeffs_vec:
    #theta.append(np.linspace(0, 1, cv.shape[0])*threshold_fact*np.linalg.norm(cv, ord=np.inf))
    theta.append(threshold_fact*np.linalg.norm(cv, ord=np.inf))

#   Plot coeffs and thresholds
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
for i, (cv, transform, th) in enumerate(zip(coeffs_vec, transform_list, theta)):
    utils.plot_decomposition_coeffs(cv, th,
                                    label='{}, level={}, mode={}'.format(transform['wavelet_type'],
                                                                             transform['level'],
                                                                             transform['mode']))

ax.legend()
# plt.show()

# Wavelet reconstruction
patch_vec_rec = np.zeros((np.prod(patch_size)))
scale_factor = np.sqrt(len(transform_list))
for i, (cv, bk, tl) in enumerate(zip(coeffs_vec, bk_mat, transform_list)):
    patch_vec_rec += utils.wavelet_reconstruction(cv, bk, tl)/scale_factor

patch_mat_rec = utils.reshape_vec_in_patch(patch_vec_rec, patch_size)

fig = plt.figure(num=2, figsize=(15, 5))
fig.suptitle('Wavelet dec + rec check', fontsize=18)
#   Patch before wavelet decomposition
ax1 = fig.add_subplot(1, 3, 1)
p1 = ax1.imshow(patch_mat, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference patch', fontsize=12)
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(p1, cax=cax)

#   Patch after wavelet decompostion and reconstruction
ax2 = fig.add_subplot(1, 3, 2)
p2 = ax2.imshow(patch_mat_rec, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Reconstructed patch', fontsize=12)
ax2.set_ylim(ax1.get_ylim())
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(p2, cax=cax)

#   Diff
ax3 = fig.add_subplot(1, 3, 3)
p3 = ax3.imshow(np.abs(patch_mat - patch_mat_rec), cmap='gray')
ax3.set_axis_off()
ax3.set_title('Absolute difference', fontsize=12)
ax3.set_ylim(ax1.get_ylim())
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(p3, cax=cax)

plt.show()

print(np.linalg.norm(patch_mat_rec))
print(np.linalg.norm(patch_mat))
print('done')

# print(np.linalg.norm(M, axis=0))
#   Normalize by the norm of the columns
# normc_M =  np.linalg.norm(M, axis=0)



