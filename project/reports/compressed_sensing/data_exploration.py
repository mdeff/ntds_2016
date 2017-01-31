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

#   Mix and compress train and test sets
mm_type = 'gaussian-rip'  # or 'bernoulli-rip'
M = utils.create_measurement_model(mm_type, patch_size, compression_percent)

train_set = np.matmul(M, train_set_ref)
val_set = np.matmul(M, val_set_ref)
test_set = np.matmul(M, test_set_ref)

# Data exploration
#   Figure 1: display training set images
fig = plt.figure(1, figsize=(10, 7))
fig.suptitle('Dataset: {}'.format(dataset_name), fontsize=12)
utils.plot_image_set(train_image_list, train_name_list, fig=fig)

#   Figure 2: display test set images
fig = plt.figure(2, figsize=(10, 7))
fig.suptitle('Dataset: {}'.format(dataset_name), fontsize=12)
utils.plot_image_set(test_image_list, test_name_list, fig=fig)

#   Figure 3: detail first image of the reference train set
im = train_image_list[0]
im_name = train_name_list[0]

fig = plt.figure(num=3, figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(im, cmap='gray', vmin=im.min(), vmax=im.max())
ax.set_axis_off()
ax.set_title('{}\n({}, {})'.format(im_name, im.shape[0], im.shape[1]), fontsize=12)

#   Figure 4: shows some patches of the first image of the reference train set
#       Select randomly some patches of the selected image
im_patch_number = np.floor(np.asarray(im.shape) / np.asarray(patch_size)).astype(int)
num_pw = 5  # patch number in width dim
num_ph = 2  # patch number in height dim
patch_idx = sorted(np.random.choice(im_patch_number.prod() - 1, num_pw * num_ph))

#       Recover patches from the reference train set
rec_patches = utils.reshape_vec_in_patch(full_train_set_ref, patch_size)

fig = plt.figure(num=4, figsize=(10, 5))
fig.suptitle('{} -- patch size = ({}, {}) -- patch number = {}'.format(im_name, patch_size[0], patch_size[1],
                                                                       im_patch_number.prod()), fontsize=12)
ax = []
for i, idx in enumerate(patch_idx):
    ax.append(fig.add_subplot(num_ph, num_pw, i+1))
    ax[i].imshow(rec_patches[:, :, idx], cmap='gray', vmin=im.min(), vmax=im.max())
    ax[i].set_axis_off()
    ax[i].set_title('#{}'.format(idx + 1), fontsize=12)

#   Figure 5: shows the reference train set and the compressed train set
fig = plt.figure(num=5, figsize=(10, 3))
fig.suptitle('Compression = {}%'.format(compression_percent), fontsize=12)
#   Full reference set
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(test_set_ref, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference set:\n({}, {})'.format(test_set_ref.shape[0], test_set_ref.shape[1]), fontsize=12)
#   Compressed set
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(test_set, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Compressed set:\n({}, {})'.format(test_set.shape[0], test_set.shape[1]), fontsize=12)
ax2.set_ylim(ax1.get_ylim())

plt.show()