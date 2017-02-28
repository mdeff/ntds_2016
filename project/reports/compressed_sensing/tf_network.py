import numpy as np
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

#   Generate test set
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
train_set /= 255.0
val_set = np.matmul(M, val_set_ref)
val_set /= 255.0
test_set = np.matmul(M, test_set_ref)  # in this case identical to np.dot()
test_set /= 255.0


# Network configuration
# TODO: configuration of the network
algorithm = 'ISTA'  # only ISTA available yet
transform_name = ['wavelet', 'wavelet', 'wavelet', 'wavelet']
wavelet_type = ['db1', 'db2', 'db4', 'db4']
level = [1, 2, 2, 1]
mode = ['symmetric', 'symmetric', 'symmetric', 'periodization']

transform_list = utils.generate_transform_list(patch_size, transform_name, wavelet_type, level, mode)

# Parameters set up (ISTA)
lmdb = 1e-2
L, _ = scipy.sparse.linalg.eigsh(np.dot(M.transpose(), M), k=1, which='LM')
#   Filter matrix
We = 1/L*M.transpose()
#   Mutual inhibition matrix
S = np.eye(M.shape[1]) - 1/L*np.dot(M.transpose(), M)
# TODO: bornes exactes pour theta?
theta = lmdb/L

#   Convert transform list in tensorflow
tf_transform_list = utils.convert_transform_list_to_tf(transform_list)

# Placeholders
#   Compressed patches
tf_patch_vec_train = tf.placeholder(dtype=tf.float32, shape=(train_set.shape[0], 1))
tf_patch_vec_test = tf.placeholder(dtype=tf.float32, shape=val_set.shape[0])
#   Measurement model
tf_M = tf.constant(M, dtype=tf.float32)
#   Thresholds
tf_threshold = []
for transform in transform_list:
    tf_threshold.append(tf.Variable(tf.ones(transform['coeff_number'], dtype=tf.float32)))
#   ISTA
tf_L = tf.constant(L, dtype=tf.float32)
tf_We = tf.constant(We, dtype=tf.float32)
tf_S  = tf.constant(S, dtype=tf.float32)

# TODO: multiple layers...

# Wavelet decomposition
# patch_vec = np.dot(M.transpose(), patch_comp)
# dec_coeff_list, bk_mat_list = utils.multiple_transform_decomposition(patch_vec, transform_list)
tf_patch_vec_train_back = tf.matmul(tf.transpose(tf_M), tf_patch_vec_train)
tf_decomposition_coeff, tf_bookkeeping_mat = \
    utils.tf_multiple_transform_decomposition(tf_patch_vec_train_back, tf_transform_list)

# Threshold applies here on each transform
tf_decomposition_coeff_th = []
for i in range(len(transform_list)):
    tf_decomposition_coeff_th.append(utils.tf_soft_thresholding(tf_decomposition_coeff[i], tf_threshold[i]))

# Reconstruction
tf_patch_vec_rec = utils.tf_multiple_transform_reconstruction(tf_decomposition_coeff, tf_bookkeeping_mat,
                                                              tf_transform_list)

init = tf.initialize_all_variables()
with tf.Session() as sess:

    sess.run(init)

    tf_patch_vec_rec_val = sess.run(tf_patch_vec_rec, feed_dict={tf_patch_vec_train: train_set[:, :1]})

    # print('count non zero decomposition:')
    # print(np.count_nonzero(cv_tf_val - cv.astype(np.float32)))
    # print(np.count_nonzero(bk_tf_val - bk.astype(np.int32)))
    # print('count non zero reconstruction:')
    # print(np.count_nonzero(patch_vec_rec_tf_val - patch_vec_rec.astype(np.float32)))
    # print('multiple decomposition:')

print('***DONE***')


# CONTROL VALUES
# Wavelet parameters
patch_vec = train_set_ref[:, 0].astype(np.float32)
# patch_vec = np.dot(M.transpose(), train_set[:,0])
# cv, bk = utils.wavelet_decomposition(patch_vec, transform_dict)
# patch_vec_rec = utils.wavelet_reconstruction(cv, bk, transform_dict)
# dec_coeff, bk_mat = utils.multiple_transform_decomposition(patch_vec, transform_list)
decomposition_coeff, bookkeeping_mat = utils.multiple_transform_decomposition(patch_vec, transform_list)
patch_vec_rec = utils.multiple_transform_reconstruction(decomposition_coeff, bookkeeping_mat, transform_list)
