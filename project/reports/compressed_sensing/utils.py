import scipy.misc
import os
import shutil
import zipfile
import numpy as np
from itertools import product
import pywt
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import warnings
import tensorflow as tf

def is_array_str(obj):
    """
    Check if obj is a list of strings or a tuple of strings or a set of strings
    :param obj: an object
    :return: flag: True or False
    """
    # TODO: modify the use of is_array_str(obj) in the code to is_array_of(obj, classinfo)
    flag = False
    if isinstance(obj, str):
        pass
    elif all(isinstance(item, str) for item in obj):
        flag = True

    return flag


def is_array_of(obj, classinfo):
    """
    Check if obj is a list of classinfo or a tuple of classinfo or a set of classinfo
    :param obj: an object
    :param classinfo: type of class (or subclass). See isinstance() build in function for more info
    :return: flag: True or False
    """
    flag = False
    if isinstance(obj, classinfo):
        pass
    elif all(isinstance(item, classinfo) for item in obj):
        flag = True

    return flag


def check_and_convert_to_list_str(obj):
    """
    Check if obj is a string or an array like of strings and return a list of strings
    :param obj: and object
    :return: list_str: a list of strings
    """
    if isinstance(obj, str):
        list_str = [obj]  # put in a list to avoid iterating on characters
    elif is_array_str(obj):
        list_str = []
        for item in obj:
            list_str.append(item)
    else:
        raise TypeError('Input must be a string or an array like of strings.')

    return list_str


def load_images(path, file_ext='.png'):
    """
    Load images in grayscale from the path
    :param path: path to folder
    :param file_ext: a string or a list of strings (even an array like of strings)
    :return: image_list, image_name_list
    """
    # Check file_ext type
    file_ext = check_and_convert_to_list_str(file_ext)

    image_list = []
    image_name_list = []
    for file in os.listdir(path):
        file_name, ext = os.path.splitext(file)
        if ext.lower() not in file_ext:
            continue
        # Import image and convert it to 8-bit pixels, black and white (using mode='L')
        image_list.append(scipy.misc.imread(os.path.join(path, file), mode='L'))
        image_name_list.append(file_name)

    return image_list, image_name_list


def extract_zip_archive(zip_file_path, extract_path, file_ext=''):
    """
    Extract zip archive. If file_ext is specified, only extracts files with specified extension
    :param zip_file_path: path to zip archive
    :param extract_path: path to export folder
    :param file_ext: a string or a list of strings (even an array like of strings)
    :return:
    """
    # Check file_ext type
    file_ext = check_and_convert_to_list_str(file_ext)

    # Check if export_path already contains the files with a valid extension
    valid_files_in_extract_path = [os.path.join(root, name)
                                   for root, dirs, files in os.walk(extract_path)
                                   for name in files
                                   if name.endswith(tuple(file_ext))]

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        files_to_extract = [name for name in zip_ref.namelist()
                            if name.endswith(tuple(file_ext))
                            and os.path.join(extract_path, name) not in valid_files_in_extract_path]

        # Only extracts files if not already extracted
        # TODO: load directly the images without extracting them
        for file in files_to_extract:
            print(file)
            zip_ref.extract(file, path=extract_path)

    return


def export_image_list(image_list, image_name_list, export_name_list=True, path='', file_ext='.png'):
    """
    Export images export_name_list of (image_list, image_name_list) in path as image_name.ext
    :param image_list: list of array
    :param image_name_list: list of strings
    :param export_name_list: True, False, None, string or list of strings
    :param path: path to export folder
    :param file_ext: file extension
    :return:
    """
    # Check if file_ext is a string
    if not isinstance(file_ext, str):
        raise TypeError('File extension must be a string')

    # Check if image_name_list is list of string or simple string
    image_name_list = check_and_convert_to_list_str(image_name_list)

    # Check export_name_list type
    #   if is True, i.e. will export all images
    #   Otherwise check if export_name_list is list of strings or simple string
    if isinstance(export_name_list, bool):
        if export_name_list:  # True case
            export_name_list = image_name_list
        else:  # False case
            export_name_list = ['']
    elif export_name_list is None:
        export_name_list = ['']
    else:
        export_name_list = check_and_convert_to_list_str(export_name_list)

    # Check if folder already exists
    if os.path.exists(path):  # never True if path = ''
        # Check if folder content is exactly the same as what will be exported
        if not sorted(os.listdir(path)) == [item + file_ext for item in sorted(export_name_list)]:
            shutil.rmtree(path)
            print('Folder {} has been removed'.format(path))
        else:
            return

    # Check if folder doesn't exist and if path not empty to create the folder
    if not os.path.exists(path) and path:
        os.makedirs(path)
        print('Folder {} has been created'.format(path))

    # Save images
    for i, image_name in enumerate(image_name_list):
        if image_name not in export_name_list:
            continue
        scipy.misc.imsave(os.path.join(path, image_name + file_ext), image_list[i])
        print('Saved {} {} as {}'.format(image_name, image_list[i].shape, os.path.join(path, image_name + file_ext)))

    return


def get_data_paths(dataset_name):
    """
    Generate and return data paths
    :param dataset_name: string
    :return: data_paths: dict
    """
    if not isinstance(dataset_name, str):
        raise TypeError('Data set name must be a string')

    keys = ['sources_base', 'source', 'source_archive', 'dataset', 'orig', 'train', 'test']
    data_paths = dict.fromkeys(keys)

    data_paths['sources_base'] = os.path.join('datasets', 'sources')
    data_paths['source'] = os.path.join(data_paths['sources_base'], dataset_name)
    data_paths['source_archive'] = data_paths['source'] + '.zip'
    data_paths['dataset'] = os.path.join('datasets', dataset_name)
    data_paths['orig'] = os.path.join(data_paths['dataset'], 'orig')
    data_paths['train'] = os.path.join(data_paths['dataset'], 'train')
    data_paths['test'] = os.path.join(data_paths['dataset'], 'test')

    return data_paths


def generate_original_images(dataset_name):
    """
    Generate original images
    :param dataset_name: name of the dataset such that dataset.zip exists
    :return:
    """
    # TODO: download from the web so it doesn't have to be hosted on github

    # Parameters
    data_sources_main_path = os.path.join('datasets', 'sources')
    data_source_path = os.path.join(data_sources_main_path, dataset_name)
    data_source_zip_path = data_source_path + '.zip'
    valid_ext = ['.jpg', '.tif', '.tiff', '.png', '.bmp']
    export_path = os.path.join('datasets', dataset_name, 'orig')

    # Unzip archive
    extract_zip_archive(data_source_zip_path, data_sources_main_path, file_ext=valid_ext)

    # Loading valid image in grayscale
    image_list, image_name_list = load_images(data_source_path, file_ext=valid_ext)

    # Export original images
    export_image_list(image_list, image_name_list, path=export_path, file_ext='.png')

    return


def export_set_from_orig(dataset_name, set_name, name_list):
    """
    Export a set from the original set based on the name list provided
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param set_name: string, name of the set (yet only 'train' and 'test')
    :param name_list: image name list to extract from the 'orig' set
    :return:
    """
    # Get paths
    data_paths = get_data_paths(dataset_name)

    # Load original images
    orig_image_list, orig_name_list = load_images(data_paths['orig'], file_ext='.png')

    export_image_list(orig_image_list, orig_name_list, export_name_list=name_list,
                      path=data_paths[set_name], file_ext='.png')

    return


def generate_train_images(dataset_name, name_list=None):
    """
    Generate training image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """
    # TODO: generalize for different datasets
    if name_list is None:
        name_list = ['airplane', 'arctichare', 'baboon', 'barbara', 'boat', 'cameraman', 'cat', 'goldhill', 'zelda']

    export_set_from_orig(dataset_name, set_name='train', name_list=name_list)

    return


def generate_test_images(dataset_name, name_list=None):
    """
    Generate testing image set from original set
    :param dataset_name: string, name of the dataset such that dataset.zip exists
    :param name_list: (optional) image name list to extract from the 'orig' set
    :return:
    """
    # TODO: generalize for different datasets
    if name_list is None:
        name_list = ['fruits', 'frymire', 'girl', 'monarch', 'mountain', 'peppers', 'pool', 'sails', 'tulips', 'watch']
    export_set_from_orig(dataset_name, set_name='test', name_list=name_list)

    return


def extract_2d_patches_old_as_list(image, patch_size):
    """
    Extract non-overlapping patches of size patch_height x patch_width
    :param image: array, shape = (image_height, image_width)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: patches: list of patches
    """
    image_size = np.asarray(image.shape)  # convert to numpy array to allow array computations
    patch_size = np.asarray(patch_size)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    patches = []

    # Cartesian iteration using itertools.product()
    # Equivalent to the nested for loop
    # for r in range(patches_number[0]):
    #     for c in range(patches_number[1]):
    for r, c in product(range(patches_number[0]), range(patches_number[1])):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        patches.append(image[rr:rr + patch_size[0], cc:cc + patch_size[1]])

    return patches


def extract_2d_patches(image, patch_size):
    """
    Extract non-overlapping patches of size patch_height x patch_width
    :param image: array, shape = (image_height, image_width)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: patches: array, shape = (patch_height, patch_width, patches_number)
    """
    image_size = np.asarray(image.shape)  # convert to numpy array to allow array computations
    patch_size = np.asarray(patch_size)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    # patches = np.zeros([np.prod(patches_number), patch_size[0], patch_size[1]])
    patches = np.zeros([patch_size[0], patch_size[1], np.prod(patches_number)])

    # Cartesian iteration using itertools.product()
    # Equivalent to the nested for loop
    # for r in range(patches_number[0]):
    #     for c in range(patches_number[1]):
    for k, (r, c) in zip(range(np.prod(patches_number)), product(range(patches_number[0]), range(patches_number[1]))):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        # patches[k, :, :] += image[rr:rr + patch_size[0], cc:cc + patch_size[1]]
        patches[:, :, k] += image[rr:rr + patch_size[0], cc:cc + patch_size[1]]  # TODO: use [..., k]

    return patches


def reconstruct_from_2d_patches_old_as_list(patches, image_size):
    """
    Reconstruct image from patches of size patch_height x patch_width
    :param patches: list of patches
    :param image_size: tuple of ints (image_height, image_width)
    :return: rec_image: array of shape (rec_image_height, rec_image_width)
    """
    image_size = np.asarray(image_size)         # convert to numpy array to allow array computations
    patch_size = np.asarray(patches[0].shape)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    rec_image_size = patches_number * patch_size
    rec_image = np.zeros(rec_image_size)

    # Cartesian iteration using itertools.product()
    for patch, (r, c) in zip(patches, product(range(patches_number[0]), range(patches_number[1]))):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        rec_image[rr:rr + patch_size[0], cc:cc + patch_size[1]] += patch

    return rec_image


def reconstruct_from_2d_patches(patches, image_size):
    """
    Reconstruct image from patches of size patch_height x patch_width
    :param patches: array, shape = (patch_height, patch_width, patches_number)
    :param image_size: tuple of ints (image_height, image_width)
    :return: rec_image: array of shape (rec_image_height, rec_image_width)
    """
    image_size = np.asarray(image_size)         # convert to numpy array to allow array computations
    # patch_size = np.asarray(patches[0].shape)   # convert to numpy array to allow array computations
    patch_size = np.asarray(patches[:, :, 0].shape)   # convert to numpy array to allow array computations

    if patch_size[0] > image_size[0]:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if patch_size[1] > image_size[1]:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # Patches number: floor might lead to missing parts if patch_size is a int multiplier of image_size
    patches_number = np.floor(image_size / patch_size).astype(int)

    rec_image_size = patches_number * patch_size
    rec_image = np.zeros(rec_image_size)

    # Cartesian iteration using itertools.product()
    for k, (r, c) in zip(range(np.prod(patches_number)), product(range(patches_number[0]), range(patches_number[1]))):
        rr = r * patch_size[0]
        cc = c * patch_size[1]
        # rec_image[rr:rr + patch_size[0], cc:cc + patch_size[1]] += patches[k, :, :]
        rec_image[rr:rr + patch_size[0], cc:cc + patch_size[1]] += patches[:, :, k]  # TODO: use [..., k]

    return rec_image


def reshape_patch_in_vec(patches):
    """
    :param patches: array, shape = (patch_height, patch_width, patches_number)
    :return: vec_patches: array, shape = (patch_height * patch_width, patches_number)
    """
    # Check if only a single patch (i.e. ndim = 2) or multiple patches (i.e. ndim = 3)
    if patches.ndim == 2:
        vec_patches = patches.reshape((patches.shape[0]*patches.shape[1]))
    elif patches.ndim == 3:
        vec_patches = patches.reshape((patches.shape[0]*patches.shape[1], patches.shape[-1]))
    else:
        raise TypeError('Patches cannot have more than 3 dimensions (i.e. only grayscale for now)')

    return vec_patches


def reshape_vec_in_patch(vec_patches, patch_size):
    """
    :param vec_patches: array, shape = (patch_height * patch_width, patches_number)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return patches: array, shape = (patch_height, patch_width, patches_number)
    """
    # Check if vec_patches is 1D (i.e. only one patch) or 2D (i.e. multiple patches)
    if vec_patches.ndim == 1:
        patches = vec_patches.reshape((patch_size[0], patch_size[1]))
    elif vec_patches.ndim == 2:
        patches = vec_patches.reshape((patch_size[0], patch_size[1], vec_patches.shape[-1]))
    else:
        raise TypeError('Vectorized patches array cannot be more than 2D')

    return patches


def generate_vec_set(image_list, patch_size):
    """
    Generate vectorized set of image based on patch_size
    :param image_list: list of array
    :param patch_size: tuple of ints (patch_height, patch_width)
    :return: vec_set: array, shape = (patch_height * patch_width, n_patches)
    """
    patch_list = []
    for _, image in enumerate(image_list):
        patch_list.append(extract_2d_patches(image, patch_size))

    patches = np.concatenate(patch_list, axis=-1)

    vec_set = reshape_patch_in_vec(patches)

    return vec_set


def generate_cross_validation_sets(full_set, fold_number=5, fold_combination=1):
    """
    Generate cross validations sets (i.e train and validation sets) w.r.t. a total fold number and the fold combination
    :param full_set: array, shape = (set_dim, set_size)
    :param fold_number: positive int
    :param fold_combination: int
    :return: train_set, val_set
    """
    if not isinstance(fold_combination, int):
        raise TypeError('Fold combination must be an integer')
    if not isinstance(fold_number, int):
        raise TypeError('Fold number must be an integer')
    if fold_number < 1:
        raise ValueError('Fold number must be a postive integer')
    if fold_combination > fold_number:
        raise ValueError('Fold combination must be smaller or equal to fold number')
    if not isinstance(full_set, np.ndarray):
        raise TypeError('Full set must be a numpy array')
    if full_set.ndim is not 2:
        raise TypeError('Full set must be a 2 dimensional array')

    patch_number = full_set.shape[1]
    fold_len = int(patch_number / fold_number)  # int -> floor
    val_set_start = (fold_combination - 1) * fold_len
    val_set_range = range(val_set_start, val_set_start + fold_len)
    train_set_list = [idx for idx in range(fold_number * fold_len) if idx not in val_set_range]
    train_set = full_set[:, val_set_range]
    val_set = full_set[:, train_set_list]

    return train_set, val_set


def create_gaussian_rip_matrix(size=None, seed=None):
    """
    Create a Gaussian matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional. Default is None
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    mean = 0.0
    stdev = 1 / np.sqrt(m)
    prng = np.random.RandomState(seed=seed)
    matrix = prng.normal(loc=mean, scale=stdev, size=size)

    return matrix


def create_bernoulli_rip_matrix(size=None, seed=None):
    """
    Create a Bernoulli matrix satisfying the Restricted Isometry Property (RIP).
    See: H. Rauhut - Compressive Sensing and Structured Random Matrices
    :param size: int or tuple of ints, optional. Default is None
    :param seed: int or array_like, optional
    :return: matrix: array, shape = (m, n)
    """
    m, n = size
    prng = np.random.RandomState(seed=seed)
    matrix = prng.randint(low=0, high=2, size=size).astype('float')  # gen 0, +1 sequence
    # astype('float') required to use the true divide (/=) which follows
    matrix *= 2
    matrix -= 1
    matrix /= np.sqrt(m)

    return matrix


def create_measurement_model(mm_type, patch_size, compression_percent):
    """
    Create measurement model depending on
    :param mm_type: string defining the measurement model type
    :param patch_size: tuple of ints (patch_height, patch_width)
    :param compression_percent: int
    :return: measurement_model: array, shape = (m, n)
    """
    # TODO: check if seed should be in a file rather than hardcoded
    seed = 1234567890

    patch_height, patch_width = patch_size

    n = patch_height * patch_width

    m = round((1 - compression_percent / 100) * n)

    if mm_type.lower() == 'gaussian-rip':
        measurement_model = create_gaussian_rip_matrix(size=(m, n), seed=seed)
    elif mm_type.lower() == 'bernoulli-rip':
        measurement_model = create_bernoulli_rip_matrix(size=(m, n), seed=seed)
    else:
        raise NameError('Undefined measurement model type')

    return measurement_model


def generate_transform_dict(patch_size, name, **kwargs):
    """
    Create a transform dictionary based on the name and various parameters (kwargs)
    :param patch_size: tuple of ints (patch_height, patch_width)
    :param name: string
    :param kwargs:
    :return: transform_dict: dictionary
    """
    # TODO: check how to properly document **kwargs

    transform_dict = dict()
    transform_dict['name'] = name.lower()
    transform_dict['patch_size'] = patch_size

    if transform_dict['name'] == 'dirac':
        transform_dict['level'] = kwargs.get('level')
        if transform_dict['level'] is not 0:
            warnings.warn('Level of \'dirac\' transform automatically set to 0')
            transform_dict['level'] = 0
    elif transform_dict['name'] == 'wavelet':
        transform_dict['name'] = name
        # Wavelet (default = 'db4)
        transform_dict['wavelet_type'] = kwargs.get('wavelet', 'db4')
        # TODO: check if good idea to add the wavelet object from pywt in the transform dict
        transform_dict['wavelet'] = pywt.Wavelet(transform_dict['wavelet_type'])
        # Wavelet decomposition level (default = 2)
        transform_dict['level'] = kwargs.get('level', 1)
        if not isinstance(transform_dict['level'], int):
            raise TypeError('Must be an int')
        #   Check decomposition level if not above max level
        check_wavelet_level(size=min(patch_size), dec_len=transform_dict['wavelet'].dec_len,
                            level=transform_dict['level'])
        if transform_dict['level'] < 0:
            raise ValueError(
                "Level value of %d is too low . Minimum level is 0." % transform_dict['level'])
        else:
            max_level = pywt.dwt_max_level(min(patch_size), transform_dict['wavelet'].dec_len)
            if transform_dict['level'] > max_level:
                raise ValueError(
                    "Level value of %d is too high.  Maximum allowed is %d." % (
                        transform_dict['level'], max_level))
        # Wavelet boundaries mode (default = 'symmetric')
        transform_dict['mode'] = kwargs.get('mode', 'symmetric')
        if not isinstance(transform_dict['mode'], str):
            raise TypeError('Must be a string')
    else:
        raise NotImplementedError('Only supports \'dirac\' or \'wavelet\'')

    # Compute transform coefficient number
    transform_dict['coeff_number'] = get_transform_coeff_number(transform_dict)

    return transform_dict


def check_wavelet_level(size, dec_len, level):
    # (see pywt._multilevel._check_level())
    if level < 0:
        raise ValueError('Level value of {} is too low . Minimum level is 0.'.format(level))
    else:
        max_level = pywt.dwt_max_level(size, dec_len)
        if level > max_level:
            raise ValueError('Level value of {} is too high.  Maximum allowed is {}.'.format(level, max_level))

    return


def generate_transform_list(patch_size, name_list, type_list, level_list, mode_list):
    """
    Generate transform list based on name, type, level and mode lists
    :param patch_size: tuple of ints (patch_height, patch_width)
    :param name_list: list, len = transform_number
    :param type_list: list, len = transform_number
    :param level_list: list, len = transform_number
    :param mode_list: list, len = transform_number
    :return: transform_list: list of transform dict, len = transform_number
    """
    # Check that input are of type list and are of the same size
    input_lists = [name_list, type_list, level_list, mode_list]
    input_len = []
    input_type_flag = []
    for lst in input_lists:
        input_len.append(len(lst))
        input_type_flag.append(isinstance(lst, list))
    if input_type_flag.count(True) is not len(input_type_flag):
        raise TypeError('Name, type, level and mode must be lists')
    if input_len.count(input_len[0]) is not len(input_len):
        raise ValueError('Name, type, level and mode lists must have the same length')

    transform_list = []
    for nm, wt, lvl, mode in zip(name_list, type_list, level_list, mode_list):
        transform_list.append(generate_transform_dict(patch_size, name=nm, wavelet=wt, level=lvl, mode=mode))

    return transform_list


def get_transform_coeff_number(transform_dict):
    """
    Get transform coefficient number
    :param transform_dict: transform dictionary
    :return: coeff_number: int
    """
    coeff_number = None
    # Dirac transform
    if transform_dict['name'] == 'dirac':
        coeff_number = np.prod(transform_dict['patch_size'])
    # Wavelet transform
    elif transform_dict['name'] == 'wavelet':
        # Wavelet mode: symmetric
        # TODO: check documentation which claims to have the same modes as Matlab:
        #   [link](http://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html)
        if transform_dict['mode'] == 'symmetric':
            lvl_patch_size = np.asarray(transform_dict['patch_size'], dtype=float)
            coeff_number = 0
            lvl_coeff_number = lvl_patch_size  # for the level=0 case
            for lvl in range(transform_dict['level']):
                # TODO: make sure that the level patch size used has to be "floored"
                lvl_patch_size = np.floor(0.5 * (lvl_patch_size + float(transform_dict['wavelet'].dec_len)))
                lvl_coeff_number = lvl_patch_size - 1  # bookkeeping_mat can be deduced here
                #  print('level coeff number:', lvl_coeff_number)
                coeff_number += 3 * np.prod(lvl_coeff_number).astype(int)
            # Last (approximated) level, i.e. cAn which has the same size as (cHn, cVn, cDn)
            coeff_number += np.prod(lvl_coeff_number).astype(int)
        # Wavelet mode: periodization
        elif transform_dict['mode'] == 'periodization':
            coeff_number = np.prod(transform_dict['patch_size'])
        else:
            raise NotImplementedError('Only supports \'symmetric\' and \'perdiodization\'')
    else:
        raise NotImplementedError('Only supports \'dirac\' and \'wavelet\' transform')

    return coeff_number


def wavelet_decomposition(patch_vec, transform_dict):
    """
    Compute 2D wavelet decomposition of a vectorized patch with respect to the transform parameters (transform_dict)
    See Matlab wavedec2 documentation for more information
    :param patch_vec: array, shape = (patch_height * patch_width,)
    :param transform_dict: transform dictionary
    :return: coeffs_vec, bookkeeping_mat: vectorized wavelet coefficients and bookkeeping matrix
    """

    patch_mat = reshape_vec_in_patch(patch_vec, transform_dict['patch_size'])

    # coeffs are in the shape [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)] with n the level of the decomposition
    coeffs = pywt.wavedec2(patch_mat, wavelet=transform_dict['wavelet'], mode=transform_dict['mode'],
                           level=transform_dict['level'])

    # Vectorize coeffs and compute the corresponding bookkeeping matrix S (see wavedec2 Matlab documentation)
    #   Initialization
    bookkeeping_mat = np.zeros((transform_dict['level'] + 2, 2), dtype=int)
    #   Approximated level n, i.e. cAn
    cAn = coeffs[0]
    bookkeeping_mat[0, :] = cAn.shape
    coeffs_vec = cAn.reshape(np.prod(cAn.shape))
    #   From level n to 1, i.e. (cHn, cVn, cDn) -> (cH1, cV1, cD1)
    for i, c_lvl in enumerate(coeffs[1:]):
        cHn, cVn, cDn = c_lvl
        bookkeeping_mat[i + 1, :] = cHn.shape  # cHn, cVn and cDn have the same shape
        # TODO: check if the concatenation could be safely avoided by pre-computing the final number of coefficients
        #   Check utils.get_transform_coeff_number()
        coeffs_vec = np.concatenate((coeffs_vec, cHn.reshape(np.prod(cHn.shape))))  # tf.concat
        coeffs_vec = np.concatenate((coeffs_vec, cVn.reshape(np.prod(cVn.shape))))
        coeffs_vec = np.concatenate((coeffs_vec, cDn.reshape(np.prod(cDn.shape))))

    #   Data shape
    bookkeeping_mat[-1, :] = patch_mat.shape

    return coeffs_vec, bookkeeping_mat


def wavelet_reconstruction(coeffs_vec, bookkeeping_mat, transform_dict):
    """
    Compute 2D wavelet reconstruction of a vectorized set of wavelet coefficients and its corresponding bookkeeping
    matrix and the transform parameters (transform_dict)
    See Matlab waverec2 documentation for more information
    :param coeffs_vec: vectorized wavelet coefficients
    :param bookkeeping_mat: bookkeeping matrix
    :param transform_dict: transform dictionary
    :return: patch_vec: array, shape = (patch_height * patch_width,)
    """
    # Recover the coeffs in the shape [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)] with n the level of the decomposition
    coeffs = []
    #   Approximated level n, i.e. cAn
    s_lvl = bookkeeping_mat[0, :]
    start_index = 0
    coeffs.append(coeffs_vec[start_index: start_index + np.prod(s_lvl)].reshape(s_lvl))
    start_index += np.prod(s_lvl)
    #       From level n to 1, i.e. (cHn, cVn, cDn) -> (cH1, cV1, cD1)
    for s_lvl in bookkeeping_mat[1:-1, :]:
        cHn = coeffs_vec[start_index: start_index + np.prod(s_lvl)].reshape(s_lvl)
        start_index += np.prod(s_lvl)
        cVn = coeffs_vec[start_index: start_index + np.prod(s_lvl)].reshape(s_lvl)
        start_index += np.prod(s_lvl)
        cDn = coeffs_vec[start_index: start_index + np.prod(s_lvl)].reshape(s_lvl)
        start_index += np.prod(s_lvl)
        coeffs.append((cHn, cVn, cDn))

    patch_vec = reshape_patch_in_vec(pywt.waverec2(coeffs, wavelet=transform_dict['wavelet'],
                                                   mode=transform_dict['mode']))

    return patch_vec


def multiple_transform_decomposition(patch_vec, transform_list):
    """
    Perform the decomposition of a patch in a concatenation of transforms
    :param patch_vec: array, shape = (patch_height * patch_width,)
    :param transform_list: list of transform dict
    :return: decomposition_coeff, bookkeeping_mat: list of arrays
    see Matlab wavedec2 documentation for more information)
    """
    # Check if transform_list is a list of dict
    if not is_array_of(transform_list, dict):
        raise ValueError('Transform list must be a list of dict')
    # Each transform must have the same patch_size
    patch_size_list = [tl['patch_size'] for tl in transform_list]
    if patch_size_list.count(patch_size_list[0]) is not len(transform_list):
        raise ValueError('Incoherent patch size in the concatenation of transforms.'
                         'Each transform must have the same patch size')
    # TODO: Check if patch_vec is a numpy array or tf??

    # Since multiple transforms are performed, it has to be scaled
    scale_factor = np.sqrt(len(transform_list))

    decomposition_coeff = []
    bookkeeping_mat = []
    for transform in transform_list:
        if transform['name'].lower() == 'dirac':
            decomposition_coeff.append(patch_vec/scale_factor)
            bookkeeping_mat.append(np.array((transform['patch_size'], transform['patch_size'])))  # twice to fit Matlab definition
        elif transform['name'].lower() == 'wavelet':
            cv, bk = wavelet_decomposition(patch_vec/scale_factor, transform)
            decomposition_coeff.append(cv)
            bookkeeping_mat.append(bk)
        else:
            raise NotImplementedError('Only supports \'dirac\' and \'wavelet\' transform')

    return decomposition_coeff, bookkeeping_mat


def multiple_transform_reconstruction(decomposition_coeff, bookkeeping_mat, transform_list):
    """
    Perform the reconstruction of patch by a concatenation of transforms
    :param decomposition_coeff: list of array
    :param bookkeeping_mat: list of array
    :param transform_list: list of transform dict
    :return: patch_vec: array, shape = (patch_height * patch_width,)
    """
    # TODO: tf
    if not is_array_of(decomposition_coeff, np.ndarray):
        raise ValueError('Decomposition coefficient list must be a list of np.ndarray')
    if not is_array_of(bookkeeping_mat, np.ndarray):
        raise ValueError('Bookkeeping matrix list must be a list of np.ndarray')
    # Check if transform_list is a list of dict
    if not is_array_of(transform_list, dict):
        raise ValueError('Transform list must be a list of dict')
    # Each transform must have the same patch_size
    patch_size_list = [tl['patch_size'] for tl in transform_list]
    if patch_size_list.count(patch_size_list[0]) is not len(transform_list):
        raise ValueError('Incoherent patch size in the concatenation of transforms. '
                         'Each transform must have the same patch size')

    patch_size = transform_list[0]['patch_size']
    patch_vec = np.zeros((np.prod(patch_size)))
    scale_factor = np.sqrt(len(transform_list))
    for cv, bk, transform in zip(decomposition_coeff, bookkeeping_mat, transform_list):
        if transform['name'].lower() == 'dirac':
            patch_vec += cv / scale_factor
        elif transform['name'].lower() == 'wavelet':
            patch_vec += wavelet_reconstruction(cv, bk, transform) / scale_factor
        else:
            raise NotImplementedError('Only supports \'dirac\' and \'wavelet\' transform')

    return patch_vec


def plot_image_set(image_list, name_list, fig=None, sub_plt_n_w=4):
    """
    Plot an image set given as a list
    :param image_list: list of images
    :param name_list: list of names
    :param fig: figure obj
    :param sub_plt_n_w: int, number of subplot spaning the width
    :return:
    """
    # TODO: align images 'top'
    if fig is None:
        fig = plt.figure()

    sub_plt_n_h = int(np.ceil(len(image_list) / sub_plt_n_w))
    ax = []
    for i, (im, im_name) in enumerate(zip(image_list, name_list)):
        ax.append(fig.add_subplot(sub_plt_n_h, sub_plt_n_w, i + 1))
        ax[i].imshow(im, cmap='gray', vmin=im.min(), vmax=im.max())
        ax[i].set_axis_off()
        ax[i].set_title('{}\n({}, {})'.format(im_name, im.shape[0], im.shape[1]), fontsize=10)


def plot_image_with_cbar(image, title=None, cmap='gray', vmin=None, vmax=None, ax=None):
    """
    Plot an image with it's colorbar
    :param image: array, shape = (image_height, image_width)
    :param title: option title
    :param cmap: optional cmap
    :param vmin: optional vmin
    :param vmax: optional vmax
    :param ax: optional axis
    :return:
    """
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title is not None:
        ax.set_title('{}'.format(title), fontsize=12)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)


def plot_decomposition_coeffs(coeffs_vec, title=None, ax=None, theta=None, theta_same_col=False):
    """
    Plot decomposition coefficients and the corresponding threshold (theta) if given
    :param coeffs_vec: array, shape = (patch_height * patch_width,)
    :param title: optional title
    :param ax: optional axis
    :param theta: optional threshold
    :param theta_same_col: optional color flag
    :return:
    """
    if ax is None:
        ax = plt.gca()

    base_line, = ax.plot(coeffs_vec)
    if title is not None:
        ax.set_title('{}'.format(title), fontsize=12)
    if theta is not None:
        # If theta is a scalar, i.e. same threshold applied to each coefficients, a vector of theta is created
        theta_plt = theta*np.ones(coeffs_vec.shape)
        if theta_same_col:
            ax.plot(theta_plt, '--', color=base_line.get_color())
        else:
            ax.plot(theta_plt, '--')


def convert_transform_dict_to_tf(transform_dict):
    """
    Transform dictionary conversion to tensorflow
    :param transform_dict:
    :return: tf_transform_dict
    """
    tf_transform_dict = dict()
    tf_transform_dict['name'] = tf.constant(transform_dict['name'], tf.string)
    tf_transform_dict['patch_size'] = transform_dict['patch_size']
    # tf_transform_dict['tf_patch_size'] = tf.TensorShape(dims=transform_dict['patch_size'])
    tf_transform_dict['coeff_number'] = tf.constant(transform_dict['coeff_number'], tf.int64)
    tf_transform_dict['level'] = tf.constant(transform_dict['level'], tf.int32)
    if transform_dict['name'] == 'dirac':
        pass
    elif transform_dict['name'] == 'wavelet':
        tf_transform_dict['mode'] = tf.constant(transform_dict['mode'], tf.string)
        tf_transform_dict['wavelet_type'] = tf.constant(transform_dict['wavelet_type'], tf.string)

    return tf_transform_dict


def convert_transform_list_to_tf(transform_list):
    """
    List of transform dictionary conversion to tensorflow
    :param transform_list:
    :return: tf_transform_list
    """
    tf_transform_list = [convert_transform_dict_to_tf(transform_dict) for transform_dict in transform_list]

    return tf_transform_list

def tf_pywt_wavelet_decomposition(patch_vec, patch_size, name, wavelet_type, level, mode):
    """

    :param patch_vec:
    :param patch_size:
    :param name:
    :param wavelet_type:
    :param level:
    :param mode:
    :return:
    """
    # TODO: docstring

    # Convert input values for pywt
    wavelet_type = wavelet_type.decode('utf-8')
    mode = mode.decode('utf-8')
    level = int(level)
    patch_size = tuple(patch_size)
    name = name.decode('utf-8')
    # print('wavelet_type: {}, {}'.format(wavelet_type, type(wavelet_type)))
    # print('mode: {}, {}'.format(mode, type(mode)))
    # print('level: {}, {}'.format(level, type(level)))
    # print('patch_vec: {}, {}'.format(patch_vec, type(patch_vec)))
    # print('patch_size: {}, {}'.format(patch_size, type(patch_size)))
    # print('name: {}, {}'.format(name, type(name)))

    # Rebuild transform_dict from unpacked inputs
    transform_dict = generate_transform_dict(patch_size, name, wavelet=wavelet_type, level=level, mode=mode)

    # print(transform_dict)

    # Decomposition
    coeffs_vec, bookkeeping_mat = wavelet_decomposition(patch_vec, transform_dict)

    return coeffs_vec.astype(np.float32), bookkeeping_mat.astype(np.int32)


def tf_pywt_wavelet_reconstruction(coeffs_vec, bookkeeping_mat, patch_size, name, wavelet_type, level, mode):
    """

    :param coeffs_vec:
    :param bookkeeping_mat:
    :param patch_size:
    :param name:
    :param wavelet_type:
    :param level:
    :param mode:
    :return:
    """
    # TODO: docstring

    # Convert input values for pywt
    # print(coeffs_vec, type(coeffs_vec))
    # print(bookkeeping_mat, type(bookkeeping_mat))
    wavelet_type = wavelet_type.decode('utf-8')
    mode = mode.decode('utf-8')
    level = int(level)
    patch_size = tuple(patch_size)
    name = name.decode('utf-8')
    # print('wavelet_type: {}, {}'.format(wavelet_type, type(wavelet_type)))
    # print('mode: {}, {}'.format(mode, type(mode)))
    # print('level: {}, {}'.format(level, type(level)))
    # print('patch_vec: {}, {}'.format(patch_vec, type(patch_vec)))
    # print('patch_size: {}, {}'.format(patch_size, type(patch_size)))
    # print('name: {}, {}'.format(name, type(name)))

    # Rebuild transform_dict from unpacked inputs
    transform_dict = generate_transform_dict(patch_size, name, wavelet=wavelet_type, level=level, mode=mode)

    # Reconstruction
    patch_vec = wavelet_reconstruction(coeffs_vec, bookkeeping_mat, transform_dict)

    return patch_vec.astype(np.float32)


def tf_wavelet_decomposition(tf_patch_vec, tf_transform_dict, flag_pywt=True):

    # ONLY POSSIBLE YET USING THE PYWT INTERFACE
    # TODO: wavelet decomposition within tf
    if not flag_pywt:
        raise NotImplementedError('Only possible using the PyWavelet interface')
    tf_coeffs_vec, tf_bookkeeping_mat = \
        tf.py_func(tf_pywt_wavelet_decomposition, [tf_patch_vec,
                                                   tf_transform_dict['patch_size'],
                                                   tf_transform_dict['name'],
                                                   tf_transform_dict['wavelet_type'],
                                                   tf_transform_dict['level'],
                                                   tf_transform_dict['mode']],
                   Tout=[tf.float32, tf.int32])

    return tf_coeffs_vec, tf_bookkeeping_mat


def tf_wavelet_reconstruction(tf_coeffs_vec, tf_bookkeeping_mat, tf_transform_dict, flag_pywt=True):

    # ONLY POSSIBLE YET USING THE PYWT INTERFACE
    # TODO: wavelet decomposition within tf
    if not flag_pywt:
        raise NotImplementedError('Only possible using the PyWavelet interface')
    tf_patch_vec = tf.py_func(tf_pywt_wavelet_reconstruction, [tf_coeffs_vec, tf_bookkeeping_mat,
                                                               tf_transform_dict['patch_size'],
                                                               tf_transform_dict['name'],
                                                               tf_transform_dict['wavelet_type'],
                                                               tf_transform_dict['level'],
                                                               tf_transform_dict['mode']],
                              Tout=tf.float32)

    return tf_patch_vec


def tf_multiple_transform_decomposition(tf_patch_vec, tf_transform_list):

    # TODO: some checks as the non-tf version
    # TODO: condition doesn't work with tf.cond()

    # # If True, i.e. 'dirac'
    # def f1():
    #     tf_cv = tf_patch_vec / scale_factor
    #     tf_bk = tf.constant(transform['patch_size'])
    #     return tf_cv, tf_bk
    #
    # # If False, i.e. 'wavelet'
    # def f2():
    #     tf_cv, tf_bk = tf_wavelet_decomposition(tf_patch_vec / scale_factor, transform)
    #     return tf_cv, tf_bk

    # Since multiple transforms are performed, it has to be scaled
    scale_factor = tf.sqrt(float(len(tf_transform_list)))

    tf_decomposition_coeff = []
    tf_bookkeeping_mat = []

    for transform in tf_transform_list:
        # TODO: use tf.case which seems more safe and can handle other possibility
        # tf_cv, tf_bk = tf.cond(tf.equal(transform['name'], tf.constant('dirac', tf.string)), f1, f2)
        # Shape is unknown since it comes from a python interface for the moment
        tf_cv, tf_bk = tf_wavelet_decomposition(tf_patch_vec/scale_factor, transform)
        tf_decomposition_coeff.append(tf_cv)
        tf_bookkeeping_mat.append(tf_bk)

    return tf_decomposition_coeff, tf_bookkeeping_mat


def tf_multiple_transform_reconstruction(tf_decomposition_coeff, tf_bookkeeping_mat, tf_transform_list):

    # TODO: some checks as the non-tf version
    # TODO: condition doesn't work with tf.cond()

    # Same patch sizes for each transform
    patch_size = tf_transform_list[0]['patch_size']
    tf_patch_vec = tf.zeros((np.prod(patch_size)), dtype=tf.float32)

    scale_factor = tf.sqrt(float(len(tf_transform_list)))
    for cv, bk, transform in zip(tf_decomposition_coeff, tf_bookkeeping_mat, tf_transform_list):
        # TODO: condition doesn't work with tf.cond(), transform['name'] should be checked!! only wavelet now
        tf_patch_vec += tf_wavelet_reconstruction(cv, bk, transform) / scale_factor

    return tf_patch_vec


def tf_soft_thresholding(coeff, theta):
    return tf.mul(tf.sign(coeff), tf.maximum(tf.constant(0, dtype=tf.float32), tf.sub(tf.abs(coeff), theta)))