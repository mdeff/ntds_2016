"""
This module contains the function to run the cnn model
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import cnn
import utils
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from ipywidgets import FloatProgress
from IPython.display import display



def run_model_1(train_data, train_labels, test_data, test_labels, test_patients, dropout_keep, iterations, PATCH_NUMBER, PATCH_SIZE, TEST_FREQUENCY = 500, correction=1, strides='pool'):
    """
        run cnn model
    
        Usage: [losses, acc_train_patch, sens_train_patch, spec_train_patch, acc_test_patch, sens_test_patch, spec_test_patch, acc_test_image, sens_test_image, spec_test_image, acc_test_patient, sens_test_patient, spec_test_patient, mean_batch_time, mean_test_time, cnf_matrix, W] = run_model_1(train_data, train_labels, test_data, test_labels, test_patients, dropout_keep, iterations, PATCH_NUMBER, PATCH_SIZE, TEST_FREQUENCY = 500, correction=1, strides='pool')

        Input variables:
            train_data: vectorized train data
            train_labels: labels of train data
            test_data: vectorized test data
            test_labels: labels of test data
            test_patients: the patients corresponding to the test data
            dropout_keep: probability of keeping in dropout
            iterations: number of iterations to do
            PATCH_NUMBER: number of patches per image
            PATCH_SIZE: size of patch
            TEST_FREQUENCY: the number of iterations between evaluations of the test data
            correction: 1 if bias correction needed
            strides: place where the subsampling should happen, either 'pool' or 'conv'

    """
    
    ## DEFINE PLOTS
    (fig, ax_acc_train, ax_sens_train, ax_loss, ax_acc_p_test, ax_sens_p_test, ax_acc_i_test, ax_sens_i_test, ax_acc_pat_test, ax_sens_pat_test, ax_pred) = utils.define_plots()
    
    ## DEFINE PROGRESS BAR
    pb = FloatProgress(min=0, max=iterations)
    display(pb)
    
    
    ## DEFINE MODEL PARAMETERS
    NC = train_labels.shape[1] #number of classes

    # first convolutional layer
    KERNEL_1_SIZE = (5,5)
    KERNEL_1_NUM = 32
    MAXPOOL_1_SIZE = (3,3)
    if strides=='conv':
        KERNEL_1_STRIDE = (2,2)
        MAXPOOL_1_STRIDE = (1,1)
    elif strides=='pool': 
        KERNEL_1_STRIDE = (1,1)
        MAXPOOL_1_STRIDE = (2,2)
    else:
        raise Error('strides argument should be either conv or pool')

    # second convolutional layer
    KERNEL_2_SIZE = (5,5)
    KERNEL_2_NUM = 32
    MAXPOOL_2_SIZE = (3,3)
    if strides=='conv':
        KERNEL_2_STRIDE = (2,2)
        MAXPOOL_2_STRIDE = (1,1)
    elif strides=='pool': 
        KERNEL_2_STRIDE = (1,1)
        MAXPOOL_2_STRIDE = (2,2)
    else:
        raise Error('strides argument should be either conv or pool')

    # third convolutional layer
    KERNEL_3_SIZE = (5,5)
    KERNEL_3_NUM = 32
    MAXPOOL_3_SIZE = (3,3)
    if strides=='conv':
        KERNEL_3_STRIDE = (2,2)
        MAXPOOL_3_STRIDE = (1,1)
    elif strides=='pool': 
        KERNEL_3_STRIDE = (1,1)
        MAXPOOL_3_STRIDE = (2,2)
    else:
        raise Error('strides argument should be either conv or pool')



    # first fully connected layer
    FC_1_SIZE = 64

    # adam optimiser initial value
    OPTIMISER_VALUE = .001
   

    ## DEFINE COMPUTATIONAL GRAPH (CG)
    dd = train_data.shape[1]  # data dimensionality

    # CG inputs
    xin = tf.placeholder(tf.float32,[None,dd]); #print('xin=',xin,xin.get_shape()) 
    y_label = tf.placeholder(tf.float32,[None,NC]); #print('y_label=',y_label,y_label.get_shape())
    d = tf.placeholder(tf.float32);
    distr = tf.placeholder(tf.float32,[NC]) #distribution of data within classes for anti-biasing in loss term

    # Convolutional layer 1
    x_2d = tf.reshape(xin, [-1,PATCH_SIZE[0],PATCH_SIZE[1],3]);
    (x, Wcl, bcl) = cnn.define_CL(KERNEL_1_SIZE, 3, KERNEL_1_STRIDE, KERNEL_1_NUM, x_2d)
    x = cnn.define_MP(MAXPOOL_1_SIZE, MAXPOOL_1_STRIDE,x)
    x = tf.nn.relu(x)
    
    # Convolutional layer 2
    (x,Wcl2,bcl2) = cnn.define_CL(KERNEL_2_SIZE, KERNEL_1_NUM, KERNEL_2_STRIDE, KERNEL_2_NUM, x)
    x = cnn.define_AP(MAXPOOL_2_SIZE, MAXPOOL_2_STRIDE, x)
    x = tf.nn.relu(x)
    
    # Convolutional layer 3
    (x,Wcl3,bcl3) = cnn.define_CL(KERNEL_3_SIZE, KERNEL_2_NUM, KERNEL_3_STRIDE, KERNEL_3_NUM, x)
    x = cnn.define_AP(MAXPOOL_3_SIZE, MAXPOOL_3_STRIDE, x)
    x = tf.nn.relu(x)  

    # Fully Connected layer 1
    downsampling = KERNEL_1_STRIDE[0]*KERNEL_1_STRIDE[1]*KERNEL_2_STRIDE[0]*KERNEL_2_STRIDE[1]*KERNEL_3_STRIDE[0]*KERNEL_3_STRIDE[1]*MAXPOOL_1_STRIDE[0]*MAXPOOL_1_STRIDE[1]*MAXPOOL_2_STRIDE[0]*MAXPOOL_2_STRIDE[1]*MAXPOOL_3_STRIDE[0]*MAXPOOL_3_STRIDE[1]
    nfc = np.int((PATCH_SIZE[0]*PATCH_SIZE[1]*KERNEL_3_NUM)/downsampling)
    x = tf.reshape(x, [-1,nfc]);
    (x, Wfc1, bfc1) = cnn.define_FC(nfc, FC_1_SIZE, x)
    x = tf.nn.relu(x)

    # Dropout
    x = tf.nn.dropout(x, d)

    # Output layer / Fully connected layer 2
    (y, Wfc2, bfc2) = cnn.define_FC(FC_1_SIZE, NC, x)
    y = tf.nn.softmax(y); #print('y=',y,y.get_shape())

    # Loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(1/distr*y_label* tf.log(y+1e-10), 1))
    total_loss = cross_entropy

    # Optimization scheme
    train_step = tf.train.AdamOptimizer(OPTIMISER_VALUE).minimize(total_loss)

    # Output
    predictions = tf.argmax(y,1)
    truth = tf.argmax(y_label,1)
    correct_prediction = tf.equal(predictions, truth)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    split = int(NC/2)
    sensitivity = tf.reduce_sum(tf.cast(tf.greater_equal(predictions,split),tf.float32)*tf.cast(tf.greater_equal(truth,split), tf.float32))/tf.reduce_sum(tf.cast(tf.greater_equal(truth,split),tf.float32))
    specificity = tf.reduce_sum(tf.cast(tf.less(predictions,split),tf.float32)*tf.cast(tf.less(truth,split), tf.float32))/tf.reduce_sum(tf.cast(tf.less(truth,split),tf.float32))
    
  
    
    ## RUNNING CG

    # Define results
    losses = []
    acc_train_patch = []
    sens_train_patch = []
    spec_train_patch = []

    acc_test_patch = []
    sens_test_patch = []
    spec_test_patch = []

    acc_test_image = []
    sens_test_image = []
    spec_test_image = []
    
    acc_test_patient = []
    sens_test_patient = []
    spec_test_patient = []    

    # Run Computational Graph
    n = train_data.shape[0]
    indices = collections.deque()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    batch_size = 100

    if correction==1:
        [distribution,_] = np.histogram(np.argmax(train_labels, axis=1),np.arange(0,NC+1))
        distribution = distribution/train_labels.shape[0]
    else:
        distribution = np.ones(shape=(NC),dtype=float)/NC
    
    mean_batch_time = 0
    mean_test_time = 0

    
    for i in range(iterations):
        pb.value+=1
        
        # Batch extraction
        if len(indices) < batch_size:
            indices.extend(np.random.permutation(n)) 
        idx = [indices.popleft() for i in range(batch_size)]
        batch_x, batch_y = train_data[idx,:], train_labels[idx]

        # Run CG for variable training

        start = timer()
        _, W, y_train, pred_train, acc_train, sens_train, spec_train, total_loss_o = sess.run([train_step,Wcl,y,predictions,accuracy,sensitivity,specificity,total_loss], feed_dict={xin: batch_x, y_label: batch_y, d: dropout_keep, distr: distribution})
        end = timer()
        mean_batch_time = (mean_batch_time*i+end-start)/(i+1)

        losses.append(total_loss_o)
        acc_train_patch.append(acc_train)
        sens_train_patch.append(sens_train)
        spec_train_patch.append(spec_train)
        
        if not i%100:
            
            
            # Update plots
            utils.update_accuracy_plot(fig, ax_acc_train, acc_train_patch, title='Patch minibatch accuracy')
            utils.update_sensitivity_plot(fig, ax_sens_train, sens_train_patch, spec_train_patch)
            utils.update_loss_plot(fig, ax_loss, losses)

            fig.canvas.draw()
        
        # Run CG for test set
        if not i%TEST_FREQUENCY:
            start = timer()
            y_test, pred_test, acc_test, sens_test, spec_test = sess.run([y,predictions,accuracy,sensitivity,specificity], feed_dict={xin: test_data, y_label: test_labels, d:1.0})
            end = timer()
            mean_test_time = (mean_test_time*i+end-start)/(i+1)

            (acc_im,sens_im,spec_im,pred_im) = utils.compute_accuracy_image(y_test, np.argmax(test_labels,axis=1), PATCH_NUMBER, NC)
            (acc_patient, sens_patient, spec_patient) = utils.compute_accuracy_patient(y_test, np.argmax(test_labels,axis=1), test_patients, NC)
            acc_test_patch.append(acc_test)
            sens_test_patch.append(sens_test)
            spec_test_patch.append(spec_test)
            acc_test_image.append(acc_im)
            sens_test_image.append(sens_im)
            spec_test_image.append(spec_im)
            acc_test_patient.append(acc_patient)
            sens_test_patient.append(sens_patient)
            spec_test_patient.append(spec_patient)

            # Update plots
            utils.update_accuracy_plot(fig, ax_acc_p_test, acc_test_patch,title='Patch test accuracy',iteration_factor=TEST_FREQUENCY)
            utils.update_sensitivity_plot(fig, ax_sens_p_test, sens_test_patch, spec_test_patch)

            utils.update_accuracy_plot(fig, ax_acc_i_test, acc_test_image,title='Image test accuracy',iteration_factor=TEST_FREQUENCY)
            utils.update_sensitivity_plot(fig, ax_sens_i_test, sens_test_image, spec_test_image)
            
            utils.update_accuracy_plot(fig, ax_acc_pat_test, acc_test_patient,title='Patient test accuracy',iteration_factor=TEST_FREQUENCY)
            utils.update_sensitivity_plot(fig, ax_sens_pat_test, sens_test_patient, spec_test_patient)

            utils.update_prediction_plot(fig, ax_pred, pred_im, np.argmax(test_labels[::PATCH_NUMBER],axis=1),title='Image Prediction')
            
            fig.canvas.draw()
            

    # Calculate confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(test_labels,axis=1), pred_test)

   
    
    return (losses, acc_train_patch, sens_train_patch, spec_train_patch, acc_test_patch, sens_test_patch, spec_test_patch, acc_test_image, sens_test_image, spec_test_image, acc_test_patient, sens_test_patient, spec_test_patient, mean_batch_time, mean_test_time, cnf_matrix, W)
    
        
        

    
    
