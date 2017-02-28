import numpy as np
import matplotlib.pyplot as plt    
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import scipy
from scipy import sparse
import tensorflow as tf
import os.path
import collections
import time
import pandas as pd
from datetime import datetime
from datetime import timedelta

from DataLoader import DataLoader

class CNN:
    
    STR_RUN = 'test_'
    STR_MODEL1 = 'basic_'
    STR_MODEL2 = 'lenet5_'
    STR_DIR_RUN = 'run/'
    
    @staticmethod
    def normalize_data(data=None):
        """ Normalize data for CNN computation. Centered in 0 and normalized to 1
        """
        # Data pre-processing
        n = data.shape[0]
        for i in range(n):
            xx = data[i,:,:]
            xx -= np.mean(xx) # Centering in 0
            xx /= np.linalg.norm(xx) # Normalizing to 1
            data[i] = xx # Affect value
        return data
    
    @staticmethod
    def get_data(patients=None, mris_train=None, batch_size=100, id_train=0, is_one_hot=True):
        """ Get formated trainning and validation data and labels
            * train_data, train_labels : Data and the label for training data
            * valid_data, valid_labels : Data and the label for validation data
            * batch_size : Size of the batch
        """
    
        # Get size of both sets (defined by batch size)
        Npatients_train = (mris_train.shape[0]//batch_size)*batch_size

        # Get labels for both sets
        patients_train = patients[patients['train_valid_test'] == id_train][:Npatients_train]
        if is_one_hot:
            train_labels = CNN.convert_to_one_hot(patients_train['diagnosis']-1, len( np.unique(patients_train['diagnosis'])))
        else:
            train_labels = (patients_train['diagnosis']-1).as_matrix()

        # Get only wanted data (multiple of batch size) and convert to float
        train_data = mris_train[:Npatients_train, :]
        train_data = train_data.astype('float32')

        # Data pre-processing for train and validation
        train_data = CNN.normalize_data(train_data)

        # Print final shapes of data and labels
        # print('Train data shape=', train_data.shape)
        # print('Train data labels=', train_labels.shape, ' labels_sum=', np.sum(train_labels, axis=0))
        
        return train_data, train_labels
    
    @staticmethod
    def run_basic_cnn(train_data=None, train_labels=None, valid_data=None, valid_labels=None, batch_size=100, 
                      K=5, F=10, drop=0.25, learning_rate=0.001, reg_par = 1*1e-3, 
                      n_iter = 3000, print_iter=200, log_iter=100, run_name='run_model_basic.npy'):
        """ Running basic CNN using model : CL10-FC3
            * train_data, train_labels : Data and the label for training data
            * valid_data, valid_labels : Data and the label for validation data
            * batch_size : Size of the batch
            * K : Size of the filter
            * F : Number of filters
            * drop : Dropout factor (random points set to 0)
            * learning_rate : Adam optimizer learning rate
            * reg_par : Regularization parameter
            * n_iter : Number of iteration to complete
            * print_iter : Display accuracy on validation each print_iter iteration
            * log_iter : Save accuracy results each n iteration
            * run_name : Name of the file the run is saved to
        """
        
        # Check is correct args as input
        if train_data is None or train_labels is None or valid_data is None or valid_labels is None:
            print("ERROR - RUN_BASIC_CNN - Argument is None", nbr)
            return

        # Define computational graph (CG)
        d1 = train_data.shape[1]    # data dimensionality
        d2 = train_data.shape[2]    # data dimensionality
        nc = train_labels.shape[1]  # number of classes
        ncl = K*K*F
        nfc = d1*d2*F

        # CNN inputs variables
        xin = tf.placeholder(tf.float32,[batch_size, d1, d2]); #print('xin=',xin,xin.get_shape())
        y_label = tf.placeholder(tf.float32,[batch_size,nc]);  #print('y_label=',y_label,y_label.get_shape())
        d = tf.placeholder(tf.float32);
        
        # CNN learning variables
        Wcl = tf.Variable(tf.truncated_normal([K,K,1,F], stddev=tf.sqrt(2./tf.to_float(ncl)) ));
        bcl = tf.Variable(tf.zeros([F])); 
        Wfc = tf.Variable(tf.truncated_normal([nfc,nc], stddev=tf.sqrt(6./tf.to_float(nfc+nc))));
        bfc = tf.Variable(tf.zeros([nc]));
        
        # Layer No 1 --- Convolution
        x_2d = tf.reshape(xin, [-1,d1,d2,1]); 
        x = tf.nn.conv2d(x_2d, Wcl, strides=[1, 1, 1, 1], padding='SAME') + bcl;
        x = tf.nn.relu(x)
        
        # Layer No 2 --- Fully connected
        x = tf.nn.dropout(x, d)
        x = tf.reshape(x, [batch_size,-1]); 
        y = tf.matmul(x, Wfc) + bfc; 
        y = tf.nn.softmax(y);

        # Loss
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), 1))
        # L2 Regularization
        reg_loss = tf.nn.l2_loss(Wfc) + tf.nn.l2_loss(bfc)
        total_loss = cross_entropy + reg_par* reg_loss

        # Optimization scheme
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # ----------- Run Computational Graph
        n = train_data.shape[0]
        indices = collections.deque()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        # Keep trace of running process
        t_start = datetime.now()
        acc_train_tot = []; acc_valid_tot = []; loss_tot = []

        for i in range(n_iter+1):

            # Batch extraction (random permutation)
            if len(indices) < batch_size:
                indices.extend(np.random.permutation(n)) 
            idx = [indices.popleft() for i in range(batch_size)]
            batch_x, batch_y = train_data[idx,:,:], train_labels[idx]

            # Run CNN for variable training (feed dictionnary with batch data, label and dropout)
            _,acc_train,total_loss_o = sess.run([train_step,accuracy,total_loss], feed_dict={xin: batch_x, y_label: batch_y, d: drop})

            # Run CNN for validation set
            if (not i%log_iter) or (not i%print_iter):
                nrange = valid_data.shape[0]//batch_size
                tot_acc = 0
                y_est = np.zeros((valid_data.shape[0], 3))
                for j in range(nrange):
                    acc_valid, y_o, Wcl_o, bcl_o, Wfc_o, bfc_o = \
                    sess.run([accuracy, y, Wcl, bcl, Wfc, bfc],
                             feed_dict={xin: valid_data[j*batch_size:(j+1)*batch_size],
                                        y_label: valid_labels[j*batch_size:(j+1)*batch_size], d: 1.0})
                    tot_acc += acc_valid
                    y_est[j*batch_size:(j+1)*batch_size, :] = y_o
                    
                acc_train_tot.append(acc_train); 
                acc_valid_tot.append(np.round(tot_acc/nrange,4));
                loss_tot.append(total_loss_o)
                # Print if needed
                if not i%print_iter:
                    print('\nIteration i=',i,', train accuracy=',acc_train,', loss=',total_loss_o,
                              'time spend=', datetime.now()-t_start)
                    print('valid accuracy=', np.round(tot_acc/nrange,4))
                    
        # Save run settings
        data_run = {'K':K, 'F':F, 'drop':drop, 'learning_rate':learning_rate, 'reg_par':reg_par, 
                    'n_iter':n_iter, 't_tot':(datetime.now()-t_start).total_seconds(),
                    'y_cgt':valid_labels, 'y_est':y_est, 'loss':loss_tot,
                    'acc_train_tot':acc_train_tot, 'acc_valid_tot':acc_valid_tot,
                    'Wcl':Wcl_o, 'bcl':bcl_o, 'Wfc':Wfc_o, 'bfc':bfc_o}
        
        
        DataLoader.save_run(data_run, run_name) 
    
    @staticmethod
    def apply_basic_cnn(data, label, file_data_model):
        """ Apply basic CNN using model : CL10-FC3 n data
            * data : Data to aply model on
            * label : The label for data (ground truth)
            * file_data_model : Name of the file that contains the model data (located in STR_DIR_RUN)
        """
        
        # Get input learned data
        data_model = np.load(os.path.join(CNN.STR_DIR_RUN, file_data_model)).item()
        K = data_model['K']
        F = data_model['F']
        Wcl_i = data_model['Wcl'].astype('float32')
        bcl_i = data_model['bcl'].astype('float32')
        Wfc_i = data_model['Wfc'].astype('float32')
        bfc_i = data_model['bfc'].astype('float32')
        
        # Deifne model sizes
        batch_size = data.shape[0]
        d1 = data.shape[1]    # data dimensionality
        d2 = data.shape[2]    # data dimensionality
        nc = label.shape[1]  # number of classes
        ncl = K*K*F
        nfc = d1*d2*F
        
        # Create model
        # CNN inputs variables
        xin = tf.placeholder(tf.float32,[batch_size, d1, d2]); 
        y_label = tf.placeholder(tf.float32,[batch_size,nc]); 
        Wcl = tf.placeholder(tf.float32,[K,K,1,F]);
        bcl = tf.placeholder(tf.float32,[F]); 
        Wfc = tf.placeholder(tf.float32,[nfc,nc]);
        bfc = tf.placeholder(tf.float32,[nc]); 

        # Layer No 1 --- Convolution
        x = tf.reshape(xin, [-1,d1,d2,1]); 
        x = tf.nn.conv2d(x, Wcl, strides=[1, 1, 1, 1], padding='SAME') + bcl;
        x = tf.nn.relu(x)

        # Layer No 2 --- Fully connected
        x = tf.reshape(x, [batch_size,-1]); 
        y = tf.matmul(x, Wfc) + bfc; 
        y = tf.nn.softmax(y);
        
        # Compute accuracy
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # Run model
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        acc_data = sess.run([accuracy], feed_dict={xin: data, y_label: label, 
                                                   Wcl: Wcl_i, bcl: bcl_i, Wfc: Wfc_i, bfc: bfc_i})
        return np.round(acc_data, 4)[0]
    
    @staticmethod
    def apply_base_cnn_deconv(data, patients, file_data_model):
        # Get input learned data
        data_model = np.load(os.path.join(CNN.STR_DIR_RUN, file_data_model)).item()
        K = data_model['K']
        F = data_model['F']

        Wcl_i = data_model['Wcl']
        bcl_i = data_model['bcl']
        
        # Define settings
        batch_size = data.shape[0]
        d1 = data.shape[1]    # data dimensionality
        d2 = data.shape[2]    # data dimensionality
        n_img = d1//d2        # number of concatenated images
        nc = 3                # number of classes
        ncl1 = K*K*F
        
        # CNN inputs variables
        xin = tf.placeholder(tf.float32,[batch_size, d1, d2]);  
        Wcl = tf.placeholder(tf.float32,[K,K,1,F]) 
        bcl = tf.placeholder(tf.float32,[F]) 
        
        # Layer 1 --- Convolution layer
        x = tf.reshape(xin, [-1,d1,d2,1])
        x = tf.nn.conv2d(x, Wcl, strides=[1, 1, 1, 1], padding='SAME') + bcl 
        x = tf.nn.relu(x)

        # Undo --- Layer 1
        x = tf.nn.relu(x)
        x = tf.nn.conv2d_transpose(x-bcl, Wcl, output_shape=[batch_size, d1, d2, 1], strides=[1, 1, 1, 1], padding='SAME') 
        
        # X out performed
        x_out = x

        # Run model
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        x_out_o = sess.run([x_out], feed_dict={xin: data, Wcl: Wcl_i, bcl: bcl_i,})
        x_out_o = np.array(x_out_o).squeeze()
        x_rect = np.maximum(x_out_o, 0)

        id_normal = np.nonzero(patients['diagnosis'][:x_rect.shape[0]] == 1)[0]
        id_mci = np.nonzero(patients['diagnosis'][:x_rect.shape[0]] == 2)[0]
        id_ad = np.nonzero(patients['diagnosis'][:x_rect.shape[0]] == 3)[0]

        id_normal_max = np.argsort(np.linalg.norm(x_rect[id_normal], axis=(1,2)))[-2]
        id_mci_max = np.argmax(np.linalg.norm(x_rect[id_mci], axis=(1,2)))
        id_ad_max = np.argmax(np.linalg.norm(x_rect[id_ad], axis=(1,2)))
        
        n = 6
        fig = plt.figure(figsize=(16,4*n))
        plt.subplot(1,n,1); 
        plt.imshow(data[id_normal[id_normal_max]]);
        plt.axis('off'); plt.title('Normal [' + str(id_normal[id_normal_max]) + ']')
        plt.subplot(1,n,2);
        plt.imshow(x_rect[id_normal[id_normal_max]], cmap='seismic');
        plt.axis('off'); plt.title('Normal Activation')
        plt.subplot(1,n,3); plt.axis('off');
        plt.imshow(data[id_mci[id_mci_max]])
        plt.title('MCI [' + str(id_mci[id_mci_max]) + ']')
        plt.subplot(1,n,4); 
        plt.imshow(x_rect[id_mci[id_mci_max]], cmap='seismic')
        plt.axis('off'); plt.title('MCI Activation')
        plt.subplot(1,n,5); 
        plt.imshow(data[id_ad[id_ad_max]])
        plt.axis('off'); plt.title('AD [' + str(id_ad[id_ad_max]) + ']')
        plt.subplot(1,n,6); 
        plt.imshow(x_rect[id_ad[id_ad_max]], cmap='seismic')
        plt.axis('off'); plt.title('AD Activation')
        plt.show();
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'cnn_basic_deconv.pdf')
 
    
    @staticmethod
    def run_leNet5_cnn(train_data=None, train_labels=None, valid_data=None, valid_labels=None, batch_size=100, 
                      K=5, F1=10, F2=20, n_feat2=512, drop=0.5, learning_rate=0.001, reg_par = 1*1e-4, 
                      n_iter = 3000, print_iter=200, log_iter=100, run_name='run_model_basic.npy'):
        """ Running basic CNN using leNet5 model : CL10-MP4-CL20-MP4-FC512-FC3 (in our case)
            * train_data, train_labels : Data and the label for training data
            * valid_data, valid_labels : Data and the label for validation data
            * batch_size : Size of the batch
            * K : Size of the filter
            * F1 : Number of filters first convolution layer
            * F2 : Number of filters second convolution layer
            * n_feat2 : Number of features for first fully conected layer
            * drop : Dropout factor (random points set to 0)
            * learning_rate : Adam optimizer learning rate
            * reg_par : Regularization parameter
            * n_iter : Number of iteration to complete
            * print_iter : Display accuracy on validation each print_iter iteration
            * log_iter : Save accuracy results each n iteration
            * run_name : Name of the file the run is saved to
        """

        train_size = train_data.shape[0]    # data dimensionality
        d1 = train_data.shape[1]    # data dimensionality
        d2 = train_data.shape[2]    # data dimensionality
        n_img = d1//d2              # number of concatenated images
        #nc = train_labels.shape[1]  # number of classes
        nc = 3
        
        ncl1 = K*K*F1
        ncl2 = K*K*F2

        # CNN inputs variables
        xin = tf.placeholder(tf.float32,[batch_size, d1, d2]); # print('xin=',xin.get_shape())
        #y_label = tf.placeholder(tf.float32,[batch_size,nc]);  # print('y_label=',y_label.get_shape())
        
        y_label = tf.placeholder(tf.int32, (None))
        d = tf.placeholder(tf.float32);

        # CNN learning variable
        Wcl1 = tf.Variable(tf.truncated_normal([K,K,1,F1], stddev=tf.sqrt(2./tf.to_float(ncl1)) ))
        bcl1 = tf.Variable(tf.zeros([F1])) 
        Wcl2 = tf.Variable(tf.truncated_normal([K,K,F1,F2], stddev=tf.sqrt(2./tf.to_float(ncl2)) ));
        bcl2 = tf.Variable(tf.zeros([F2])); 
        Wfc = tf.Variable(tf.truncated_normal([n_img*24*24*F2, n_feat2], stddev=tf.sqrt(2./tf.to_float(24*24*F2)) ));
        bfc = tf.Variable(tf.zeros([n_feat2])); 
        Wfc2 = tf.Variable(tf.truncated_normal([n_feat2, nc], stddev=tf.sqrt(2./tf.to_float(n_feat2))));
        bfc2 = tf.Variable(tf.zeros([nc])); 

        # Layer 1 --- Convolution layer
        x = tf.reshape(xin, [-1,d1,d2,1])
        x = tf.nn.conv2d(x, Wcl1, strides=[1, 1, 1, 1], padding='SAME') + bcl1  
        x = tf.nn.relu(x)
        # print('Convolutional 1 : x=',x.get_shape())

        # Layer 2 --- Pooling
        x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print('Pooling : x=',x.get_shape())

        # Layer 3 --- Convolution layer 2
        x = tf.nn.conv2d(x, Wcl2, strides=[1, 1, 1, 1], padding='SAME') + bcl2;  
        x = tf.nn.relu(x)
        # print('Convolutional 2 : x=',x.get_shape())

        # Layer 4 --- Pooling
        x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print('Pooling : x=',x.get_shape())

        # Layer 5 --- Fully connected
        x = tf.reshape(x, [-1, n_img*24*24*F2])
        x = tf.matmul(x, Wfc) + bfc
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, d)
        # print('Fully connected : x=',x.get_shape())

        # Layer 6 --- Fully connected 2
        y = tf.matmul(x, Wfc2) + bfc2
        # print('Fully connected 2 : x=',x.get_shape())
        y_pro = tf.nn.softmax(y) # Only for output
        
        # L2 Regularization
        reg_loss = tf.nn.l2_loss(Wfc) + tf.nn.l2_loss(bfc) + tf.nn.l2_loss(Wfc2) + tf.nn.l2_loss(bfc2)
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), 1))
        # Loss
        labels = tf.to_int32(y_label)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, labels, name='xentropy')
        total_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        total_loss += reg_par*reg_loss
        # total_loss = cross_entropy + reg_par*reg_loss

        # Optimization scheme
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        # Accuracy
        #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        output_classes = tf.cast(tf.argmax(tf.nn.softmax(y),1), tf.int32)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(output_classes,labels), tf.float32))/ tf.cast(tf.shape(y)[0], tf.float32)
        
        n = train_data.shape[0]
        indices = collections.deque()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        # Keep trace of running process
        acc_train_tot = []; acc_valid_tot = []; loss_tot = []
        t_start = datetime.now()

        for i in range(n_iter+1):
            # Batch extraction (random permutation)
            if len(indices) < batch_size:
                indices.extend(np.random.permutation(n)) 
            idx = [indices.popleft() for i in range(batch_size)]
            batch_x, batch_y = train_data[idx,:,:], train_labels[idx]

            # Run CNN for variable training (feed dictionnary with batch data, label and dropout)
            _,acc_train,total_loss_o,lr_o = sess.run([train_step,accuracy,total_loss, learning_rate], 
                                                feed_dict={xin: batch_x, y_label: batch_y, d: drop})

            # Run CNN for validation set
            if (not i%log_iter) or (not i%print_iter):
                nrange = valid_data.shape[0]//batch_size
                tot_acc = 0
                y_est = np.zeros((valid_data.shape[0], 3))
                for j in range(nrange):
                    acc_valid, y_o, Wcl1_o, bcl1_o, Wcl2_o, bcl2_o, Wfc_o, bfc_o, Wfc2_o, bfc2_o = \
                    sess.run([accuracy, y_pro, Wcl1, bcl1, Wcl2, bcl2, Wfc, bfc, Wfc2, bfc2], 
                             feed_dict={xin: valid_data[j*batch_size:(j+1)*batch_size],
                                        y_label: valid_labels[j*batch_size:(j+1)*batch_size], d: 1})
                    tot_acc += acc_valid
                    y_est[j*batch_size:(j+1)*batch_size, :] = y_o

                acc_train_tot.append(acc_train); 
                acc_valid_tot.append(np.round(tot_acc/nrange,4));
                loss_tot.append(total_loss_o)
                # Print if needed
                if not i%print_iter:
                    print('\nIteration i=',i,', train accuracy=',acc_train,', loss=',total_loss_o,
                              'time spend=', datetime.now()-t_start, 'lr=', lr_o)
                    print('valid accuracy=', np.round(tot_acc/nrange,4))
                    
        # Get sparse representation
        Wfc_row, Wfc_col, Wfc_data, Wfc_shape = CNN.get_sparse_matrix(Wfc_o)
        # Save run settings
        data_run = {'K':K, 'F1':F1, 'F2':F2, 'drop':drop, 'n_feat2':n_feat2, 'learning_rate':learning_rate, 'reg_par':reg_par,
                    'n_iter':n_iter, 't_tot':(datetime.now()-t_start).total_seconds(),
                    'y_cgt':valid_labels, 'y_est':y_est, 'loss':loss_tot,
                    'acc_train_tot':acc_train_tot, 'acc_valid_tot':acc_valid_tot,
                    'Wcl1':Wcl1_o, 'bcl1':bcl1_o, 'Wcl2':Wcl2_o, 'bcl2':bcl2_o,
                    'Wfc_row':Wfc_row, 'Wfc_col':Wfc_col, 'Wfc_data':Wfc_data, 'Wfc_shape':Wfc_shape,
                    'bfc':bfc_o, 'Wfc2':Wfc2_o, 'bfc2':bfc2_o}
        DataLoader.save_run(data_run, run_name)
    
    @staticmethod
    def apply_lenet5_cnn(data, label, file_data_model, dt=200):
        """ Apply leNet5 CNN using model : CL10-MP4-CL20-MP4-FC512-FC3 n data
            * data : Data to aply model on
            * label : The label for data (ground truth)
            * file_data_model : Name of the file that contains the model data (located in STR_DIR_RUN)
        """
        # Get input learned data
        data_model = np.load(os.path.join(CNN.STR_DIR_RUN, file_data_model)).item()
        K = data_model['K']
        F1 = data_model['F1']
        F2 = data_model['F2']
        n_feat2= data_model['n_feat2']

        Wcl1_i = data_model['Wcl1']
        bcl1_i = data_model['bcl1']
        Wcl2_i = data_model['Wcl2']
        bcl2_i = data_model['bcl2']
        Wfc_i_row = data_model['Wfc_row']
        Wfc_i_col = data_model['Wfc_col']
        Wfc_i_data = data_model['Wfc_data']
        Wfc_i_shape = data_model['Wfc_shape']
        Wfc_i = scipy.sparse.coo_matrix((Wfc_i_data, (Wfc_i_row, Wfc_i_col)), shape=Wfc_i_shape).toarray()
        bfc_i = data_model['bfc']
        Wfc2_i = data_model['Wfc2']
        bfc2_i = data_model['bfc2']

        # Define settings
        d1 = data.shape[1]    # data dimensionality
        d2 = data.shape[2]    # data dimensionality
        n_img = d1//d2        # number of concatenated images
        nc = 3   # number of classes
        ncl1 = K*K*F1
        ncl2 = K*K*F2

        # CNN inputs variables
        xin = tf.placeholder(tf.float32,[None, d1, d2]); 
        y_label = tf.placeholder(tf.float32,[None]);  
        Wcl1 = tf.placeholder(tf.float32,[K,K,1,F1]) 
        bcl1 = tf.placeholder(tf.float32,[F1]) 
        Wcl2 = tf.placeholder(tf.float32,[K,K,F1,F2]) 
        bcl2 = tf.placeholder(tf.float32,[F2])
        Wfc = tf.placeholder(tf.float32,[n_img*24*24*F2, n_feat2]) 
        bfc = tf.placeholder(tf.float32,[n_feat2]) 
        Wfc2 = tf.placeholder(tf.float32,[n_feat2, nc]) 
        bfc2 = tf.placeholder(tf.float32,[nc])

        # Layer 1 --- Convolution layer
        x = tf.reshape(xin, [-1,d1,d2,1])
        x = tf.nn.conv2d(x, Wcl1, strides=[1, 1, 1, 1], padding='SAME') + bcl1  
        x = tf.nn.relu(x)
        # Layer 2 --- Pooling
        x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
         # Layer 3 --- Convolution layer 2
        x = tf.nn.conv2d(x, Wcl2, strides=[1, 1, 1, 1], padding='SAME') + bcl2;  
        x = tf.nn.relu(x)
        # Layer 4 --- Pooling
        x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Layer 5 --- Fully connected
        x = tf.reshape(x, [-1, n_img*24*24*F2])
        x = tf.matmul(x, Wfc) + bfc
        x = tf.nn.relu(x)
        # Layer 6 --- Fully connected 2
        y = tf.matmul(x, Wfc2) + bfc2

        # Accuracy
        labels = tf.to_int32(y_label)
        output_classes = tf.cast(tf.argmax(tf.nn.softmax(y),1), tf.int32)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(output_classes,labels), tf.float32))/ tf.cast(tf.shape(y)[0], tf.float32)

        n_space = data.shape[0]//dt
        d = np.linspace(0,dt*n_space,n_space+1)
        d = np.concatenate((d, [data.shape[0]])).astype(int)
        accuracy_tmp = 0
        
        for i in range(len(d)-1):
            # Run model
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)
            acc_data = sess.run([accuracy], feed_dict={xin: data[d[i]:d[i+1]], y_label: label[d[i]:d[i+1]],
                                                      Wcl1: Wcl1_i, bcl1: bcl1_i, Wcl2: Wcl2_i, bcl2: bcl2_i,
                                                      Wfc: Wfc_i, bfc: bfc_i, Wfc2: Wfc2_i, bfc2: bfc2_i })
            if not np.isnan(acc_data):
                accuracy_tmp = accuracy_tmp + acc_data[0]*(d[i+1]-d[i])
           
        accuracy_tmp = accuracy_tmp/data.shape[0]
        return np.round(accuracy_tmp, 4)
    
    @staticmethod
    def apply_lenet5_cnn_deconv(data, patients, file_data_model, id_layer_conv=2):
        """ Apply leNet5 CNN using model : CL10-MP4-CL20-MP4-FC512-FC3 n data
            * data : Data to aply model on
            * label : The label for data (ground truth)
            * file_data_model : Name of the file that contains the model data (located in STR_DIR_RUN)
        """
        # Get input learned data
        data_model = np.load(os.path.join(CNN.STR_DIR_RUN, file_data_model)).item()
        K = data_model['K']
        F1 = data_model['F1']
        F2 = data_model['F2']
        n_feat2= data_model['n_feat2']

        Wcl1_i = data_model['Wcl1']
        bcl1_i = data_model['bcl1']
        Wcl2_i = data_model['Wcl2']
        bcl2_i = data_model['bcl2']
        Wfc_i_row = data_model['Wfc_row']
        Wfc_i_col = data_model['Wfc_col']
        Wfc_i_data = data_model['Wfc_data']
        Wfc_i_shape = data_model['Wfc_shape']
        Wfc_i = scipy.sparse.coo_matrix((Wfc_i_data, (Wfc_i_row, Wfc_i_col)), shape=Wfc_i_shape).toarray()
        bfc_i = data_model['bfc']
        Wfc2_i = data_model['Wfc2']
        bfc2_i = data_model['bfc2']
        
        # Define settings
        batch_size = data.shape[0]
        d1 = data.shape[1]    # data dimensionality
        d2 = data.shape[2]    # data dimensionality
        n_img = d1//d2        # number of concatenated images
        nc = 3                # number of classes
        ncl1 = K*K*F1
        ncl2 = K*K*F2
        
        # CNN inputs variables
        xin = tf.placeholder(tf.float32,[batch_size, d1, d2]);  
        Wcl1 = tf.placeholder(tf.float32,[K,K,1,F1]) 
        bcl1 = tf.placeholder(tf.float32,[F1]) 
        
        Wcl2 = tf.placeholder(tf.float32,[K,K,F1,F2]);
        bcl2 = tf.placeholder(tf.float32,[F2]); 
        
        # Layer 1 --- Convolution layer
        x = tf.reshape(xin, [-1,d1,d2,1])
        x = tf.nn.conv2d(x, Wcl1, strides=[1, 1, 1, 1], padding='SAME') + bcl1  
        x = tf.nn.relu(x)
        
        # Layer 2 --- Pooling
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if id_layer_conv > 1:
            # Layer 3 --- Convolution layer 2
            x = tf.nn.conv2d(x, Wcl2, strides=[1, 1, 1, 1], padding='SAME') + bcl2;  
            x = tf.nn.relu(x)
            # Layer 4 --- Pooling
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Undo --- Layer 4
            x = CNN.unpool(x)
            # Undo --- Layer 3
            x = tf.nn.relu(x)
            x = tf.nn.conv2d_transpose(x-bcl2, Wcl2, output_shape=[batch_size, d1//2, d2//2, 10], 
                                       strides=[1, 1, 1, 1], padding='SAME') 
            
        # Undo --- Layer 2
        x = CNN.unpool(x)
        # Undo --- Layer 1
        x = tf.nn.relu(x)
        x = tf.nn.conv2d_transpose(x-bcl1, Wcl1, output_shape=[batch_size, d1, d2, 1], 
                                   strides=[1, 1, 1, 1], padding='SAME') 
        
        # X out performed
        x_out = x

        # Run model
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        x_out_o = sess.run([x_out], feed_dict={xin: data, 
                                                Wcl1: Wcl1_i, bcl1: bcl1_i, Wcl2: Wcl2_i, bcl2: bcl2_i })
        x_out_o = np.array(x_out_o).squeeze()
        x_rect = np.maximum(x_out_o, 0)

        id_normal = np.nonzero(patients['diagnosis'][:x_rect.shape[0]] == 1)[0]
        id_mci = np.nonzero(patients['diagnosis'][:x_rect.shape[0]] == 2)[0]
        id_ad = np.nonzero(patients['diagnosis'][:x_rect.shape[0]] == 3)[0]
        
        id_normal_max = np.argsort(np.linalg.norm(x_rect[id_normal], axis=(1,2)))[-2]
        id_mci_max = np.argmax(np.linalg.norm(x_rect[id_mci], axis=(1,2)))
        id_ad_max = np.argmax(np.linalg.norm(x_rect[id_ad], axis=(1,2)))
        
        n = 6
        fig = plt.figure(figsize=(16,4*n))
        plt.subplot(1,n,1); 
        plt.imshow(data[id_normal[id_normal_max]]);
        plt.axis('off'); plt.title('Normal [' + str(id_normal[id_normal_max]) + ']')
        plt.subplot(1,n,2);
        plt.imshow(x_rect[id_normal[id_normal_max]], cmap='seismic');
        plt.axis('off'); plt.title('Normal Activation')
        plt.subplot(1,n,3); plt.axis('off');
        plt.imshow(data[id_mci[id_mci_max]])
        plt.title('MCI [' + str(id_mci[id_mci_max]) + ']')
        plt.subplot(1,n,4); 
        plt.imshow(x_rect[id_mci[id_mci_max]], cmap='seismic')
        plt.axis('off'); plt.title('MCI Activation')
        plt.subplot(1,n,5); 
        plt.imshow(data[id_ad[id_ad_max]])
        plt.axis('off'); plt.title('AD [' + str(id_ad[id_ad_max]) + ']')
        plt.subplot(1,n,6); 
        plt.imshow(x_rect[id_ad[id_ad_max]], cmap='seismic')
        plt.axis('off'); plt.title('AD Activation')
        plt.show();
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'cnn_lenet_deconv.pdf')

    @staticmethod
    def unpool(value, name='unpool'):
        """N-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        # x will become [[x,0], [0, 0]] in any cases
        # From : https://github.com/tensorflow/tensorflow/issues/2169
        """
        with tf.name_scope(name) as scope:
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat(i, [out, tf.zeros_like(out)])
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=scope)
        return out

    @staticmethod
    def convert_to_one_hot(a,max_val=None):
        """ Convert to one one (eg. [0 0 1] for 2 or [0 1 0] for 1). a is the input vector to convert. max_val is the maximum 
            value of a, which is the number of classes
        """
        N = a.size
        data = np.ones(N,dtype=int)
        sparse_out = sparse.coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
        return np.array(sparse_out.todense())
    
    @staticmethod
    def get_sparse_matrix(matrix, zero=1e-20):
        # Evaluate 
        zero_like = np.sum(matrix <= zero)
        n_element = np.size(matrix)
        print('Matrix is', np.round(100*zero_like/n_element,2), '% sparse')
        # Cleare zers like
        matrix[np.abs(matrix) < zero] = 0
        sp = sparse.coo_matrix(matrix)
        return sp.row, sp.col, sp.data, sp.shape

    @staticmethod
    def plot_sets_preview(train_data=None, valid_data=None, n_sample = 8):
        """ Plot preview of train and validation sets defied by train_data and valid_data. n_sample is the
            number of sample to display per dataset
        """
        # Check is correct args as input
        if train_data is None or valid_data is None:
            print("ERROR - PLOT_SETS_PREVIEW - Argument is None", nbr)
            return
        
        # Get random ids to display
        id_train = np.random.randint(low=0, high=train_data.shape[0], size=n_sample)
        id_valid = np.random.randint(low=0, high=valid_data.shape[0], size=n_sample)

        figure_heigth = 4 * (train_data.shape[1]//96)
        fig = plt.figure(figsize=(16,figure_heigth))
        plt.suptitle('Visualization of Train/Validation sets', fontsize=18)
        for i in range(n_sample):
            # Plot train images
            plt.subplot(2,n_sample, i+1)
            plt.imshow(train_data[id_train[i]]); plt.axis('off')
            if(i == 0):
                plt.title('Train set', fontsize=16)
            # Plot validation images
            plt.subplot(2,n_sample, n_sample + i+1)
            plt.imshow(train_data[id_valid[i]]); plt.axis('off')
            if(i == 0):
                plt.title('Validation set', fontsize=16)
                
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_preview_set.pdf')
    
    @staticmethod
    def plot_cnn_results(file_data_model=None):
        """ Plot probality of data to belong to specific label. A polar representation is used. Each diagnosis has it's own
            vector Normal = 1, MCI = e^j2pi/6, AD = e^j2pi/6. Each point as a probality associated to each diagnosis. Therefore
            the point on the plot is difined as p1*1 + p2*e^j2pi/6 + p3*e^j4pi/6 (linear combinaison). Niter is the number of
            iteration performed during training/validation process.
        """
    
        if file_data_model is None:
            print("ERROR - PLOT_CNN_RESULTS - Argument is None", file_data_model)
            return

        data_model = np.load(os.path.join(CNN.STR_DIR_RUN, file_data_model)).item()
        y_prob = data_model['y_est']
        valid_labels = data_model['y_cgt']
        acc_train = data_model['acc_train_tot'] 
        acc_valid = data_model['acc_valid_tot'] 
        n_iter = data_model['n_iter']
        loss = data_model['loss'] 
                                        
        
        if len(valid_labels.shape) == 1:
            c_label = CNN.convert_to_one_hot(valid_labels, 3)
        else:
            c_label = valid_labels
            
        # Create base for Normal, MCI and AD
        angle = np.array([0, 2*np.pi/3, 4*np.pi/3])
        factors = np.array([np.exp(1j*angle)])
        # Linear combinaison with results
        prob_polar = y_prob.dot(factors.T).flatten()

        # Get ids of estimated classification
        ids = np.argmax(c_label, axis=1)
        ids_normal = np.nonzero(ids == 0)[0]
        ids_mci = np.nonzero(ids == 1)[0]
        ids_ad = np.nonzero(ids == 2)[0]
        
        fig = plt.figure(figsize=(16,4))
        # Polar plot
        ax = plt.subplot(1, 2, 1, projection='polar')
        plt.title('Validation - Prob. class.', fontsize=18); 
        ax.scatter(np.angle(prob_polar[ids_normal]), np.abs(prob_polar[ids_normal]), s = 50, c='b', 
                   alpha=0.6, linewidths=0, label='Normal')
        ax.scatter(np.angle(prob_polar[ids_mci]), np.abs(prob_polar[ids_mci]), s = 50, c='g', 
                   alpha=0.6, linewidths=0, label='MCI')
        ax.scatter(np.angle(prob_polar[ids_ad]), np.abs(prob_polar[ids_ad]), s = 50, c='r', 
                   alpha=0.6, linewidths=0, label='AD')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., scatterpoints=1)

        # Set limits of radius
        ax.set_rmax(1.0)
        # Set delimiters
        ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi/3))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(np.pi/6))
        
        # Turn off major tick labels
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        # Set the minor tick width to 0 so you don't see them
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

        # Set the names of your ticks, with blank spaces for the major ticks
        ax.set_xticklabels(['','Nor','','','','MCI','','','','AD'],minor=True)

        # Creat second plot to see evolution of accuracy over iteration
        ax2 = plt.subplot(1, 2, 2)
        plt.title('Accuracy/Loss over iteration', fontsize=18); plt.xlabel('Iteration'); plt.ylabel('Accuracy')
        lns1 = ax2.plot(np.linspace(0,n_iter,len(acc_train)), acc_train, '-*', label = 'Train'); # Accuracy train
        lns2 = ax2.plot(np.linspace(0,n_iter,len(acc_valid)), acc_valid, '-*', label = 'Valid'); # Accuracy validation 
        ax2.grid(); plt.ylim([0, 1]);  
        # Lodd grid 
        ax3 = ax2.twinx()
        lns3 = ax3.plot(np.linspace(0,n_iter,len(loss)), loss, '-+r', label = 'Loss'); # Accuracy validation
        plt.ylim([0, 1]); plt.ylabel('Loss')        
        
        # added these three lines
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=5)
        plt.show()
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'plot_polar_prob.pdf')

    @staticmethod
    def get_acc_run_basic(str_file_prefix='lr_', str_data_prefix='learning_rate'):
        """ Get accuracy on validation of corresponding setting. str_file_prefix will be concatenated with STR_RUN to 
            form the prefix of the filename e.g. 'test_lr_01.npy', where '01.npy' is an example of end of filename
            str_data_prefix is the name of the variable present in the file to plot
        """
        # Get all file in the run directory
        files = os.listdir(CNN.STR_DIR_RUN)
        # Take only files with the correct prefix, e.g. test_lr_*
        file_set = [file for file in files if file.startswith(CNN.STR_RUN + CNN.STR_MODEL1 + str_file_prefix)]
        
        for i, setting in enumerate(file_set):
            # Load file
            data = np.load(os.path.join(CNN.STR_DIR_RUN, setting)).item()
            if i == 0:
                # If first file to load create variables
                size_acc = np.shape(data['acc_valid_tot'])
                acc_tot = np.zeros((len(file_set), size_acc[0])) # Accuracy on validation
                acc_span = np.zeros((len(file_set), size_acc[0])) # Create x vector to plot accuracy
                setting_val = np.zeros((len(file_set),2))
            acc_tot[i, :] = data['acc_valid_tot'] # Store accuracy
            acc_span[i, :] = np.linspace(0, data['n_iter'], acc_span.shape[1]) # Generate span x values
            setting_val[i, 0] = data[str_data_prefix] # Store setting value
            if 'mul' in setting:
                setting_val[i, 1] = 3
            else:
                setting_val[i, 1] = 1
        return acc_span, acc_tot, setting_val
    
    @staticmethod
    def get_acc_run_lenet():
        """ Get accuracy on validation of corresponding setting. str_file_prefix will be concatenated with STR_RUN to 
            form the prefix of the filename e.g. 'test_lr_01.npy', where '01.npy' is an example of end of filename
            str_data_prefix is the name of the variable present in the file to plot
        """
        # Get all file in the run directory
        files = os.listdir(CNN.STR_DIR_RUN)
        # Take only files with the correct prefix, e.g. test_lr_*
        file_set = [fileN for fileN in files if CNN.STR_MODEL2 in fileN]
        
        K = []; F1 = []; F2 = []; drop = []; lr = []; reg = [];
        acc_final = []; acc_peak = []; N = []; N_iter = []

        for i, f in enumerate(file_set):
            data = np.load(os.path.join(CNN.STR_DIR_RUN, f)).item()
            K.append(data['K']); F1.append(data['F1']); F2.append(data['F2']); 
            drop.append(data['drop']); lr.append(data['learning_rate']); reg.append(data['reg_par'])
            acc_final.append(data['acc_valid_tot'])
            N_iter.append(data['n_iter'])           
            try:
                N.append(data['Wfc_shape'][0]//(F2[-1]*24*24))
            except KeyError:
                N.append(np.nan)
            
        df = pd.DataFrame({'N image':N, 'K':K, 'F1':F1, 'F2':F2, 'Dropout':drop, 'Learning rate': lr, 'Regul.': reg,
                           'Accuracy': acc_final, 'N iteration': N_iter})
        return df
        

    @staticmethod
    def subplot_run_basic(acc_span, acc_tot, setting_val, str_title='Learning rate', 
                            str_leg='F', subplotids=[1,1,1]):
        """ Subplot run - Plot accuracy values using acc_span (x axis) and setting_val (y axis). setting_val are
            the values of the parameters to plot. str_title is the title of the subplot. subplotids are the
            localization of the subplot.
        """
        # Get nmber of features to plot
        n_plot = acc_span.shape[0]
        # Create corresponding subplot
        plt.subplot(subplotids[0], subplotids[1], subplotids[2])
        # Iterate over sorted args (smallest to biggest)
        for i in np.argsort(setting_val[:,0]):
            # Change value display depending on its value
            if setting_val[i,0] >= 1:
                # Interger representation if > 1
                str_legend = '(' + str_leg + ',N)' + '=({}, {})'.format(int(setting_val[i,0]),int(setting_val[i,1])) 
            else:
                str_legend ='{:.1e}'.format(setting_val[i,0]) # Scientific representation if < 1
            plt.plot(acc_span[i,:], acc_tot[i,:], '-',label=str_legend, linewidth=2) # Plot feature
        # Set legend location and column numbers (max 3 features per column)
        plt.legend(loc = 4, ncol=int(np.ceil(n_plot/3)))
        plt.title('Variation - ' + str_title, fontsize=16)
        # Set plot limit
        plt.grid(); plt.ylim([0, 1]); 
        
    @staticmethod
    def subplot_run_lenet(df, name, subplotids=[1,1,1]):
        """ Subplot run - Plot accuracy values using acc_span (x axis) and setting_val (y axis). setting_val are
            the values of the parameters to plot. str_title is the title of the subplot. subplotids are the
            localization of the subplot.
        """
        # Get nmber of features to plot
        n_plot = len(df['Accuracy'])
        # Create corresponding subplot
        plt.subplot(subplotids[0], subplotids[1], subplotids[2])
        for i in range(n_plot):
            idx = df.index.values[i]
            val = df.loc[idx, name]
            if val < 0.1:
                strLabel = '{:.1e}'.format(val) # Scientific
            elif val < 1:
                strLabel = '{:.1f}'.format(val) # Decimal point
            else:
                strLabel = '{}'.format(int(val)) # Integer
            acc_tot = df.loc[idx,'Accuracy']
            # Check if too much data (limit ot one per 50)
            if len(acc_tot) > df.loc[idx, 'N iteration']//50:
                acc_tot = acc_tot[::(len(acc_tot)//50)]
            acc_span = np.linspace(0, df.loc[idx, 'N iteration'], len(acc_tot))
            plt.plot(acc_span, acc_tot, '-',label=strLabel, linewidth=2) # Plot feature
        # Size of the graph will depend on the minimum of iteration
        plt.xlim([0, df['N iteration'].min()]); plt.ylim([0, 1]); 
        plt.title('Accuracy over ' + name)
        plt.grid(); plt.legend(loc=4); 
        
      
    @staticmethod
    def plot_run_resume(str_model='basic'):
        
        if str_model is 'basic':
            CNN.plot_run_basic()
        else:
            CNN.plot_run_lenet()
        
    @staticmethod
    def plot_run_lenet():
        """ Plot run results for sweeping learning rate, filter number and regularization
        """
        strNames = ['Dropout','N image', 'Learning rate', 'Regul.']
        df = CNN.get_acc_run_lenet()
        df = df.dropna()
        
        fig = plt.figure(figsize=(16,12))
        for i, name in enumerate(strNames):
            df_sub = df.drop_duplicates([name])
            df_sub = df_sub.sort_values(by=name)
            CNN.subplot_run_lenet(df_sub, name, subplotids=[2,2,i+1])
            
    @staticmethod
    def plot_run_basic():
        """ Plot run results for sweeping learning rate, filter number and regularization
        """
        # Define files prefix, model variables to plot and title
        str_learning_rate = ['lr_', 'learning_rate', 'Learning rate', 'LR']
        str_f_number = ['f_', 'F', 'Filter number', 'F']
        str_regular = ['reg_param_', 'reg_par', 'Regularization', 'RP']

        fig = plt.figure(figsize=(16,12))
        # Plot learning rate
        acc_span, acc_tot, setting_val = CNN.get_acc_run_basic(str_learning_rate[0], str_learning_rate[1])
        CNN.subplot_run_basic(acc_span, acc_tot, setting_val, str_learning_rate[2], str_learning_rate[3], [2,2,1])
        # Plot F number
        acc_span, acc_tot, setting_val = CNN.get_acc_run_basic(str_f_number[0], str_f_number[1])
        CNN.subplot_run_basic(acc_span, acc_tot, setting_val, str_f_number[2], str_f_number[3], [2,2,2])
        # Plot Regularization
        acc_span, acc_tot, setting_val = CNN.get_acc_run_basic(str_regular[0], str_regular[1])
        CNN.subplot_run_basic(acc_span, acc_tot, setting_val, str_regular[2], str_regular[3], [2,2,3])
        plt.suptitle('Validation accuracy - Sweep parameters', fontsize=20)
        plt.show();
        
        # Save as PDF file if wanted
        if DataLoader.SAVE_FIGURE:
            DataLoader.save_plot(fig, 'sweep_parameters_CNN.pdf')
            
    @staticmethod
    def get_basic_resume():
        """ Displays results over all run performed for basic CNN. To be part of the run the file must contain the 
            string 'basic' in it """
        pd.set_option('display.precision', 3)
        # Get all file in the run directory
        files = os.listdir(CNN.STR_DIR_RUN)
        # Take only files with the correct prefix, e.g. test_lr_*
        file_set = [file for file in files if 'basic' in file]

        K = []; F = []; drop = []; lr = []; reg = [];
        acc_final = []; acc_peak = []; N = []; N_iter = []; time = []

        for f in file_set:
            data = np.load(os.path.join(CNN.STR_DIR_RUN, f)).item()
            K.append(data['K']); F.append(data['F']); drop.append(data['drop']); 
            lr.append(data['learning_rate']); reg.append(data['reg_par'])
            acc_final.append(data['acc_valid_tot'][-1])
            acc_peak.append(np.max(data['acc_valid_tot']))
            N_iter.append(data['n_iter'])
            
            try:
                # Get Number of images used
                N.append(data['Wfc'].shape[0]/(F[-1]*96*96))
            except KeyError:
                # Allow to integrate old run versions with no storage of matrices
                if 'mul' in f:
                    N.append(3)
                else:
                    N.append(1)
                
            try:
                time.append(data['t_tot']/3600)
            except KeyError:
                time.append(np.nan)                

        df = pd.DataFrame({'N image':N, 'K':K, 'F':F, 'Dropout':drop, 'Learning rate': lr, 'Regul.': reg,
                           'Accuracy': acc_final, 'Accuracy peak': acc_peak, 'N iteration': N_iter, 'Time h':time})
        df = df.sort_values(by='Accuracy', ascending=False)
        return df
       
    @staticmethod
    def get_lenet_resume():
        """ Displays results over all run performed for lenet like CNN. To be part of the run the file must contain the 
            string 'lenet' in it """
        pd.set_option('display.precision', 3)
        # Get all file in the run directory
        files = os.listdir(CNN.STR_DIR_RUN)
        # Take only files with the correct prefix, e.g. test_lr_*
        file_set = [file for file in files if 'lenet' in file]

        K = []; F1 = []; F2 = []; drop = []; lr = []; reg = [];
        acc_final = []; acc_peak = []; N = []; N_iter = []; time = [];

        for f in file_set:
            data = np.load(os.path.join(CNN.STR_DIR_RUN, f)).item()
            K.append(data['K']); F1.append(data['F1']); F2.append(data['F2']); 
            drop.append(data['drop']); 
            lr.append(data['learning_rate']); reg.append(data['reg_par'])
            acc_final.append(data['acc_valid_tot'][-1])
            acc_peak.append(np.max(data['acc_valid_tot']))
            N_iter.append(data['n_iter'])
            try:
                N.append(data['Wfc_shape'][0]//(24*24*F2[-1]))
            except KeyError:
                N.append(np.nan)
            try:
                time.append(data['t_tot']/3600)
            except KeyError:
                time.append(np.nan)

        df = pd.DataFrame({'N image':N, 'K':K, 'F1':F1, 'F2':F2, 'Dropout':drop, 'Learning rate': lr, 'Regul.': reg,
                           'Accuracy': acc_final, 'Accuracy peak': acc_peak, 'N iteration': N_iter,
                           'Time h':time})
        df = df.sort_values(by='Accuracy', ascending=False)
        return df