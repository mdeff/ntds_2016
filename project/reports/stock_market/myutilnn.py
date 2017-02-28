import tensorflow as tf
import time
import collections
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.sparse import coo_matrix
def convert_to_one_hot(a,max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())

def trainnn():
    X_train = np.load(os.path.join('x_train.npy'))
    X_test = np.load(os.path.join('x_test.npy'))
    Y_train1 = np.load(os.path.join('y_train1.npy'))
    Y_train2 = np.load(os.path.join('y_train2.npy'))
    Y_test1 = np.load(os.path.join('y_test1.npy'))
    Y_test2 = np.load(os.path.join('y_test2.npy'))


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train1 = Y_train1.astype('float32')
    Y_train2 = Y_train2.astype('float32')
    Y_test1 = Y_test1.astype('float32')
    Y_test2 = Y_test2.astype('float32')


    

    train_label=convert_to_one_hot(Y_train2,2)
    test_label=convert_to_one_hot(Y_test2,2)

    # See shapes of matrices
    print('Training data shape: ', X_train.shape)
    print('Training label shape: ', train_label.shape)
    print('Test data shape: ', X_test.shape)
    print('Test label shape: ', test_label.shape)

    # Define computational graph (CG)
    batch_size = 100         # batch size
    d = X_train.shape[1]  # data dimensionality
    nc=2


    # CG inputs
    xin = tf.placeholder(tf.float32,[batch_size,d]); #print('xin=',xin,xin.get_shape())
    y_label = tf.placeholder(tf.float32,[batch_size,nc]); #print('y_label=',y_label,y_label.get_shape())
    W1 = tf.get_variable("W1", shape=[d, 200],
             initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", shape=[200, 50],
             initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", shape=[50, nc],
             initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([batch_size,200],tf.float32))
    b2 = tf.Variable(tf.zeros([batch_size,50],tf.float32))
    b3 = tf.Variable(tf.zeros([batch_size,nc],tf.float32))

    # 1st Fully Connected layer
    y = tf.matmul(xin,W1) + b1

    # ReLU activation
    y = tf.nn.relu(y)

    # Dropout
    y = tf.nn.dropout(y,0.5)

    # 2nd Fully Connected layer
    y = tf.matmul(y,W2) + b2

    # ReLU activation
    y=tf.nn.relu(y)

    # 3th fully connected layer
    y=tf.matmul(y,W3) + b3

    # Softmax
    y = tf.nn.softmax(y)

    # Loss
    cross_entropy = tf.nn.l2_loss(tf.sub(y_label,y))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), 1))

    parameter=1e-8
    reg = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(b1)+tf.nn.l2_loss(b2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(b3)
    total_loss = cross_entropy+parameter*reg

    # Optimization scheme
    learning_rate = 1e-23
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run Computational Graph
    n = X_train.shape[0]
    indices = collections.deque()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(10001):
        
        # Batch extraction
        if len(indices) < batch_size:
            indices.extend(np.random.permutation(n)) 
        idx = [indices.popleft() for i in range(batch_size)]
        batch_x, batch_y = X_train[idx,:], train_label[idx]
        #print(batch_x.shape,batch_y.shape)
        
        # Run CG for variable training
        _,acc_train,total_loss_o = sess.run([train_step,accuracy,total_loss], feed_dict={xin: batch_x, y_label: batch_y})
        
        # Run CG for test set
        if not i%1000:
            print('\nIteration i=',i,', train accuracy=',acc_train,', loss=',total_loss_o)
            acc_testt=0
            for i in range(0,550//batch_size):
                acc_testt+= sess.run(accuracy, feed_dict={xin: X_test[i*batch_size:(i+1)*batch_size], y_label: test_label[i*batch_size:(i+1)*batch_size]})
            acc_test=acc_testt/(550//batch_size)
            print('test accuracy=',acc_test)
            







