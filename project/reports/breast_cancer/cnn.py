"""
    This module contains helper function for the construction of a neural network
"""

import numpy as np
import pandas as pd
import tensorflow as tf



def define_CL(kernel_size, kernel_depth, kernel_stride, kernel_num, x):
    """ Define convolutional layer
    
    Usage: [y, Wcl, bcl] = define_CL(kernel_size, kernel_depth, kernel_stride, kernel_num, x)
    
    Input variables:
        kernel_size: 2D vector with size of kernel
        kernel_depth: depth of the kernel, e.g. 3 for RGB images as input
        kernel_stride: 2D vector with the strides of the convolution
        kernel_num: number of kernels
        x: input tensor        
    
    Output variables:
        y: output tensor
        Wcl: weights
        bcl: biases
        
    """   


    
    ncl = kernel_size[0]*kernel_size[1]*kernel_depth*kernel_num
    Wcl = tf.Variable(tf.truncated_normal([kernel_size[0],kernel_size[1],kernel_depth,kernel_num],
                                          stddev=tf.sqrt(2./tf.to_float(ncl)))); # Xavier initialization
    bcl = tf.Variable(tf.zeros([kernel_num]));
    y = tf.nn.conv2d(x, Wcl, strides=[1,kernel_stride[0],kernel_stride[1], 1], padding='SAME')
    return (y,Wcl,bcl)

def define_MP(pool_size, pool_stride, x):
    """ Define max pooling layer
    
    Usage: y = define_MP(pool_size, pool_stride, x)
    
    Input variables:
        pool_size: 2D vector with receptive field size
        pool_stride: 2D vector with the strides of the pooling
        x: input tensor        
    
    Output variables:
        y: output tensor
                
    """
    
    y = tf.nn.max_pool(x, ksize=[1,pool_size[0],pool_size[1],1], strides=[1,pool_stride[0],pool_stride[1],1], padding='SAME')
    return y

def define_AP(pool_size, pool_stride, x):
    """ Define average pooling layer
    
    Usage: y = define_AP(pool_size, pool_stride, x)
    
    Input variables:
        pool_size: 2D vector with receptive field size
        pool_stride: 2D vector with the strides of the pooling
        x: input tensor        
    
    Output variables:
        y: output tensor
                
    """
    
    y = tf.nn.avg_pool(x, ksize=[1,pool_size[0],pool_size[1],1], strides=[1,pool_stride[0],pool_stride[1],1], padding='SAME')
    return y

def define_FC(ni, no, x):
    """ Define fully connected layer
    
    Usage: [y, Wfc, bfc] = define_FC(ni, no, x)
    
    Input variables:
        ni: number of inputs
        no: number of outputs
        x: input tensor        
    
    Output variables:
        y: output tensor
        Wfc: weights
        bfc: biases
        
    """  
    
    Wfc = tf.Variable(tf.truncated_normal([ni,no], stddev=tf.sqrt(6./tf.to_float(ni+no)) )); #Xavier initialization 
    bfc = tf.Variable(tf.zeros([no])); 
    y = tf.matmul(x, Wfc); 
    y += bfc; 
    return (y, Wfc, bfc)





