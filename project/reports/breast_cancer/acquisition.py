import numpy as np
import pandas as pd
import tensorflow as tf
import os
import skimage.io as imageio
import skimage.color as color
import skimage.transform  as trf
import collections
imageio.use_plugin('matplotlib')

from ipywidgets import FloatProgress
from IPython.display import display
import time

import sqlalchemy as sql
import sklearn.feature_extraction.image as skfeim





def read_image(link, size):
    """ Read image on link and convert it to given size
    
    Usage:
        image = readImage(link, size)
    
    Input variables: 
        link: path to image
        size: output size of image
        
    Output variables:
        image: read and resized image
    """
    
    image = imageio.imread(link)
    image = trf.resize(image, size)
    
    return image

def read_image_list(path, size):
    """ Read all images in folder, convert them to given size and put in dataframe
    
    Usage:
        df = readImageList(path, size)
    
    Input variables: 
        path: path to images
        size: output size of image
        
    Output variables:
        df: dataframe with images and metadata
    
    """
    
    num_files = len(os.listdir(path))
    
    pb = FloatProgress(min=0, max=num_files)
    display(pb)
    
    df = pd.DataFrame(columns = ('filename','type','subtype','patient','patient_slice','image'))
    for f in os.listdir(path):
        pb.value+=1
        fparts = f.split('-')
        parts = fparts[0].split('_')
        im = read_image(os.path.join(path,f),size)
        df.loc[len(df)] = (f,parts[1],parts[2],fparts[2],fparts[4].split('.')[0],im)
    return df


def convert_to_patches(df, patch_size, patch_number, image_size):
    """ Convert images to patches and add them to dataframe
        """
    
    
    n = df.shape[0]
    
    pb = FloatProgress(min=0, max=n)
    display(pb)
    
    df['patches'] = pd.Series(np.empty(n), index=df.index).astype(object)
    df['mean'] = pd.Series(np.empty(n), index=df.index).astype(object)
    
    for ii in range(0,df.shape[0]):
        pb.value+=1
        im = df.image[ii]
        df.set_value(ii,'mean',np.array(np.mean(im,axis=(0,1))))
        df.set_value(ii,'patches',skfeim.extract_patches_2d(im, patch_size, max_patches = patch_number))
   

    return df

def add_folds(df, fold1, fold2, fold3, fold4, fold5):
    """
    add folds to dataframe
    """

    n = df.shape[0]
    fold1_series = pd.Series(np.empty(n), index=df.index)
    fold2_series = pd.Series(np.empty(n), index=df.index)
    fold3_series = pd.Series(np.empty(n), index=df.index)
    fold4_series = pd.Series(np.empty(n), index=df.index)
    fold5_series = pd.Series(np.empty(n), index=df.index)
    
    pb = FloatProgress(min=0, max=n)
    display(pb)
    
    for ii in range(0,n):
        pb.value+=1
        fold1_series[ii] = fold1[fold1[:][0]==df.filename[ii]][3].iloc[0]
        fold2_series[ii] = fold2[fold2[:][0]==df.filename[ii]][3].iloc[0]
        fold3_series[ii] = fold3[fold3[:][0]==df.filename[ii]][3].iloc[0]
        fold4_series[ii] = fold4[fold4[:][0]==df.filename[ii]][3].iloc[0]
        fold5_series[ii] = fold5[fold5[:][0]==df.filename[ii]][3].iloc[0]
        
    df['fold1'] = fold1_series
    df['fold2'] = fold2_series
    df['fold3'] = fold3_series
    df['fold4'] = fold4_series
    df['fold5'] = fold5_series
  

    return df
    
    
def generate_databases(folder, image_size, patch_size, patch_number):
    """ generate the databases
    """
    
    fold1 = pd.read_csv(os.path.join(folder,'dsfold1.txt'),delimiter='|',header=None)
    fold2 = pd.read_csv(os.path.join(folder,'dsfold2.txt'),delimiter='|',header=None)
    fold3 = pd.read_csv(os.path.join(folder,'dsfold3.txt'),delimiter='|',header=None)
    fold4 = pd.read_csv(os.path.join(folder,'dsfold4.txt'),delimiter='|',header=None)
    fold5 = pd.read_csv(os.path.join(folder,'dsfold5.txt'),delimiter='|',header=None)

    
    magnifications = ['40X','100X','200X','400X']
    
    for m in magnifications:
        print('Magnification ',m)
        
        print('\t reading train...')
        df1 = read_image_list(os.path.join(folder,'fold1','train',m), image_size)
        
        print('\t reading test...')
        df2 = read_image_list(os.path.join(folder,'fold1','test',m), image_size)
        
        print('\t appending both...')        
        df1 = df1.append(df2, ignore_index = True)
       
        print('\t adding folds...')
        df1 = add_folds(df1, fold1, fold2, fold3, fold4, fold5)
    
        print('\t creating patches...')
        df1 = convert_to_patches(df1, patch_size, patch_number, image_size)
        
        print('\t storing in sqlite')
        df1.to_sql('images','sqlite:///'+str(image_size[0])+'_'+str(image_size[1])+'_'+m+'.sqlite',if_exists='replace')

def remove_patient_from_database(patient, image_size, magnification):
    """ remove a patient from the database and create the cleant database
    """
    
    df = pd.read_sql('images','sqlite:///'+str(image_size[0])+'_'+str(image_size[1])+'_'+magnification+'.sqlite')
    print('Original size: ',df.shape[0])
    df.drop(df.index[df.patient==patient],inplace=True)
    df.reindex()
    print('New size: ',df.shape[0])
    df.to_sql('images','sqlite:///'+str(image_size[0])+'_'+str(image_size[1])+'_'+magnification+
              '_clean.sqlite',if_exists='replace')



    
    
    
    
