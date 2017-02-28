import glob
import numpy as np
import scipy as sp
from scipy import io as sio
from scipy import signal
import statsmodels.api as sm

import os
from collections import defaultdict
import time

import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

import pickle

import sklearn.neighbors, sklearn.linear_model, sklearn.ensemble, sklearn.naive_bayes 
from sklearn.feature_selection import SelectKBest


def dog_patient_plot(matDog,matPat,chan):
    sub = 'segment'
    key = list(matDog)
    segKey = [s for s in key if sub in s]
    key = list(matPat)
    segKeyPat = [s for s in key if sub in s]

    numSamples = len(matDog[segKey[0]][0][0][0][0])
    numSamplesPat = len(matPat[segKeyPat[0]][0][0][0][0])

    # Get sampling rate
    fs = matDog[segKey[0]][0][0][2][0]
    fsPat = matPat[segKeyPat[0]][0][0][2][0]

    # Defining time axis
    t = range(0,numSamples)/fs
    tPat = range(0,numSamplesPat)/fsPat

    # Plotting
    fig, axes = plt.subplots(figsize=(15, 4))

    dataPat = matPat[segKeyPat[0]][0][0][0][chan]
    plt.plot(tPat[:],dataPat[:],label='Patient')

    data = matDog[segKey[0]][0][0][0][chan]
    plt.plot(t[:],data[:]+1.5*np.max(dataPat),label='Dog')

    axes.set_title('EEG recording example - both species');
    axes.set_xlabel('Time [s]');
    axes.set_ylabel('');

    handles, labelsFig = axes.get_legend_handles_labels()
    axes.legend(handles, labelsFig,loc=3);
    return
    
def count_segments():
    currentDir = os.getcwd()
    print('In: ', currentDir, ' :')
    
    directories = os.listdir()

    interictSeg = 0
    preictSeg = 0
    testSeg = 0
    for d in directories:
        # Get in subfolder
        os.chdir(d)
        #print(d)

        # Find mat files whos filename contain "interictal", and count
        a = glob.glob("*interictal*")
        #print('Interictal: ',len(a))
        interictSeg += len(a)

        # Find mat files whos filename contain "preictal", and count
        a = glob.glob("*preictal*")
        #print('Preictal: ',len(a))
        preictSeg += len(a)

        # Find mat files whos filename contain "test", and count
        a = glob.glob("*test*")
        #print('Test: ',len(a))
        testSeg += len(a)

        # Go back to Data dir
        os.chdir(currentDir)
    print('Interictal: ',interictSeg)
    print('Preictal: ',preictSeg)
    print('Test: ',testSeg)

    data = [interictSeg, preictSeg, testSeg]
    return data

def all_features(dataDir):
    FEATURES = []
    INFO = []
    LABELS = []
    for folder in os.listdir(dataDir):        
        path = dataDir + folder + '/'
        print(folder)
        features,info,labels = extract_features(path)
        FEATURES.append(features)
        INFO += info
        LABELS.append(labels)
    return (FEATURES,INFO,LABELS)

def all_ar_param(dataDir):
    FEATURES = []
    INFO = []
    LABELS = []
    for folder in os.listdir(dataDir):        
        path = dataDir + folder + '/'
        print(folder)
        features,info,labels = extract_ar_param(path)
        FEATURES.append(features)
        INFO += info
        LABELS.append(labels)
    return (FEATURES,INFO,LABELS)

def all_fft_feats(dataDir):
    FEATURES = []
    INFO = []
    LABELS = []
    for folder in os.listdir(dataDir):        
        path = dataDir + folder + '/'
        print(folder)
        features,info,labels = extract_fft_features(path)
        FEATURES.append(features)
        INFO += info
        LABELS.append(labels)
    return (FEATURES,INFO,LABELS)


def extract_features(dataDir):
    
    FILENAME = []
    AMP = [];
    ZC = [];
    VARI = [];
    CORR = [];
    CAT = [];
    CATnum = [];
    N_EL = [];

    mats = []
    for file in os.listdir(dataDir) :
        
        # Check if file is data
        if 'mat' not in file: 
            print(file)
        else:
            #print(file)
            FILENAME.append(file)

            # Load file
            mats = sp.io.loadmat( dataDir+file ) 

            # Find key of dict indicating data
            key = list(mats)
            sub = 'segment'
            segKey = [s for s in key if sub in s]

            # Get class
            if 'preictal' in segKey[0]:
                CAT.append('preictal')
                CATnum.append(1)
            elif 'interictal' in segKey[0]:
                CAT.append('interictal')
                CATnum.append(0)
            else:
                CAT.append('test')
                CATnum.append(3)
            
            # Get number of electrodes
            nEL = mats[segKey[0]][0][0][0].shape[0]
            N_EL.append(nEL)

            # Extract amp + zc + variance + "corr time"
            amp = np.empty((16));
            amp[:] = 0#np.NAN
            zc = np.empty((16));
            zc[:] = 0#np.NAN
            vari = np.empty((16));
            vari[:] = 0#np.NAN
            corr = np.empty((16));
            corr[:] = 0

            for chan in range(0,nEL):
                data = mats[segKey[0]][0][0][0][chan]
                
                
                # Get electrode names
                nameEL = int(mats[segKey[0]][0][0][3][0][chan][0][-3:])
                
                # Get mean amplitude
                amp[nameEL-1] = np.mean(np.absolute(data))
                # Get freq. estimate
                zc[nameEL-1] = len(np.where(np.diff(np.sign(data)))[0])
                # Get variance
                vari[nameEL-1] = np.var(data)
            
                # Get "corr time" 
                    # Selecting middle of data
                h = int(round(len(data/2)))
                q = int(round(len(data/4)))
                a = data[h-q:h+q]

                data_length = len(a)
                b = np.zeros(data_length * 2)
                h = int(round(data_length/2)) 
                b[h:h+data_length] = a # This works for data_length being even
                
                #t1 = time.clock()
                # Do an array flipped convolution, which is a correlation.
                idx = 500
                c = signal.fftconvolve(b, data[::-1], mode='valid') 
                corrF = abs(c[int(round(len(c))/2):])
                corrF = corrF / max(corrF)
                #t_elaps = time.clock()-t1
                #print('Conv. : ', t_elaps)
                
                corr[nameEL-1] = sp.integrate.trapz(corrF[0:idx])
                #tsum = 0
                #idx = -1
                #while tsum < 50:
                #    idx += 1
                #    tsum += corrF[idx]
                #corr[nameEL-1] = idx  

            AMP.append(amp)
            VARI.append(vari)
            ZC.append(zc)
            CORR.append(corr)
   
    AMP = np.asarray(AMP)
    ZC = np.asarray(ZC)
    VARI = np.asarray(VARI)
    
    features = np.concatenate((AMP,ZC,VARI,CORR),axis=1)
    labels = np.asarray(CATnum)
    info = FILENAME #+ [nEL]
    return (features,info,labels)

def extract_ar_param(dataDir):
    
    FILENAME = []
    CATnum = [];
    PAR = [];
    N_EL = [];

    mats = []
    for file in os.listdir(dataDir) :
        
        # Check if file is data
        if 'mat' not in file: 
            print(file)
        else:
            #print(file)
            FILENAME.append(file)

            # Load file
            mats = sp.io.loadmat( dataDir+file ) 

            # Find key of dict indicating data
            key = list(mats)
            sub = 'segment'
            segKey = [s for s in key if sub in s]

            # Get class
            if 'preictal' in segKey[0]:
                CATnum.append(1)
            elif 'interictal' in segKey[0]:
                CATnum.append(0)
            else:
                CATnum.append(3)
            
            # Get number of electrodes
            nEL = mats[segKey[0]][0][0][0].shape[0]
            N_EL.append(nEL)

            par = np.empty((3,nEL));
            par[:] = 0
            for chan in range(0,nEL):                
                # Get data
                data = mats[segKey[0]][0][0][0][chan].astype('float64')
                h = int(round(len(data/2)))
                q = int(round(len(data/4)))
                # down-sampling
                data = data[h-q:h+q:4]
                # to pandas dataframe
                n = len(data)
                idx =  range(0, n , 1) 
                d = {'Dat' : pd.Series(data, index=idx)}
                df = pd.DataFrame(d)
                # Get ar parameters
                arma_mod20 = sm.tsa.ARMA(df, (2,0)).fit()
                par[:,chan] = np.asarray(arma_mod20.params)
                
            PAR.append(np.hstack(par))
   
    PAR = np.asarray(PAR)
    
    features = PAR
    labels = np.asarray(CATnum)
    info = FILENAME #+ [nEL]
    return (features,info,labels)

def extract_fft_features(dataDir):
    FILENAME = []
    CATnum = [];
    POWR = [];
    N_EL = [];
    fStart = [7.5, 1, 4, 13, 30]
    fEnd = [12.5, 4, 8, 30, 70]
    
    mats = []
    for file in os.listdir(dataDir) :
        
        # Check if file is data
        if 'mat' not in file: 
            print(file)
        else:
            FILENAME.append(file)
            # Load file
            mats = sp.io.loadmat( dataDir+file ) 
            # Find key of dict indicating data
            key = list(mats)
            sub = 'segment'
            segKey = [s for s in key if sub in s]
            # Get class
            if 'preictal' in segKey[0]:
                CATnum.append(1)
            elif 'interictal' in segKey[0]:
                CATnum.append(0)
            else:
                CATnum.append(3)            
            # Get number of electrodes
            nEL = mats[segKey[0]][0][0][0].shape[0]
            N_EL.append(nEL)
            # Get sampling frequency 
            Fs = mats[segKey[0]][0][0][2][0]
            
            powr = np.empty((6,16));
            powr[:] = 0
            for chan in range(0,nEL):                
                # Get data
                data = mats[segKey[0]][0][0][0][chan]
                # FFT 
                n = int(len(data))
                k = np.arange(n)
                T = n/Fs
                frq = k/T # two sides frequency range
                frq = frq[range(int(n/2))] # one side frequency range   
                Data = np.fft.fft(data)/n # fft computing and normalization
                Data = Data[range(int(n/2))]
                # Power in frequency ranges
                idx_prev = 0
                for i in range(len(fStart)):
                    idxSt = np.argmax(frq>fStart[i])
                    idxEnd = np.argmax(frq>fEnd[i])
                    powr[i,chan] = sp.integrate.trapz(abs(Data[idxSt:idxEnd])/(idxEnd-idxSt))
            POWR.append(np.hstack(powr)) 
    POWR = np.asarray(POWR)
    
    features = POWR
    labels = np.asarray(CATnum)
    info = FILENAME
    return (features,info,labels)

def load_data(dataDir):
    data = []
    CATnum = []
    for file in os.listdir( dataDir ) :

        # Check if file is data
        if 'mat' not in file: 
            print(file)
        else:
            # Get data matrix
            mats = sp.io.loadmat(dataDir+file )
            key = list(mats)
            sub = 'segment'
            segKey = [s for s in key if sub in s]
            m = mats[segKey[0]][0][0][0]
            if m.shape[0] != 16:
                print('if')
                m = np.vstack((m[0:4],np.zeros(1,m.shape[1]),m[4:]))
            data.append(m[:,::10])
            
            # Get class
            if 'preictal' in segKey[0]:
                CATnum.append(1)
            elif 'interictal' in segKey[0]:
                CATnum.append(0)
            else:
                CATnum.append(3)
        labels = np.asarray(CATnum)
    return data,labels

def get_el_names(dataDir):
    N_EL = [];
    nameEL_TOT = [];
    for folder in os.listdir(dataDir):        
        path = dataDir + folder + '/'
        for file in os.listdir(path) :
            # Check if file is data
            if 'mat' in file: 
                mats = sp.io.loadmat(path+file)
                key = list(mats)
                sub = 'segment'
                segKey = [s for s in key if sub in s]

                # Get number of electrodes
                nEL = mats[segKey[0]][0][0][0].shape[0]

                for chan in range(0,nEL):
                    data = mats[segKey[0]][0][0][0][chan]
                    # Get electrode name
                    nameEL_TOT.append(mats[segKey[0]][0][0][3][0][chan][0][-4:])
                N_EL.append(nEL)
    return N_EL, nameEL_TOT
    
def feature_distrib():        
    idxDogs = [[3391,4397],[0,1542],[2213,3391],[4397,6288],[1542,2213]]
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))

    for dog in range(5):
        with open('f_A_save', 'rb') as f:
            features = pickle.load(f)
        with open('labels_save', 'rb') as f:
            labels = pickle.load(f)
        with open('info_save', 'rb') as f:
            info = pickle.load(f)

                
        features = features[idxDogs[dog][0]:idxDogs[dog][1]]
        labels = labels[idxDogs[dog][0]:idxDogs[dog][1]]
        info = info[idxDogs[dog][0]:idxDogs[dog][1]]

        # remove electrode 4 for dog 5
        if dog == 4:
            rem = np.concatenate((range(0,3),range(4,19),range(20,35),range(36,51),range(52,64)))
            features = features[:,rem]
   
        # Splitting train set (labelled) from test set (unlabelled)
        idxTrain = np.where(labels!=3)
        idxTest = np.where(labels==3)

        X_train = np.squeeze(features[idxTrain,:])
        X_test = np.squeeze(features[idxTest,:])
        y_train = labels[idxTrain]
        y_test = labels[idxTest]

        info_train = [info[i] for i in idxTrain[0]]
        info_test = [info[i] for i in idxTest[0]]
        
        # Feature distribution
        AMP = np.mean(X_train[:,0:16],axis=1)
        ZC = np.mean(X_train[:,16:32],axis=1)
        VARI = np.mean(X_train[:,32:48],axis=1)
        CORR = np.mean(X_train[:,48:64],axis=1)
        CAT = np.transpose(y_train)

        n = len(AMP)
        idx = np.linspace(1,n,n,endpoint=True)
        idx =  range(0, n , 1) 
        d = {'Amp' : pd.Series(AMP, index=idx),'Zc' : pd.Series(ZC, index=idx),
                 'Vari' : pd.Series(VARI, index=idx),'Corr' : pd.Series(CORR, index=idx), 'Class' : pd.Series(CAT, index=idx)}
        df = pd.DataFrame(d)

        
        g = sns.boxplot('Class', 'Amp', data=df, ax=axes[dog,0])
        if dog == 0:
            g.set(title='Distribution of amplitude by class');

        g = sns.boxplot('Class', 'Zc', data=df, ax=axes[dog,1])
        if dog == 0:
            g.set(title='Distribution of zero crossings by class');

        g = sns.boxplot('Class', 'Vari', data=df, ax=axes[dog,2])
        if dog == 0:
            g.set(title='Distribution of variance by class');

        g = sns.boxplot('Class', 'Corr', data=df, ax=axes[dog,3])
        if dog == 0:
            g.set(title='Distribution of corr. time by class');
    return

def feature_band_distrib():        
    idxDogs = [[3391,4397],[0,1542],[2213,3391],[4397,6288],[1542,2213]]
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    for dog in range(5):
        with open('f_band_save', 'rb') as f:
            features = pickle.load(f)
        with open('labels_band_save', 'rb') as f:
            labels = pickle.load(f)
        with open('info_band_save', 'rb') as f:
            info = pickle.load(f)
               
        features = features[idxDogs[dog][0]:idxDogs[dog][1]]
        labels = labels[idxDogs[dog][0]:idxDogs[dog][1]]
        info = info[idxDogs[dog][0]:idxDogs[dog][1]]

        # remove electrode 4 for dog 5
        if dog == 4:
            rem = np.concatenate((range(0,15),range(16,31),range(32,47),range(48,63),range(65,79)))
            features = features[:,rem]
   
        # Splitting train set (labelled) from test set (unlabelled)
        idxTrain = np.where(labels!=3)
        idxTest = np.where(labels==3)

        X_train = np.squeeze(features[idxTrain,:])
        X_test = np.squeeze(features[idxTest,:])
        y_train = labels[idxTrain]
        y_test = labels[idxTest]

        info_train = [info[i] for i in idxTrain[0]]
        info_test = [info[i] for i in idxTest[0]]
        
        # remove electrode 4 for dog 5
        if dog == 4:      
            # Feature distribution
            B1 = np.mean(X_train[:,0:15],axis=1)
            B2 = np.mean(X_train[:,15:30],axis=1)
            B3 = np.mean(X_train[:,30:45],axis=1)
            B4 = np.mean(X_train[:,45:60],axis=1)
            B5 = np.mean(X_train[:,60:75],axis=1)
            
        else:
            # Feature distribution
            B1 = np.mean(X_train[:,0:16],axis=1)
            B2 = np.mean(X_train[:,16:32],axis=1)
            B3 = np.mean(X_train[:,32:48],axis=1)
            B4 = np.mean(X_train[:,48:64],axis=1)
            B5 = np.mean(X_train[:,64:80],axis=1)
        CAT = np.transpose(y_train)

        n = len(B1)
        idx = np.linspace(1,n,n,endpoint=True)
        idx =  range(0, n , 1) 
        d = {'Alpha' : pd.Series(B1, index=idx),'Delta' : pd.Series(B2, index=idx),
                 'Theta' : pd.Series(B3, index=idx),'Beta' : pd.Series(B4, index=idx),'Gamma' : pd.Series(B5, index=idx), 'Class' : pd.Series(CAT, index=idx)}
        df = pd.DataFrame(d)

        
        g = sns.boxplot('Class', 'Alpha', data=df, ax=axes[dog,0])
        if dog == 0:
            g.set(title='Distribution of Alpha by class');

        g = sns.boxplot('Class', 'Delta', data=df, ax=axes[dog,1])
        if dog == 0:
            g.set(title='Distribution of Delta by class');

        g = sns.boxplot('Class', 'Theta', data=df, ax=axes[dog,2])
        if dog == 0:
            g.set(title='Distribution of Theta by class');

        g = sns.boxplot('Class', 'Beta', data=df, ax=axes[dog,3])
        if dog == 0:
            g.set(title='Distribution of Beta by class');
            
        g = sns.boxplot('Class', 'Gamma', data=df, ax=axes[dog,4])
        if dog == 0:
            g.set(title='Distribution of Gamma by class');
    return

def feature_distrib_all():            
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    with open('f_A_save', 'rb') as f:
        features = pickle.load(f)
    with open('labels_save', 'rb') as f:
        labels = pickle.load(f)
    with open('info_save', 'rb') as f:
        info = pickle.load(f)
   
    # Splitting train set (labelled) from test set (unlabelled)
    idxTrain = np.where(labels!=3)
    idxTest = np.where(labels==3)
    
    X_train = np.squeeze(features[idxTrain,:])
    X_test = np.squeeze(features[idxTest,:])
    y_train = labels[idxTrain]
    y_test = labels[idxTest]
    info_train = [info[i] for i in idxTrain[0]]
    info_test = [info[i] for i in idxTest[0]]
        
    # Feature distribution
    AMP = np.mean(X_train[:,0:16],axis=1)
    ZC = np.mean(X_train[:,16:32],axis=1)
    VARI = np.mean(X_train[:,32:48],axis=1)
    CORR = np.mean(X_train[:,48:64],axis=1)
    CAT = np.transpose(y_train)
    
    n = len(AMP)
    idx = np.linspace(1,n,n,endpoint=True)
    idx =  range(0, n , 1) 
    d = {'Amp' : pd.Series(AMP, index=idx),'Zc' : pd.Series(ZC, index=idx),
             'Vari' : pd.Series(VARI, index=idx),'Corr' : pd.Series(CORR, index=idx), 'Class' : pd.Series(CAT, index=idx)}
    df = pd.DataFrame(d)
    
    pal = sns.color_palette("husl", 8)
    g = sns.boxplot('Class', 'Amp', data=df, ax=axes[0],palette=pal)
    g.set(title='Distribution of amplitude by class');

    g = sns.boxplot('Class', 'Zc', data=df, ax=axes[1],palette=pal)
    g.set(title='Distribution of zero crossings by class');

    g = sns.boxplot('Class', 'Vari', data=df, ax=axes[2],palette=pal)
    g.set(title='Distribution of variance by class');

    g = sns.boxplot('Class', 'Corr', data=df, ax=axes[3],palette=pal)
    g.set(title='Distribution of corr. time by class');
    return

def band_distrib_all():            
    with open('f_band_save', 'rb') as f:
        features = pickle.load(f)
    with open('labels_band_save', 'rb') as f:
        labels = pickle.load(f)
    with open('info_band_save', 'rb') as f:
        info = pickle.load(f)
   
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Splitting train set (labelled) from test set (unlabelled)
    idxTrain = np.where(labels!=3)
    idxTest = np.where(labels==3)

    X_train = np.squeeze(features[idxTrain,:])
    X_test = np.squeeze(features[idxTest,:])
    y_train = labels[idxTrain]
    y_test = labels[idxTest]

    # Feature distribution
    CAT = np.transpose(y_train)
    Ti = ['alpha','delta','theta','beta','gamma']
    for i in range(5):
        r1 = np.mean(X_train[:,i*16+0:i*16+16],axis=1)

        n = len(r1)
        idx = np.linspace(1,n,n,endpoint=True)
        idx =  range(0, n , 1) 
        d = {'r' : pd.Series(r1, index=idx),'Class' : pd.Series(CAT, index=idx)}
        df = pd.DataFrame(d)

        pal = sns.color_palette("husl", 2)
        g = sns.boxplot('Class', 'r', data=df, ax=axes[i],palette=pal)
        g.set(title=Ti[i]);
    return

def train_classifier_ALL(clf,names,param): 
    num_param = param.shape[1]
    idxDogs = [[3391,4397],[0,1542],[2213,3391],[4397,6288],[1542,2213]]
    

    for dog in range(5):
        with open('f_A_save', 'rb') as f:
            features = pickle.load(f)
        with open('labels_save', 'rb') as f:
            labels = pickle.load(f)
        with open('info_save', 'rb') as f:
            info = pickle.load(f)
        with open('f_band_save', 'rb') as f:
            features2 = pickle.load(f)
        features = np.hstack( (features[idxDogs[dog][0]:idxDogs[dog][1]], features2[idxDogs[dog][0]:idxDogs[dog][1]] ))
        labels = labels[idxDogs[dog][0]:idxDogs[dog][1]]
        info = info[idxDogs[dog][0]:idxDogs[dog][1]]

        # remove electrode 4 for dog 5
        if dog == 4:
            rem = np.concatenate((range(0,3),range(4,19),range(20,35),range(36,51),range(52,64),range(64,79),range(80,95),range(96,111),range(112,127),range(128,143)))
            features = features[:,rem]
           
        # Splitting train set (labelled) from test set (unlabelled)
        idxTrain = np.where(labels!=3)
        idxTest = np.where(labels==3)

        X_train = np.squeeze(features[idxTrain,:])
        X_test = np.squeeze(features[idxTest,:])
        y_train = labels[idxTrain]
        y_test = labels[idxTest]

        info_train = [info[i] for i in idxTrain[0]]
        info_test = [info[i] for i in idxTest[0]]
        
        
        # Separate training set in folds for cross-validation
        num_folds = 2
        X_train_folds = np.array_split(X_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)
        
        # Cross-validation
        accuracyTRAIN = np.zeros([len(clf)])
        accuracyVAL = np.zeros([len(clf)])


        acc_foldsTRAIN = np.zeros([num_folds, len(clf),num_param-1])
        acc_foldsVAL = np.zeros([num_folds,len(clf),num_param-1])

        for fold_idx in range(num_folds):

            mdl_idx = 0
            for model in clf:

                # Extract train dataset for the current fold
                fold_x_train = np.concatenate([X_train_folds[i] for i in range(num_folds) if i!=fold_idx])       
                fold_y_train = np.concatenate([y_train_folds[i] for i in range(num_folds) if i!=fold_idx])   

                # validation dataset for the current fold
                fold_x_val  = X_train_folds[fold_idx]
                fold_y_val  = y_train_folds[fold_idx]

                # Normalize features (mean and sd of training fold only)
                m = np.mean(fold_x_train,axis=0)
                sd = np.std(fold_x_train,axis=0)       
                fold_x_train =  (fold_x_train-m)/sd
                fold_x_val = (fold_x_val-m)/sd
                
                # Feature selection
                selection = SelectKBest(k=20)
                KfeatIdx = selection.fit(fold_x_train, fold_y_train).get_support(indices=True)
                fold_x_train = fold_x_train[:,KfeatIdx]
                fold_x_val = fold_x_val[:,KfeatIdx]

                for param_idx in range(num_param-1):
                    p = param[mdl_idx,param_idx]    
                    model.C = p
                    model.n_neighbors = int(p)
                    model.alpha = p
                    model.n_estimators = int(p)

                    # Run current model for the current fold
                    model.fit(fold_x_train, fold_y_train)
                    train_pred = model.predict(fold_x_train)

                    accuracytrain = sklearn.metrics.roc_auc_score(fold_y_train, train_pred)

                    val_pred = model.predict(fold_x_val)
                    accuracy = sklearn.metrics.roc_auc_score(fold_y_val, val_pred)

                    # Store accuracy values
                    acc_foldsTRAIN[fold_idx,mdl_idx,param_idx] = accuracytrain
                    acc_foldsVAL[fold_idx,mdl_idx,param_idx] = accuracy

                mdl_idx += 1

        accuracyTRAIN = np.mean(acc_foldsTRAIN,axis=0)
        accuracyVAL = np.mean(acc_foldsVAL, axis=0)

        stdTRAIN = np.std(acc_foldsTRAIN,axis=0)
        stdVAL = np.std(acc_foldsVAL, axis=0)
        
        # Plotting
        fig, axes = plt.subplots(figsize=(15, 5))

        colors = ['b','g','r','c','m','k']
        for i in range(len(clf)):
            #Bidx = np.argmax(accuracyVAL[i])
            #print('Best Val : ', accuracyVAL[i,Bidx], 'SD: ', stdVAL[i,Bidx])
            plt.errorbar(range(num_param-1),accuracyVAL[i,:],stdVAL[i,:],color=colors[i],label=names[i]+' VAL')
            plt.errorbar(range(num_param-1),accuracyTRAIN[i,:],stdTRAIN[i,:],color=colors[i], linestyle='dashed',label=names[i]+' TRAIN')
        axes.set_xlabel('param idx');
        axes.set_ylabel('AUC');
        axes.set_title(('Dog '+ str(dog+1)))
        axes.axis([0, num_param-1, 0, 1]);


        handles, labelsFig = axes.get_legend_handles_labels()
        axes.legend(handles, labelsFig,loc=3);
        
    return


def train_classifier(model,names,param): 
    num_param = param.shape[0]
    idxDogs = [[3391,4397],[0,1542],[2213,3391],[4397,6288],[1542,2213]]
    K = [10,20,30,40,50]

    for dog in range(5):
        with open('f_A_save', 'rb') as f:
            features = pickle.load(f)
        with open('labels_save', 'rb') as f:
            labels = pickle.load(f)
        with open('info_save', 'rb') as f:
            info = pickle.load(f)
        with open('f_band_save', 'rb') as f:
            features2 = pickle.load(f)
        features = np.hstack( (features[idxDogs[dog][0]:idxDogs[dog][1]], features2[idxDogs[dog][0]:idxDogs[dog][1]] ))
        labels = labels[idxDogs[dog][0]:idxDogs[dog][1]]
        info = info[idxDogs[dog][0]:idxDogs[dog][1]]

        # remove electrode 4 for dog 5
        if dog == 4:
            rem = np.concatenate((range(0,3),range(4,19),range(20,35),range(36,51),range(52,64),range(64,79),range(80,95),range(96,111),range(112,127),range(128,143)))
            features = features[:,rem]
           
        # Splitting train set (labelled) from test set (unlabelled)
        idxTrain = np.where(labels!=3)
        idxTest = np.where(labels==3)

        X_train = np.squeeze(features[idxTrain,:])
        X_test = np.squeeze(features[idxTest,:])
        y_train = labels[idxTrain]
        y_test = labels[idxTest]

        info_train = [info[i] for i in idxTrain[0]]
        info_test = [info[i] for i in idxTest[0]]
        
        
        # Separate training set in folds for cross-validation
        num_folds = 2
        X_train_folds = np.array_split(X_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)
        
        # Cross-validation
        accuracyTRAIN = np.zeros([len(K)])
        accuracyVAL = np.zeros([len(K)])
        
        acc_foldsTRAIN = np.zeros([num_folds, len(K),num_param-1])
        acc_foldsVAL = np.zeros([num_folds,len(K),num_param-1])

        for fold_idx in range(num_folds):

            # Extract train dataset for the current fold
            fold_x_train = np.concatenate([X_train_folds[i] for i in range(num_folds) if i!=fold_idx])       
            fold_y_train = np.concatenate([y_train_folds[i] for i in range(num_folds) if i!=fold_idx])   

            # validation dataset for the current fold
            fold_x_val  = X_train_folds[fold_idx]
            fold_y_val  = y_train_folds[fold_idx]

            # Normalize features (mean and sd of training fold only)
            m = np.mean(fold_x_train,axis=0)
            sd = np.std(fold_x_train,axis=0)       
            fold_x_train =  (fold_x_train-m)/sd
            fold_x_val = (fold_x_val-m)/sd

            # Feature selection
            for i in range(len(K)):
                selection = SelectKBest(k=K[i])
                KfeatIdx = selection.fit(fold_x_train, fold_y_train).get_support(indices=True)
                fold_x_trainK = fold_x_train[:,KfeatIdx]
                fold_x_valK = fold_x_val[:,KfeatIdx]

                for param_idx in range(num_param-1):
                    p = param[param_idx]    
                    model.C = p
                    model.n_neighbors = int(p)
                    model.alpha = p
                    model.n_estimators = int(p)

                    # Run current model for the current fold
                    model.fit(fold_x_trainK, fold_y_train)
                    train_pred = model.predict(fold_x_trainK)

                    accuracytrain = sklearn.metrics.roc_auc_score(fold_y_train, train_pred)

                    val_pred = model.predict(fold_x_valK)
                    accuracy = sklearn.metrics.roc_auc_score(fold_y_val, val_pred)

                    # Store accuracy values
                    acc_foldsTRAIN[fold_idx,i,param_idx] = accuracytrain
                    acc_foldsVAL[fold_idx,i,param_idx] = accuracy

        accuracyTRAIN = np.mean(acc_foldsTRAIN,axis=0)
        accuracyVAL = np.mean(acc_foldsVAL, axis=0)

        stdTRAIN = np.std(acc_foldsTRAIN,axis=0)
        stdVAL = np.std(acc_foldsVAL, axis=0)
        
        # Plotting
        fig, axes = plt.subplots(figsize=(15, 5))

        colors = ['b','g','r','c','m','k']
        for i in range(len(K)):
            #Bidx = np.argmax(accuracyVAL[i])
            #print('Best Val : ', accuracyVAL[i,Bidx], 'SD: ', stdVAL[i,Bidx])
            plt.errorbar(range(num_param-1),accuracyVAL[i,:],stdVAL[i,:],color=colors[i],label='VAL '+str(K[i]))
            plt.errorbar(range(num_param-1),accuracyTRAIN[i,:],stdTRAIN[i,:],color=colors[i], linestyle='dashed',label='TRAIN '+str(K[i]))
                         
        axes.set_xlabel('param idx');
        axes.set_ylabel('AUC');
        axes.set_title(('Dog '+ str(dog+1)))
        axes.axis([0, num_param-1, 0, 1]);


        handles, labelsFig = axes.get_legend_handles_labels()
        axes.legend(handles, labelsFig,loc=3);
        
    return



def test_classifierOld(model,param): 
    idxDogs = [[3391,4397],[0,1542],[2213,3391],[4397,6288],[1542,2213]]
    all_test_pred = []
    all_info = []
    all_accuracytrain = []
    for dog in range(5):
        with open('f_A_save', 'rb') as f:
            features = pickle.load(f)
        with open('labels_save', 'rb') as f:
            labels = pickle.load(f)
        with open('info_save', 'rb') as f:
            info = pickle.load(f)
        with open('f_fft_save', 'rb') as f:
            features2 = pickle.load(f)
        features = np.hstack( (features[idxDogs[dog][0]:idxDogs[dog][1]], features2[idxDogs[dog][0]:idxDogs[dog][1]] ))
        labels = labels[idxDogs[dog][0]:idxDogs[dog][1]]
        info = info[idxDogs[dog][0]:idxDogs[dog][1]]
        
        # remove electrode 4 for dog 5
        if dog == 4:
            rem = np.concatenate((range(0,3),range(4,19),range(20,35),range(36,51),range(52,154)))
            features = features[:,rem]
            
        # Splitting train set (labelled) from test set (unlabelled)
        idxTrain = np.where(labels!=3)
        idxTest = np.where(labels==3)

        X_train = np.squeeze(features[idxTrain,:])
        X_test = np.squeeze(features[idxTest,:])
        y_train = labels[idxTrain]
        y_test = labels[idxTest]

        info_train = [info[i] for i in idxTrain[0]]
        info_test = [info[i] for i in idxTest[0]]

        all_info.append(info_test)
        
        # Normalize features (mean and sd of training fold only)
        m = np.mean(X_train,axis=0)
        sd = np.std(X_train,axis=0)       
        X_train =  (X_train-m)/sd
        X_test = (X_test-m)/sd
        
        # Feature selection
        selection = SelectKBest(k=50)
        KfeatIdx = selection.fit(X_train, y_train).get_support(indices=True)
        #X_train = X_train[:,KfeatIdx]
        #X_test = X_test[:,KfeatIdx]

                
        # Setting parameter
        model.C = param
        #model.n_neighbors = int(param)
        #model.alpha = param

        # Train model
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)

        accuracytrain = sklearn.metrics.roc_auc_score(y_train, train_pred)
        
        all_accuracytrain.append(accuracytrain)
        all_test_pred.append(model.predict(X_test))

    return (all_test_pred, all_info, all_accuracytrain)

def test_classifier(model,param): 
    idxDogs = [[3391,4397],[0,1542],[2213,3391],[4397,6288],[1542,2213]]
    all_test_pred = []
    all_info = []
    all_accuracytrain = []
    for dog in range(5):
        with open('f_band_save', 'rb') as f:
            features = pickle.load(f)
        with open('labels_band_save', 'rb') as f:
            labels = pickle.load(f)
        with open('info_band_save', 'rb') as f:
            info = pickle.load(f)
        features = features[idxDogs[dog][0]:idxDogs[dog][1]]
        labels = labels[idxDogs[dog][0]:idxDogs[dog][1]]
        info = info[idxDogs[dog][0]:idxDogs[dog][1]]
        
        # remove electrode 4 for dog 5
        if dog == 4:
            rem = np.concatenate((range(0,15),range(16,31),range(32,47),range(48,63),range(65,79)))
            features = features[:,rem]
            
        # Splitting train set (labelled) from test set (unlabelled)
        idxTrain = np.where(labels!=3)
        idxTest = np.where(labels==3)

        X_train = np.squeeze(features[idxTrain,:])
        X_test = np.squeeze(features[idxTest,:])
        y_train = labels[idxTrain]
        y_test = labels[idxTest]

        info_train = [info[i] for i in idxTrain[0]]
        info_test = [info[i] for i in idxTest[0]]

        all_info.append(info_test)
        
        # Normalize features (mean and sd of training fold only)
        m = np.mean(X_train,axis=0)
        sd = np.std(X_train,axis=0)       
        X_train =  (X_train-m)/sd
        X_test = (X_test-m)/sd
        
        # Feature selection
        selection = SelectKBest(k=20)
        KfeatIdx = selection.fit(X_train, y_train).get_support(indices=True)
        X_train = X_train[:,KfeatIdx]
        X_test = X_test[:,KfeatIdx]

                
        # Setting parameter
        model.C = param
        #model.n_neighbors = int(param)
        #model.alpha = param

        # Train model
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)

        accuracytrain = sklearn.metrics.roc_auc_score(y_train, train_pred)
        
        all_accuracytrain.append(accuracytrain)
        all_test_pred.append(model.predict(X_test))

    return (all_test_pred, all_info, all_accuracytrain)