"""
This module contains all kinds of functions needed for execution of the notebook
"""

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from skimage.feature import local_binary_pattern
import sqlalchemy as sql
import random
import scipy






def convertToNDArray(df, patch_number, patch_size):
    """ Get data for network from dataframe
    
    Usage: [data, labels_bin, labels_all] = convertToNDArray(df, patch_number, patch_size)
    
    Input variables:
        df: dataframe from sqlite database
        patch_number: number of patches per image (size = (1,))
        patch_size: 2D size of patch (size = (2,))
    
    Output variables:
        data: input data for neural network (size = (total number of patches, patch size 0, patch size 1, 3))
        labels_bin: binary labels for all patches (size = (total number of patches,))
        labels_all: class labels for all patches (size = (total number of patches,))
        
    """
    
    ## CONVERT BINARY CLASSES TO NUMERIC    
    df['type'] = df['type'].astype('category')
    df['type'].cat.categories = [0, 1]
    df['type'] = df['type'].astype(np.int)

    ## CONVERT SUBCLASSES TO NUMERIC
    df['subtype'] = df['subtype'].astype('category')
    mapping = {'A':0,'F':1,'TA':2,'PT':3,'DC':4,'LC':5,'MC':6,'PC':7}
    numeric = []
    for ii,tt in enumerate(list(df['subtype'].cat.categories)):
        numeric.append(mapping[tt])
    df['subtype'].cat.categories = numeric
    df['subtype'] = df['subtype'].astype(np.int)
    
    ## CONVERT PATIENTS TO NUMERIC
    df['patient'] = df['patient'].astype('category')
    df['patient'].cat.categories = range(df.patient.unique().shape[0])
    df['patient'] = df['patient'].astype(np.int)
    
    ## INITIALIZE VARIABLES
    n = df.shape[0]
    data = np.ndarray(shape=(n*patch_number,patch_size[0],patch_size[1],3))
    labels_bin = np.ndarray(shape=(n*patch_number,))
    labels_all = np.ndarray(shape=(n*patch_number,))
    patients = np.ndarray(shape=(n*patch_number,))
    
    ## FILL OUTPUT
    for ii in range(0,df.shape[0]):
        all_patches = np.reshape(np.fromstring(df.patches[ii]),(patch_number,patch_size[0],patch_size[1],3)) #from string to data
        
        data[ii*patch_number:(ii+1)*patch_number]=all_patches.astype(np.float32)-np.fromstring(df['mean'][ii])
        labels_bin[ii*patch_number:(ii+1)*patch_number] = df.type[ii].astype(np.float32)
        labels_all[ii*patch_number:(ii+1)*patch_number] = df.subtype[ii].astype(np.float32)
        patients[ii*patch_number:(ii+1)*patch_number] = df.patient[ii].astype(np.float32)

        
        
    return (data, labels_bin, labels_all, patients)

def convert_to_one_hot(a,max_val=None):
    """ Convert to one-hot
    
    Usage: [one_hot] = convertToNDArray(a, [max_val])
    
    Input variables:
        a: input array with integers (size = (N,))
        max_val: size of one-hot representation
    
    Output variables:
        one_hot: one-hot representation of a (size = (N,max_val))
        
    """
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())


def compute_accuracy_image(y_patched, ground_truth_patched, patch_number, nc):

    """ compute the performance measures on the image scale
    """
    
    n = y_patched.shape[0]
    sums = np.empty(shape=(np.int(n/patch_number),nc))

    predictions = np.empty(shape=(np.int(n/patch_number)))
    for ii in np.arange(0,n,patch_number):
        sums[np.int(ii/patch_number)]=np.sum(y_patched[ii:ii+patch_number],axis=0)
        predictions[np.int(ii/patch_number)] = np.argmax(sums[np.int(ii/patch_number)])
    
    acc = np.sum(predictions==ground_truth_patched[0::patch_number])/predictions.shape[0]
    
    split = nc/2
    
    sens = np.sum((predictions>=split)*(ground_truth_patched[0::patch_number]>=split))/np.sum(ground_truth_patched[0::patch_number]>=split)
    spec = np.sum((predictions<split)*(ground_truth_patched[0::patch_number]<split))/np.sum(ground_truth_patched[0::patch_number]<split)
    
    return (acc, sens, spec, predictions)

def compute_accuracy_patient(y_patched, ground_truth_patched, patients_patched, nc):
    
    """ compute the performance measures on the patient scale
    """
    
    n = y_patched.shape[0]
    N = len(np.unique(patients_patched))
    sums = np.empty(shape=(N,nc))
    predictions = np.empty(shape=(N))
    ground_truth = np.empty(shape=(N))
    
    for ii,pat in enumerate(np.unique(patients_patched)):
        sums[ii]=np.sum(y_patched[patients_patched==pat],axis=0)
        predictions[ii] = np.argmax(sums[ii])
        ground_truth[ii] = ground_truth_patched[patients_patched==pat][0]
    
    acc = np.sum(predictions==ground_truth)/predictions.shape[0]
    
    split = nc/2
    sens = np.sum((predictions>=split)*(ground_truth>=split))/np.sum(ground_truth>=split)
    spec = np.sum((predictions<split)*(ground_truth<split))/np.sum(ground_truth<split)
    
    return (acc, sens, spec)

def define_plots():
    
    """ define the plots for during the training
    """
    
    fig = plt.figure(figsize = (12,18))

    gs = GridSpec(6, 4, hspace=.5, wspace=.5)#, wspace=0.0, hspace=0.0)

    ax_acc_train = fig.add_subplot(gs[1, :-1])
    ax_sens_train = fig.add_subplot(gs[1, -1:])
    ax_acc_p_test = fig.add_subplot(gs[2, :-1])
    ax_sens_p_test = fig.add_subplot(gs[2, -1:])
    ax_acc_i_test = fig.add_subplot(gs[3, :-1])
    ax_sens_i_test = fig.add_subplot(gs[3, -1:])
    ax_acc_pat_test = fig.add_subplot(gs[4, :-1])
    ax_sens_pat_test = fig.add_subplot(gs[4, -1:])
    ax_loss = fig.add_subplot(gs[0, :])
    ax_pred = fig.add_subplot(gs[5, :])

   
    return (fig, ax_acc_train, ax_sens_train, ax_loss, ax_acc_p_test, ax_sens_p_test, ax_acc_i_test, ax_sens_i_test, ax_acc_pat_test, ax_sens_pat_test, ax_pred)

def update_accuracy_plot(fig, ax, acc, title='', iteration_factor=1):
    """ update an accuracy plot """
    ax.clear()
    ax.set_ylim((0,1))
    ax.plot(iteration_factor*np.arange(0,len(acc)),acc)
    ax.set_xlabel('iteration')
    ax.set_ylabel('accuracy')
    ax.set_title(title, size=16)

def update_sensitivity_plot(fig, ax, sens, spec, title=''):
    """ update a ROC plot """
    mem = 10000
    sens = sens[max(0,len(sens)-mem):]
    spec = spec[max(0,len(spec)-mem):]
    ax.clear()
    ax.set_ylim((-0.05,1.05))
    ax.set_xlim((-0.05,1.05))
    ax.set_xlabel('1-specificity')
    ax.set_ylabel('sensitivity')
    ax.scatter(1-np.array(spec), sens, c = cm.Reds(np.arange(0,1.0,1.0/float(len(sens)))), linewidths = 0.2)
    ax.plot([0,1],[0,1])
    ax.set_title(title, size=16)

def update_loss_plot(fig, ax, losses, title='Loss value'):
    """ update a loss plot """
    ax.clear()
    ax.set_ylim((0,2*max(losses)))
    ax.plot(losses, label='loss')
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(title, size=16)

def update_prediction_plot(fig, ax, pred, truth,title=''):
    """ update a prediction plot """
    ax.clear()
    ax.set_ylim((-1,max(truth)+4))
    ax.plot(truth,'g',lw=4, label='truth')

    ax.plot(pred, '.b', label='prediction')
    ax.set_xlabel('index')
    ax.set_ylabel('prediction')
    ax.set_xlim((0,len(truth)-1))
    ax.set_title(title, size=16)
    if np.max(truth)==7:
        ax.set_yticks(range(0,8))
        ax.set_yticklabels(['A','F','TA','PT','DC','LC','MC','PC'])


    ax.legend(ncol=2)
    
    
def make_example_plot(df, image_size, magnification, n):
    """ make figure with example images for the different tumor subtypes
    """
    SUBTYPES = [('A','Adenosis'),
            ('F','Fibroadenoma'),
            ('TA','Tubular Adenoma'),
            ('PT','Phyllodes Tumor'),
            ('DC', 'Ductal Carcinoma'),
            ('LC', 'Lobular Carcinoma'),
            ('MC', 'Mucinous Carcinoma'),
            ('PC', 'Papillary Carcinoma')
           ]
    fig, axes = plt.subplots(n,8,figsize=(12,n+2))
    fig.suptitle(magnification, size=16);
    for ii, tt in enumerate(SUBTYPES):
        d = df.loc[df.subtype == tt[0]]
        idx =  random.sample(range(0,d.shape[0]), n)
        for jj in range(0,n):
            im = np.reshape(np.fromstring(d.image.iloc[idx[jj]]),(image_size[0],image_size[1],3))
            axes[jj,ii].imshow(im)
            axes[jj,ii].axis('off')
            if jj == 0:
                axes[jj,ii].set_title(tt[1], size=10)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=.95, wspace=0.05, hspace=0.001) 
    
def make_example_patches_plot(df, image_size, patch_size, patch_number):
    """ make figure with example patching """
    fig = plt.figure(figsize = (12,6))

    n = int(np.ceil(np.sqrt(patch_number)))
    gs = GridSpec(int(np.ceil(patch_number/n)), 2*n, hspace=.5, wspace=.5)#, wspace=0.0, hspace=0.0)

    idx = random.randint(0,df.shape[0]-1)
    
    ax1 = fig.add_subplot(gs[:, :-n])
    im = np.reshape(np.fromstring(df.image.iloc[idx]),(image_size[0],image_size[1],3))
    ax1.imshow(im)
    ax1.axis('off')
    
    patches = np.reshape(np.fromstring(df.patches.iloc[idx]),(patch_number,patch_size[0],patch_size[1],3))
    
    for ii in range(0,n):
        for jj in range(0,n):
            if ii*n+jj < patch_number:
                ax = fig.add_subplot(gs[ii,(-n+jj)])
                ax.imshow(patches[ii*n+jj])
                ax.axis('off')
                
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.001) 

def make_statistics_plots(IMAGE_SIZE, MAGNIFICATIONS, TYPES_SHORT, TYPES_LONG, SUBTYPES_SHORT, SUBTYPES_LONG, clean=0):
    """ make the pie charts for the dataset statistics """
    for m in MAGNIFICATIONS:
        if clean==0:
            engine = sql.create_engine('sqlite:///'+str(IMAGE_SIZE[0])+'_'+str(IMAGE_SIZE[1])+'_'+m+'.sqlite',echo=False)
        else:
            engine = sql.create_engine('sqlite:///'+str(IMAGE_SIZE[0])+'_'
                                       +str(IMAGE_SIZE[1])+'_'+m+'_clean.sqlite',echo=False)

        df = pd.read_sql('images',engine, columns=['type','subtype','patient'])

        fig, (ax_pie1, ax_pie2) = plt.subplots(1,2,figsize=(12,4))
        fig.suptitle(m+' statistics',size=16)

        # pie plot types
        patient_counts_types = np.zeros(len(TYPES_SHORT))
        for ii,t in enumerate(TYPES_SHORT):
            patient_counts_types[ii]=df.groupby(['type','patient']).size().loc[t].size
        labels = ["%s\n%d patients" % t for t in zip(TYPES_LONG, patient_counts_types)]
        sizes = df.groupby('type').size().loc[TYPES_SHORT]
        colors = ['lightgreen', 'red']
        ax_pie1.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
        ax_pie1.axis('equal')

        # pie plot subtypes
        patient_counts_subtypes = np.zeros(len(SUBTYPES_SHORT))
        for ii,t in enumerate(SUBTYPES_SHORT):
            patient_counts_subtypes[ii]=df.groupby(['subtype','patient']).size().loc[t].size
        labels = ["%s\n%d patients" % t for t in zip(SUBTYPES_LONG, patient_counts_subtypes)]
        sizes = df.groupby('subtype').size().loc[SUBTYPES_SHORT]
        colors = ['lightgreen','forestgreen','darkgreen','limegreen', 'red', 'firebrick', 'orangered', 'tomato']
        ax_pie2.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=90)
        ax_pie2.axis('equal')

def make_exploration_figures(IMAGE_SIZE, MAGNIFICATION, TYPES_SHORT, TYPES_LONG, SUBTYPES_SHORT, SUBTYPES_LONG):
    """ make all the figures for the data exploration """
    
    # load data
    engine = sql.create_engine('sqlite:///'+str(IMAGE_SIZE[0])+'_'
                               +str(IMAGE_SIZE[1])+'_'+MAGNIFICATION+'_clean.sqlite', echo=False)
    df = pd.read_sql('images', engine, columns=['type','subtype','image'])
    
    # calculate means and stds
    means_and_stds = np.zeros((df.shape[0], 6))

    for ii in range(0, df.shape[0]):
        means_and_stds[ii,0:3] = np.mean(np.reshape(np.fromstring(df.image[ii]),(IMAGE_SIZE[0],IMAGE_SIZE[1],3)),(0,1))
        means_and_stds[ii,3:] = np.std(np.reshape(np.fromstring(df.image[ii]),(IMAGE_SIZE[0],IMAGE_SIZE[1],3)),(0,1))

        
    # plot histograms  
    fig, ax = plt.subplots(3,2,figsize=(12,9),sharex ='col',sharey='col')

    titles = ['mean R','mean G','mean B','std R','std G','std B']
    for ii in range(0,6):
        ax[ii%3,ii//3].hist(means_and_stds[:,ii], 50, facecolor='blue', alpha=0.75)
        ax[ii%3,ii//3].set_title(titles[ii],size=14)
        ax[ii%3,ii//3].grid(True)

    ax[0,0].set_xlim((0,1))
    ax[2,0].set_xlabel('Value',size=12)
    ax[2,1].set_xlabel('Value',size=12)

    ax[0,0].set_ylabel('Count',size=12)
    ax[1,0].set_ylabel('Count',size=12)
    ax[2,0].set_ylabel('Count',size=12)
    
    plt.suptitle('Histograms of means and standard deviation of colors',size=18)
    
    # make scatterplot
    fig, ax = plt.subplots(figsize=(9,8))

    colors = ['lightgreen','forestgreen','darkgreen','limegreen', 'red', 'firebrick', 'orangered', 'tomato']
    colors_dict = dict(zip(SUBTYPES_SHORT, colors))

    ax.scatter(np.mean(means_and_stds[:,0:3],axis=1), np.mean(means_and_stds[:,3:],axis=1),
               c=df['subtype'].apply(lambda x: colors_dict[x]), lw=0, s=50, alpha =.8)
    ax.set_xlabel('mean',size=12)
    ax.set_ylabel('std',size=12)


    legend1_line2d = list()
    for ii in range(len(colors)):
        legend1_line2d.append(mlines.Line2D([0], [0],
                                            linestyle='none',
                                            marker='o',
                                            alpha=0.8,
                                            markersize=10,
                                            markerfacecolor=colors[ii]))
    plt.legend(legend1_line2d, SUBTYPES_LONG, numpoints=1,fontsize=12,loc='best',shadow='False') 
    
    
    # calculate local binary pattern histograms
    lbp = np.zeros((df.shape[0],10))
    for ii in range(0,df.shape[0]):
        im = np.mean(np.reshape(np.fromstring(df.image[ii]),(IMAGE_SIZE[0],IMAGE_SIZE[1],3)),2)
        lbp_image=local_binary_pattern(im,8,2,method='uniform')
        histogram=scipy.stats.itemfreq(lbp_image)
        lbp[ii,:]=histogram[:,1]
    
    # convert to dataframe
    lbp_df = pd.DataFrame(index=df.index,data=lbp)
    lbp_df['type']=df.type
    lbp_df['subtype']=df.subtype

    means_type = np.array(lbp_df.groupby('type').mean().reindex(TYPES_SHORT))
    stds_type = np.array(lbp_df.groupby('type').std().reindex(TYPES_SHORT))
    means_subtype = np.array(lbp_df.groupby('subtype').mean().reindex(SUBTYPES_SHORT))
    stds_subtype = np.array(lbp_df.groupby('subtype').std().reindex(SUBTYPES_SHORT))

    # make bar plots
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,8))

    ind = np.arange(10) # the x locations for the groups
    width = 0.35 # the width of the bars

    ## the bars
    rects = []
    c = ['lightgreen', 'red']
    for ii in range(len(TYPES_LONG)):
        rects.append(ax1.bar(ind+ii*width, means_type[ii,:], width,
                        color=c[ii],
                        yerr=stds_type[ii,:],
                        error_kw=dict(elinewidth=2,ecolor='black')))

    # axes and labels
    ax1.set_xlim(-width,len(ind)+width)
    ax1.set_ylabel('Count')
    ax1.set_title('Histogram of local binary patterns')
    ax1.set_xticks(ind+width)
    ax1.set_xticklabels([])

    ## add a legend
    ax1.legend( rects, TYPES_LONG, fontsize=12, loc='best' )

    ind = np.arange(10) # the x locations for the groups
    width = 0.1 # the width of the bars

    ## the bars
    rects = []
    c = colors
    for ii in range(len(SUBTYPES_LONG)):
        rects.append(ax2.bar(ind+ii*width, means_subtype[ii,:], width,
                        color=c[ii],
                        yerr=stds_subtype[ii,:],
                        error_kw=dict(elinewidth=2,ecolor='black')))

    # axes and labels
    ax2.set_xlim(-width,len(ind)+width)
    ax2.set_ylim(0,1.5*np.max(means_subtype))
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram of local binary patterns')
    ax2.set_xticks(ind+width)
    ax2.set_xticklabels([])

    ## add a legend
    ax2.legend( rects, SUBTYPES_LONG, fontsize=12, loc='upper center',ncol=3 )
    


    