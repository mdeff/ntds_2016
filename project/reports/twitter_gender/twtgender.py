#List of functions : 
# 	colorsGraphs(df, feature, genderConfidence = 1, nbToRemove = 1)
# 	text_normalizer(s)
# 	compute_bag_of_words(text)
# 	print_most_frequent(bow, vocab, gender, n=20)
# 	model_test(model,X_train,y_train,X_test,y_test, full_voc, displayResults = True, displayColors = False)
# 	predictors(df, feature, model, modelname, displayResults = True, displayColors = False)
# 	test_external_data(text, full_voc, model)
# 	combine_features(model_text, model_pic, model_color, data, voc_text, voc_pic, voc_color, acc_text, acc_pic, acc_color)

import pandas as pd
import numpy as np
from IPython.display import display
import re

#graph
from bokeh.plotting import output_notebook, figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

from matplotlib import pyplot as plt
# 3D visualization
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

from collections import Counter


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display
from sklearn import linear_model, metrics
from sklearn import naive_bayes
from sklearn import neural_network


#Definition of function for data exploration for the colors
#feature : 'sidebar_color', 'link_color'
# The colorGraphs function plots the most used colors by gender in 3 bar graphs
def colorsGraphs(df, feature, genderConfidence = 1, nbToRemove = 1):

    dfCol = df.loc[:,['gender:confidence', 'gender', feature]] #Remove weird values : E+17...
    dfColFiltered = dfCol[(dfCol['gender:confidence'] >= genderConfidence)&((dfCol[feature]).str.contains('E\+') != True)]   
    dfColFilteredMale = dfColFiltered[dfColFiltered['gender'] == 'male']
    dfColFilteredFemale = dfColFiltered[dfColFiltered['gender'] == 'female']
    dfColFilteredBrand = dfColFiltered[dfColFiltered['gender'] == 'brand']
    
    colorMale = dfColFilteredMale[feature]
    colorFemale = dfColFilteredFemale[feature]
    colorBrand = dfColFilteredBrand[feature]
    
    listMale = list(colorMale.values.flatten())
    listFemale = list(colorFemale.values.flatten())
    listBrand = list(colorBrand.values.flatten())
        
    nCommon = 30
    commonFemale = Counter(listFemale).most_common(nCommon)
    commonMale = Counter(listMale).most_common(nCommon)
    commonBrand = Counter(listBrand).most_common(nCommon)
    
    #print(commonBrand[0])
    del commonFemale[0:nbToRemove]
    del commonMale[0:nbToRemove]
    del commonBrand[0:nbToRemove]
    
    colorsFemale = [x[0] for x in commonFemale]
    colorsMale = [x[0] for x in commonMale]
    colorsBrand = [x[0] for x in commonBrand]
    
    colorsNumbFemale = [x[1] for x in commonFemale]
    colorsNumbMale = [x[1] for x in commonMale]
    colorsNumbBrand = [x[1] for x in commonBrand]
    
    colorsHexFemale = ['#' + x + '000000' for x in colorsFemale]
    colorsHexFemale = [x[0:7] for x in colorsHexFemale]
    colorsHexMale = ['#' + x + '000000' for x in colorsMale]
    colorsHexMale = [x[0:7] for x in colorsHexMale]
    colorsHexBrand = ['#' + x + '000000' for x in colorsBrand]
    colorsHexBrand = [x[0:7] for x in colorsHexBrand]
    
    rangeColFemale = list(range(len(colorsFemale)))
    rangeColMale = list(range(len(colorsMale)))
    rangeColBrand = list(range(len(colorsBrand)))
    
    fig1, ax1 = plt.subplots()
    
    bar_width = 0.5
    rects1 = plt.barh(rangeColFemale, colorsNumbFemale, bar_width, label = 'Female', color = colorsHexFemale)
    plt.yticks(rangeColFemale, colorsHexFemale)
    plt.xlabel('Color')
    plt.ylabel(feature)
    plt.title('Most used colors by Females for ' + feature + '\n' + str(nbToRemove) + ' most common occurences removed')
    plt.tight_layout()
    plt.show()
    
    fig2, ax2 = plt.subplots()
    
    bar_width = 0.5
    rects1 = plt.barh(rangeColMale, colorsNumbMale, bar_width, label = 'Male', color = colorsHexMale)
    plt.yticks(rangeColMale, colorsHexMale)
    plt.xlabel('Color')
    plt.ylabel(feature)
    plt.title('Most used colors by Males for ' + feature + '\n' + str(nbToRemove) + ' most common occurences removed')
    plt.tight_layout()
    plt.show()
    
    
    fig3, ax3 = plt.subplots()
    bar_width = 0.5
    rects1 = plt.barh(rangeColBrand, colorsNumbBrand, bar_width, label = 'Brand', color = colorsHexBrand)
    plt.yticks(rangeColBrand, colorsHexBrand)
    plt.xlabel('Color')
    plt.ylabel(feature)
    plt.title('Most used colors by Brands for ' + feature + '\n' + str(nbToRemove) + ' most common occurences removed')
    plt.tight_layout()
    plt.show()

def text_normalizer(s):
    #we will normalize the text by using strings, lowercases and removing all the punctuations
    s = str(s) 
    s = s.lower()
    s = re.sub('\W\s',' ',s)
    s = re.sub('\s\W',' ',s)
    #s = re.sub('\s[^[@\w]]',' ',s) #to keep the @ symbols used for "addressing"
    #s = re.sub('@',' search_arobass_sign ',s) #The CountVectorizer cant handle the @
    s = re.sub('\s+',' ',s) #replace double spaces with single spaces
    
    return s
# The compute_bag_of_words function returns a table with the # of occurence of a word in the text
# and a vocabulary of all the different words
def compute_bag_of_words(text):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(text)
    vocabulary = vectorizer.get_feature_names()
    return vectors, vocabulary


#Exploration of which words are most used by which gender
def print_most_frequent(bow, vocab, gender, n=20, feature = 'text'):
    switcher = {
        'all_text' : "text",
        'pic_text' : "profile picture features",
    }
    featureText =  switcher.get(feature, 'text')
    color_idx = ['brand', 'female', 'male']
    color_table = ['#4a913c', '#f5abb5', '#0084b4']
    label_table = ['Most used words by brands for ' + featureText, 'Most used words by females for ' + featureText, 'Most used words by males for ' + featureText]
    idx = np.argsort(bow.sum(axis=0))
    idx_most_used = np.zeros(n)
    occurence_number = np.zeros(n)
    words_most_used = ["" for x in range(n)]
    for i in range(0,n):
        idx_most_used[i] = idx[0, -1-i]
        words_most_used[i] = vocab[np.int64(idx_most_used[i])]
        occurence_number[i] = (bow.sum(axis=0))[0,idx_most_used[i]]
        #print(vocab[j])

    fig, ax = plt.subplots()
    
    bar_width = 0.5
    word_number = np.arange(n)+1
    rects1 = plt.barh(word_number,occurence_number, bar_width, label = label_table[color_idx.index(gender)], color = color_table[color_idx.index(gender)])
    plt.yticks(word_number,words_most_used)
    plt.ylabel('Most used words')
    plt.xlabel('Number of occurences')
    plt.title(label_table[color_idx.index(gender)])
    plt.tight_layout()
    plt.show()
	
	
# Definition of functions for data analysis and classification

# The model_test function is used to extract the best word predictors and
# anti-predictors for each gender. The model used must have a coef_ attribute
# representing the weight of each word
def model_test(model,X_train,y_train,X_test,y_test, full_voc, displayResults = True, displayColors = False, featureIntent = 'text'):
    
   
        
    switcher = {
        'all_text' : "text",
        'pic_text' : "profile picture features",
        'link_color' : "theme color",
    }
    featureText =  switcher.get(featureIntent, '')
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # compute MSE
    mse = metrics.mean_squared_error(y_test,y_pred)
    print('mse: {:.4f}'.format(mse))

    # Prints the accuracy of the gender prediction
    acc = model.score(X_test,y_test)
    print('score: ', acc)
    
    
    if(displayResults&hasattr(model,'coef_')):
    # W contain the weight for each predictor, for each gender
        W = model.coef_
    
    # Male Predictors 
        print('Best 20 male predictors:')
        idx_male = np.argsort((W[2,:]))
        weight_male_pred = np.zeros(20)
        male_pred_label = ["" for x in range(20)]
        for i in range(20):
            j = idx_male[-1-i]
            weight_male_pred[i] = W[2,j]
            male_pred_label[i] = full_voc[j]
    
        fig1, ax1 = plt.subplots()
    
        bar_width = 0.5
        pred_number = np.arange(20)+1
        if(displayColors):
            colorsHexMale = ['#' + x + '000000' for x in male_pred_label]
            colorsHexMale = [x[0:7] for x in colorsHexMale] 
            rects1 = plt.barh(pred_number,weight_male_pred, bar_width, label = 'Male Predictors', color = colorsHexMale)  
            plt.yticks(pred_number,colorsHexMale)
        else:
            rects1 = plt.barh(pred_number,weight_male_pred, bar_width, label = 'Male Predictors', color = '#0084b4')
            plt.yticks(pred_number,male_pred_label)
        plt.xlabel('Predictor')
        plt.ylabel('Weight')
        plt.title('Best 20 male predictors for ' + featureText)
        plt.tight_layout()
        plt.show()
    # Male Anti-Predictors    
        print('Best 20 male anti-predictors  for ' + featureText + ':')
        idx_male = np.argsort(-(W[2,:]))
        weight_male_antipred = np.zeros(20)
        male_antipred_label = ["" for x in range(20)]
        for i in range(20):
            j = idx_male[-1-i]
            weight_male_antipred[i] = W[2,j]
            male_antipred_label[i] = full_voc[j]
    
        fig2, ax2 = plt.subplots()
    
        bar_width = 0.5
        pred_number = np.arange(20)+1
        if(displayColors):
            colorsHexMaleAnti = ['#' + x + '000000' for x in male_antipred_label]
            colorsHexMaleAnti = [x[0:7] for x in colorsHexMaleAnti] 
            rects1 = plt.barh(pred_number,weight_male_antipred, bar_width, label = 'Male Anti-Predictors', color = colorsHexMaleAnti)
            plt.yticks(pred_number,colorsHexMaleAnti)
        else:
            rects1 = plt.barh(pred_number,weight_male_antipred, bar_width, label = 'Male Anti-Predictors', color = '#0084b4')
            plt.yticks(pred_number,male_antipred_label)
        plt.xlabel('Anti-Predictor')
        plt.ylabel('Weight')
        plt.title('Best 20 male anti-predictors for ' + featureText)
        plt.tight_layout()
        plt.show()
    # Female Predictors    
        print('Best 20 female predictors  for ' + featureText + ':')
        idx_female = np.argsort((W[1,:]))
        weight_female_pred = np.zeros(20)
        female_pred_label = ["" for x in range(20)]
        for i in range(20):
            j = idx_female[-1-i]
            weight_female_pred[i] = W[1,j]
            female_pred_label[i] = full_voc[j]
    
        fig3, ax3 = plt.subplots()
    
        bar_width = 0.5
        pred_number = np.arange(20)+1
        if(displayColors):
            colorsHexFemale = ['#' + x + '000000' for x in female_pred_label]
            colorsHexFemale = [x[0:7] for x in colorsHexFemale] 
            rects1 = plt.barh(pred_number,weight_female_pred, bar_width, label = 'Female Predictors', color = colorsHexFemale)  
            plt.yticks(pred_number,colorsHexFemale)
        else:
            rects1 = plt.barh(pred_number,weight_female_pred, bar_width, label = 'Female Predictors', color = '#f5abb5')
            plt.yticks(pred_number,female_pred_label)
        plt.xlabel('Predictor')
        plt.ylabel('Weight')
        plt.title('Best 20 Female predictors for ' + featureText)
        plt.tight_layout()
        plt.show()
    # Female Anti-Predictors    
        print('Best 20 Female anti-predictors for ' + featureText + ':')
        idx_female = np.argsort(-(W[1,:]))
        weight_female_antipred = np.zeros(20)
        female_antipred_label = ["" for x in range(20)]
        for i in range(20):
            j = idx_female[-1-i]
            weight_female_antipred[i] = W[1,j]
            female_antipred_label[i] = full_voc[j]
    
        fig4, ax4 = plt.subplots()
    
        bar_width = 0.5
        pred_number = np.arange(20)+1
        if(displayColors):
            colorsHexFemaleAnti = ['#' + x + '000000' for x in female_antipred_label]
            colorsHexFemaleAnti = [x[0:7] for x in colorsHexFemaleAnti] 
            rects1 = plt.barh(pred_number,weight_female_antipred, bar_width, label = 'Female Anti-Predictors', color = colorsHexFemaleAnti)  
            plt.yticks(pred_number,colorsHexFemaleAnti)
        else:
            rects1 = plt.barh(pred_number,weight_female_antipred, bar_width, label = 'Female Anti-Predictors', color = '#f5abb5')
            plt.yticks(pred_number,female_antipred_label)
        plt.xlabel('Anti-Predictor')
        plt.ylabel('Weight')
        plt.title('Best 20 Female anti-predictors for ' + featureText)
        plt.tight_layout()
        plt.show()
    # Brand Predictors    
        print('Best 20 brand predictors for ' + featureText + ':')
        idx_brand = np.argsort((W[0,:]))
        weight_brand_pred = np.zeros(20)
        brand_pred_label = ["" for x in range(20)]
        for i in range(20):
            j = idx_brand[-1-i]
            weight_brand_pred[i] = W[0,j]
            brand_pred_label[i] = full_voc[j]
    
        fig5, ax5 = plt.subplots()
    
        bar_width = 0.5
        pred_number = np.arange(20)+1
        if(displayColors):
            colorsHexBrand = ['#' + x + '000000' for x in brand_pred_label]
            colorsHexBrand = [x[0:7] for x in colorsHexBrand] 
            rects1 = plt.barh(pred_number,weight_brand_pred, bar_width, label = 'Brand Predictors', color = colorsHexBrand)
            plt.yticks(pred_number,colorsHexBrand)
        else:
            rects1 = plt.barh(pred_number,weight_brand_pred, bar_width, label = 'Brand Predictors', color = '#4a913c')
            plt.yticks(pred_number,brand_pred_label)
        plt.xlabel('Predictor')
        plt.ylabel('Weight')
        plt.title('Best 20 Brand predictors for ' + featureText)
        plt.tight_layout()
        plt.show()
    # Brand Anti-Predictors    
        print('Best 20 Brand anti-predictors for ' + featureText + ':')
        idx_brand = np.argsort(-(W[0,:]))
        weight_brand_antipred = np.zeros(20)
        brand_antipred_label = ["" for x in range(20)]
        for i in range(20):
            j = idx_brand[-1-i]
            weight_brand_antipred[i] = W[0,j]
            brand_antipred_label[i] = full_voc[j]
    
        fig6, ax6 = plt.subplots()
    
        bar_width = 0.5
        pred_number = np.arange(20)+1
        if(displayColors):
            colorsHexBrandAnti = ['#' + x + '000000' for x in brand_antipred_label]
            colorsHexBrandAnti = [x[0:7] for x in colorsHexBrandAnti] 
            rects1 = plt.barh(pred_number,weight_brand_antipred, bar_width, label = 'Brand Anti-Predictors', color = colorsHexBrandAnti)  
            plt.yticks(pred_number,colorsHexBrandAnti)
        else:
            rects1 = plt.barh(pred_number,weight_brand_antipred, bar_width, label = 'Brand Anti-Predictors', color = '#4a913c')
            plt.yticks(pred_number,brand_antipred_label)
        plt.xlabel('Anti-Predictor')
        plt.ylabel('Weight')
        plt.title('Best 20 Brand anti-predictors for ' + featureText)
        plt.tight_layout()
        plt.show()
    
    return model, acc

# feature is a string in order to use df[feature]

# The predictors function takes a dataframe, a specific feature (should be a string) and a model
# and performs the gender prediction. The set is split in 5 for cross-validation  
def predictors(df, feature, model, modelname, displayResults = True, displayColors = False):
    print('Testing', modelname, 'model for gender prediction using', feature)
    full_bow, full_voc = compute_bag_of_words(df[feature])
    X = full_bow
    y = LabelEncoder().fit_transform(df['gender'])
    # Create Training and testing sets.
    n,d = X.shape
    test_size = n // 5
    print('Split: {} testing and {} training samples'.format(test_size, y.size - test_size))
    perm = np.random.permutation(y.size)
    X_test  = X[perm[:test_size]]
    X_train = X[perm[test_size:]]
    y_test  = y[perm[:test_size]]
    y_train = y[perm[test_size:]]
    print('model: ', modelname)
    model, acc = model_test(model,X_train,y_train,X_test,y_test, full_voc, displayResults = displayResults, displayColors = displayColors, featureIntent = feature)
    
    return model, full_voc, acc
	
def test_external_data(text, full_voc, model, feature, display = True):
    gender_list = ['brand', 'female', 'male']
    vect = CountVectorizer(vocabulary=full_voc)
    new_bow = vect.fit_transform(text).toarray()
    
    predicted_class = model.predict(new_bow)
    
    
    if(hasattr(model, 'predict_proba')):
        proba = model.predict_proba(new_bow)
        if(display):
            print('The predicted gender by using the', feature, 'is', gender_list[predicted_class[0]],'with probability',np.sort(proba[0])[2])
        return proba, predicted_class
    else:
        if(display):
            print('The predicted gender by using the', feature, 'is', gender_list[predicted_class[0]])
        return [], predicted_class
 
 
def combine_features(model_text, model_pic, model_color, data, voc_text, voc_pic, voc_color, acc_text, acc_pic, acc_color, display = True):
    gender_list =['brand', 'female', 'male']
    success = 0
    resultList = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(0,len(data)):
        proba_text, class_text = test_external_data(data['all_text'][i:i+1], voc_text, model_text, 'text', display = display)
        proba_pic, class_pic = test_external_data(data['pic_text'][i:i+1], voc_pic, model_pic, 'profile picture', display = display)
        proba_color, class_color = test_external_data(data['link_color'][i:i+1], voc_color, model_color, 'link color', display = display)
        
        if(proba_text!=[] and proba_pic!=[] and proba_color!=[]):
            weighted_proba = (proba_text*acc_text + proba_pic*acc_pic + proba_color*acc_color)/(acc_text + acc_pic + acc_color)
            proba = weighted_proba[0]
            pred_class = (np.argsort(proba))[2]
            if(display):
                print('Overall, the predicted gender of user',data.iloc[i]['user_name'], 'is' ,gender_list[pred_class], 'with a confidence of',proba[pred_class])
        else:
            result=np.zeros(3)
            result[class_text] = result[class_text]+1
            result[class_pic] = result[class_pic]+1
            result[class_color] = result[class_color]+1
            if(np.max(result)==1):
                pred_class = class_pic
            else:
                pred_class = (np.argsort(result))[2]
            if(display):
                print('Overall, the predicted gender  of user',data.iloc[i]['user_name'], 'is' ,gender_list[pred_class])
        if(gender_list[pred_class]==data.iloc[i]['gender']):
             success = success+1
        originalGender = gender_list.index(data.iloc[i]['gender'])
        resultList[originalGender][pred_class] = resultList[originalGender][pred_class] + 1
    success_rate = success/len(data)
    print('The average success rate for this test data is',success_rate)    
    return resultList



def display_resultList(resultList):
    fig2, ax2 = plt.subplots()
    ax2.set_ylim([-0.5, 3.5]);
    bar_width = 0.5
    rects1 = plt.barh(range(0,3),[x[0] for x in resultList], bar_width, color = '#4a913c', label = 'brand')
    rects2 = plt.barh(range(0,3), [x[1] for x in resultList], bar_width, color = '#f5abb5', left = [x[0] for x in resultList], label = 'female')
    rects3 = plt.barh(range(0,3), [x[2] for x in resultList], bar_width, color = '#0084b4', left = [x[0] + x[1] for x in resultList], label = 'male')
    plt.yticks(range(0,3) ,['brand', 'female', 'male'])
    plt.xlabel('Number of users')
    plt.ylabel('Gender')
    plt.title('Predicted gender vs. Real gender')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
        

def combine_features_without_pic(model_text, model_color, data, voc_text, voc_color, acc_text, acc_color, display = True):
    gender_list =['brand', 'female', 'male']
    success = 0
    resultList = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(0,len(data)):
        proba_text, class_text = test_external_data(data['all_text'][i:i+1], voc_text, model_text, 'text', display = display)
        proba_color, class_color = test_external_data(data['link_color'][i:i+1], voc_color, model_color, 'link color', display = display)
        
        if(proba_text!=[] and proba_color!=[]):
            weighted_proba = (proba_text*acc_text +proba_color*acc_color)/(acc_text + acc_color)
            proba = weighted_proba[0]
            pred_class = (np.argsort(proba))[2]
            if(display):
                print('Overall, the predicted gender of user',data.iloc[i]['user_name'], 'is' ,gender_list[pred_class], 'with a confidence of',proba[pred_class])
        else:
            result=np.zeros(2)
            result[class_text] = result[class_text]+1
            result[class_color] = result[class_color]+1
            if(np.max(result)==1):
                pred_class = class_text
            else:
                pred_class = (np.argsort(result))[1]
            if(display):
                print('Overall, the predicted gender  of user',data.iloc[i]['user_name'], 'is' ,gender_list[pred_class])
        if(gender_list[pred_class]==data.iloc[i]['gender']):
             success = success+1
        originalGender = gender_list.index(data.iloc[i]['gender'])
        resultList[originalGender][pred_class] = resultList[originalGender][pred_class] + 1
    success_rate = success/len(data)
    print('The average success rate for this test data is',success_rate)    
    return resultList
        