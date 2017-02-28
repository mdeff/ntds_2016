import pandas as pd
data = pd.read_csv('bigdata1.csv',header=None)
import numpy as np
from scipy import stats
import bokeh
from bokeh.plotting import output_notebook, figure, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
import sklearn.neighbors, sklearn.linear_model, sklearn.ensemble, sklearn.naive_bayes # Baseline classification techniques

def load_data():
    df = pd.DataFrame( columns =('Name','Date','CUR_MKT_CAP','PXnow','PX1YR','DIVIDENDY','BEST_EPS','EPS_GROWTH','Sales_growth','PE','fiveyrAvPriceEarnings','Pricebook','Pricesales','CURratio','Quick','DebtEQ','Rating','Prof_margin','oper_margin','assetTurnover'))
    for i in range(0,data.shape[0],30):
        x=pd.DataFrame()
        x['Name'] = pd.Series(15*[data.iloc[i,0]])
        x['Date'] = data.iloc[i+1,1:16]
        #x['BVPS'] = data.iloc[i+2,1:16]
        x['CUR_MKT_CAP'] =data.iloc[i+3,1:16]
        x['PXnow']  =data.iloc[i+4,1:16]
        #x['PX1YR']  =(data.iloc[i+4,2:16]).append(pd.DataFrame(['empty']))
        x['PX1YR']  =pd.concat([data.iloc[i+4,4:16],pd.DataFrame(['empty','empty','empty'])],axis=0).reset_index(drop=True)
        x['DIVIDENDY']  =data.iloc[i+8,1:16]
        x['BEST_EPS'] =data.iloc[i+9,1:16]
        x['EPS_GROWTH'] =data.iloc[i+10,1:16]
        x['Sales_growth'] =data.iloc[i+14,1:16]
        x['PE'] =data.iloc[i+15,1:16]
        x['fiveyrAvPriceEarnings'] =data.iloc[i+16,1:16]
        x['Pricebook'] =data.iloc[i+17,1:16]
        x['Pricesales'] =data.iloc[i+18,1:16]
        x['CURratio'] =data.iloc[i+20,1:16]
        x['Quick'] =data.iloc[i+21,1:16]
        x['DebtEQ'] =data.iloc[i+22,1:16]
        x['Rating'] =data.iloc[i+23,1:16]
        x['Prof_margin'] =data.iloc[i+26,1:16]
        x['oper_margin'] =data.iloc[i+27,1:16]
        x['assetTurnover'] =data.iloc[i+28,1:16]
        df=df.append(x,ignore_index=True)
        if i%6000 == 0: print (i*100/data.shape[0])       
    print('100')
    return df

def columnstofloat(df):
    attributes = df.columns.tolist()
    attributes = attributes[2:]
    for att in attributes:
        df[att]=df[att].astype(float)
    return df,attributes

def remove_outliers(df):
    cols = list(df.columns)
    cols.remove('Name')
    cols.remove('Date')
    for col in cols:
        col_zscore = col + '_zscore'
        df[col_zscore] = (df[col] - df[col].astype(float).mean())/df[col].astype(float).std(ddof=0)

    cols = list(df.columns)
    cols[25:]
    for col in cols[25:]:
        df = df[df[col] < 3]
    return df

def print_correlation_return(df,attributes):
    df3=df.copy()
    for att in  df3.columns.tolist():
        df3[att]=df3[att].astype(float)
    df3 -= df3.mean(axis=0)
    df3 /= df3.std(axis=0)

    dict1 ={}
    attributes2=attributes.copy()
    attributes2.remove('return')
    attributes2.remove('breturn')
    for att in attributes2:
        dict1[att]=df3['return'].corr(df3[att], method='pearson')
        
    for k,v in sorted(dict1.items(),key=lambda p:p[1]):
        print(k, v)
    return

def split_and_permute(df,attributes,test_size,train_size):
    attributesnow=attributes.copy()
    attributesnow.remove('return')
    attributesnow.remove('breturn')
    attributesnow.remove('PX1YR')
    X = df[attributesnow]
    Y = df['return']
    Y2 = df['breturn']
    print('Split: {} testing and {} training samples'.format(test_size, df.shape[0] - test_size))
    perm = np.random.permutation(df.shape[0])
    print(perm)
    x_test  = X.iloc[perm[train_size:]]
    x_train = X.iloc[perm[:train_size]]
    y_test1  = Y.iloc[perm[train_size:]]
    y_test2  = Y2.iloc[perm[train_size:]]
    y_train1 = Y.iloc[perm[:train_size]]
    y_train2 = Y2.iloc[perm[:train_size]]
    return x_test,x_train,y_test1,y_test2,y_train1,y_train2



def bokehplot(df):

    x1, x2, y = 'BEST_EPS', 'CUR_MKT_CAP', 'return'
    n = 8000  # Less intensive for the browser.

    options = dict(
        tools='pan,box_zoom,wheel_zoom,box_select,lasso_select,crosshair,reset,save'   
    )
    plot1 = figure(
        x_range=[-0.1,0.35], y_range=[-2,2],
        x_axis_label=x1, y_axis_label=y,
        **options
    )
    plot2 = figure(
        x_range=[0,20000], y_range=plot1.y_range,
        x_axis_label=x2, y_axis_label=y,
        **options
    )

    html_color = lambda r,g,b: '#{:02x}{:02x}{:02x}'.format(r,g,b)
    #colors = [html_color(150,0,0) if ret <1 else html_color(0,150,0) for ret in df['breturn'][:n]]
    max_dividend= df['DIVIDENDY'].max()
    colors = [html_color(0,int(round(150/max_dividend*ret)),0) for ret in df['DIVIDENDY'][:n]]
    # The above line is a list comprehension.

    radii = np.round(df['CUR_MKT_CAP'][:n] / df['CUR_MKT_CAP'][:n]*2)
    # To link brushing (where a selection on one plot causes a selection to update on other plots).
    source = ColumnDataSource(dict(x1=df[x1][:n], x2=df[x2][:n], y=df[y][:n], radii=radii,colors = colors ))

    plot1.scatter('x1', 'y', source=source, size='radii', color='colors', alpha=0.6)
    plot2.scatter('x2', 'y', source=source, size='radii', color='colors', alpha=0.6)

    plot = gridplot([[plot1, plot2]], toolbar_location='right', plot_width=400, plot_height=400, title='adsf')

    show(plot)





def classifiers(x_test,x_train,y_test2,y_train2):
    clf,train_accuracy, test_accuracy = [], [], []

    clf.append(sklearn.svm.LinearSVC()) # linear SVM classifier
    clf.append(sklearn.linear_model.LogisticRegression()) # logistic classifier
    #clf.append(sklearn.ensemble.RandomForestClassifier())


    for c in clf:
        c.fit(x_train, y_train2)
        train_pred = c.predict(x_train)
        test_pred = c.predict(x_test)
        train_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(y_train2, train_pred)))
        test_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(y_test2, test_pred)))
        print(test_pred.sum())
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    return test_pred

def bokehplot2(x_test_orig,y_test1,test_pred):
    x1, x2, y = 'BEST_EPS', 'DIVIDENDY', 'return'
    n = test_pred.shape[0]  # Less intensive for the browser.

    options = dict(
        tools='pan,box_zoom,wheel_zoom,box_select,lasso_select,crosshair,reset,save'   
    )
    plot1 = figure(
        x_range=[-0.1,0.2], y_range=[-1,1],
        x_axis_label=x1, y_axis_label=y,
        **options
    )
    plot2 = figure(
        x_range=[0,8], y_range=plot1.y_range,
        x_axis_label=x2, y_axis_label=y,
        **options
    )

    html_color = lambda r,g,b: '#{:02x}{:02x}{:02x}'.format(r,g,b)
    colors = [html_color(150,0,0) if ret < 1  else html_color(0,150,0) for ret in test_pred[:n]]
    # The above line is a list comprehension.

    radii = np.round((test_pred[:n]*0+3))
    # To link brushing (where a selection on one plot causes a selection to update on other plots).
    source = ColumnDataSource(dict(x1=x_test_orig[x1][:n], x2=x_test_orig[x2][:n], y=y_test1[:n], radii=radii,colors = colors ))

    plot1.scatter('x1', 'y', source=source, size='radii', color='colors', alpha=0.6)
    plot2.scatter('x2', 'y', source=source, size='radii', color='colors', alpha=0.6)

    plot = gridplot([[plot1, plot2]], toolbar_location='right', plot_width=400, plot_height=400, title='adsf')

    show(plot)

