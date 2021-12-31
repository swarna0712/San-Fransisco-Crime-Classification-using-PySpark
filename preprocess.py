from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
from pyspark.sql import SQLContext
import sys
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import abs
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from collections import OrderedDict
import pyspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, create_map, lit
from itertools import chain
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import logging
logging.basicConfig(level=logging.ERROR)
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

data_dict = {'FRAUD':1, 'SUICIDE':2, 'SEX OFFENSES FORCIBLE':3, 'LIQUOR LAWS':4, 
'SECONDARY CODES':5, 'FAMILY OFFENSES':6, 'MISSING PERSON':7, 'OTHER OFFENSES':8, 
'DRIVING UNDER THE INFLUENCE':9, 'WARRANTS':10, 'ARSON':11, 'SEX OFFENSES NON FORCIBLE':12,
'FORGERY/COUNTERFEITING':13, 'GAMBLING':14, 'BRIBERY':15, 'ASSAULT':16, 'DRUNKENNESS':17,
'EXTORTION':18, 'TREA':19, 'WEAPON LAWS':20, 'LOITERING':21, 'SUSPICIOUS OCC':22, 
'ROBBERY':23, 'PROSTITUTION':24, 'EMBEZZLEMENT':25, 'BAD CHECKS':26, 'DISORDERLY CONDUCT':27,
'RUNAWAY':28, 'RECOVERED VEHICLE':29, 'VANDALISM':30,'DRUG/NARCOTIC':31, 
'PORNOGRAPHY/OBSCENE MAT':32, 'TRESPASS':33,'VEHICLE THEFT':34, 'NON-CRIMINAL':35, 
'STOLEN PROPERTY':36, 'LARCENY/THEFT':37, 'KIDNAPPING':38,'BURGLARY':39}

#dof={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
#dof not done as it has less correlation with label

pddis={'MISSION':1,'BAYVIEW':2,'CENTRAL':3,'TARAVAL':4, 'TENDERLOIN':5,'INGLESIDE':6, 'PARK':7,'SOUTHERN':8, 'RICHMOND':9,'NORTHERN':10}

def indexNum(df):
    mapping_expr1 = create_map([lit(x) for x in chain(*data_dict.items())])
    mapping_expr2 = create_map([lit(x) for x in chain(*pddis.items())])
    #mapping_expr3 = create_map([lit(x) for x in chain(*dof.items())])
    df1=df.withColumn("label", mapping_expr1.getItem(col("Category")))
    df1=df1.withColumn("pddis", mapping_expr2.getItem(col("PdDistrict")))
    return df1

def preprocess(df):

    #categorical to numerical

    indexed= indexNum(df)

    #normalize X and Y

    vector_assembler = VectorAssembler(inputCols=['X','Y'], outputCol="SS_features")
    indexed = vector_assembler.transform(indexed)
    minmax_scaler = MinMaxScaler(inputCol="SS_features", outputCol="scaled")
    scaled = minmax_scaler.fit(indexed).transform(indexed)
    scaled=scaled.withColumn("s", vector_to_array("scaled")).select(['Dates','Category','pddis','label']+[col("s")[i] for i in range(2)])

    #splitting date

    transformed = (scaled
        .withColumn("day", dayofmonth(col("Dates")))
        .withColumn("month", date_format(col("Dates"), "MM"))
        .withColumn("year", year(col("Dates")))
        .withColumn('second',second(df.Dates))
        .withColumn('minute',minute(df.Dates))
        .withColumn('hour',hour(df.Dates))
        )
    from pyspark.sql.types import IntegerType
    data_df = transformed.withColumn("month", transformed["month"].cast(IntegerType()))

    #normalize year, hour, minute

    from pyspark.sql.functions import expr


    #making featurized vector

    #['Dates','pddis', 'dof','s[0]','s[1]','day','hour','minute','year']
    #columns with the most correlation with label
    data_df=data_df.select('pddis','s[0]','s[1]','hour','minute','year','label')

    #encode label with dictionary values

    hasher = FeatureHasher(inputCols=['pddis','s[0]','s[1]','hour','minute','year'],
                        outputCol="features")
    featurized = hasher.transform(data_df)
    return featurized


from sklearn.neural_network import MLPClassifier
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
from pyspark.sql import SQLContext
import sys
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import abs
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pickle
import os
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
#from stream1.py import mlp_gs

filename = 'mlpMod.sav'
filename1 = 'sgd.sav'
filename2 = 'mnb.sav'
filename3 = 'kmeans.sav'
filename4 = 'pasagg.sav'
ndf={'pddis': [3, 1, 3, 4, 3, 6, 6, 10, 4, 10, 2, 1, 8, 2, 10, 6, 3, 8, 2, 4, 10, 8, 5, 9, 2, 9, 5, 6, 8, 5, 2, 8, 5, 6, 2, 2, 10, 6, 7], 's[0]': [0.05301615544153277, 0.03804652591423038, 0.05449150306545583, 0.019274820091565795, 0.04867270689965121, 0.037219618249988415, 0.03917169739190615, 0.04357794799292515, 0.019047167762423028, 0.04357794799292515, 0.055863797893321555, 0.04453899622008211, 0.048764031338718636, 0.06111517489846753, 0.039970457865552964, 0.0456687582906726, 0.050015397898191545, 0.05397439288859699, 0.06333654241622305, 0.020092962622070106, 0.04477663263246941, 0.04633139976190031, 0.0501800637238011, 0.02032844885391125, 0.06064975773391155, 0.031033579482772834, 0.04967751572088051, 0.0317668591282401, 0.05492898544806748, 0.050271000245853466, 0.06545559229587385, 0.049895345162261504, 0.051402793847329, 0.044851609540570096, 0.057556225727661725, 0.06545559229587385, 0.04302986069898061, 0.044344692730921155, 0.03271730519456689], 's[1]': [0.0015462282352987388, 0.0009649364494095175, 0.0015807347219082958, 0.00030782459704599955, 0.0015007950903107327, 0.00022590408762459136, 0.0002818228372571333, 0.0012759010729321981, 0.0005655478020593092, 0.0012759010729321981, 0.0005829521334363108, 0.0010339197646022829, 0.0013465603689259023, 0.0005041863280875028, 0.0014034298671486772, 0.0006682302690681396, 0.0015644384399699624, 0.001477064470475414, 0.0010099698882960358, 0.0002617166401123478, 0.0014452309094957969, 0.0012217911344288076, 0.001454387465874514, 0.0013362216094799954, 0.0005767342008775459, 0.0015266569172786152, 0.001469014529424511, 3.601888128244849e-05, 0.0014130957192073756, 0.0014366365837809396, 0.00044414589403758096, 0.0012379932990182214, 0.0014592998655747653, 0.0003300500657010469, 0.00042177809541043236, 0.00044414589403758096, 0.0017783484130538812, 0.0007107494232109594, 0.0011850248941816662], 'hour': [12, 14, 22, 21, 17, 18, 16, 23, 0, 23, 1, 1, 17, 10, 10, 21, 19, 22, 17, 20, 15, 19, 22, 19, 18, 14, 11, 9, 9, 22, 17, 18, 17, 23, 22, 17, 23, 12, 19], 'minute': [30, 4, 25, 13, 45, 58, 20, 53, 56, 53, 46, 0, 55, 20, 40, 55, 26, 15, 5, 29, 54, 30, 0, 44, 0, 45, 11, 0, 54, 30, 47, 44, 27, 30, 0, 47, 30, 21, 52], 'year': [2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2013, 2015, 2015, 2013, 2015, 2015, 2015, 2015, 2015, 2015, 2015], 'label': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]}


def mlp(df):
    pandasDF = df.toPandas()
    df2 = pd.DataFrame(ndf)
    df3 = pd.concat([pandasDF, df2], ignore_index = True)
    df3.reset_index()
    df3=df3[['pddis','s[0]','s[1]','label']]
    

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df3, test_size=0.2)

    #'pddis','s[0]','s[1]','hour','minute','year'

    X_train=train.loc[:, train.columns != 'label'].values
    y_train=train['label'].values
    X_val=test.loc[:, test.columns != 'label'].values
    y_val=test['label'].values
    

    #mlp_gs = MLPClassifier(max_iter=10)
    '''
    parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    from sklearn.model_selection import GridSearchCV
    '''

    #clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    #clf.partial_fit(X_train, y_train) # X is train samples and y is the corresponding labels
    #print('Best parameters found:\n', clf.best_params_)

    if not os.path.exists(filename):
        mlp_gs = MLPClassifier(max_iter=100,activation='tanh',alpha=0.0001)
        #print('\n\n\n\nHEREEE\n\n\n')
    else:
        mlp_gs= pickle.load(open(filename, 'rb'))
    
    import numpy as np
    classes=np.arange(1.0,40.0)
    mlp_gs.partial_fit(X_train,y_train,classes=classes)

    y_true, y_pred = y_val , mlp_gs.predict(X_val)

    from sklearn.metrics import confusion_matrix

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    

    cm = confusion_matrix(y_pred, y_val)
    print('\n\n\n\nMLP\n',classification_report(y_pred, y_val))
    print("\nAccuracy of MLPClassifier : ", accuracy_score(y_true, y_pred),'\n\n\n')

    f=open('MLP.txt','a')
    f.write(str(accuracy_score(y_true, y_pred))+'\n')
    f.close()

    pickle.dump(mlp_gs, open(filename, 'wb'))


def sgd(df):
    pandasDF = df.toPandas()
    df2 = pd.DataFrame(ndf)
        
    df3 = pd.concat([pandasDF, df2], ignore_index = True)
    df3.reset_index()

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df3, test_size=0.2)
    f=['pddis','s[0]','s[1]','hour']
    X_train=train[f].values
    y_train=train['label'].values
    X_val=test[f].values
    y_val=test['label'].values


    if not os.path.exists(filename1):
        sgdmodel = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, l1_ratio=0.15, fit_intercept=True, random_state=20, learning_rate='adaptive',eta0=0.01)
        #print('\n\n\n\nHEREEE\n\n\n')
    else:
        sgdmodel= pickle.load(open(filename1, 'rb'))
        
    import numpy as np
    classes=np.arange(1.0,40.0)
    sgdmodel.partial_fit(X_train,y_train,classes=classes)
    y_true, y_pred = y_val , sgdmodel.predict(X_val)

    from sklearn.metrics import confusion_matrix

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    cm = confusion_matrix(y_pred, y_val)
    print('\n\n\n\nSGD\n',classification_report(y_pred, y_val))
    print("\nAccuracy of SGDClassifier : ", accuracy_score(y_true, y_pred),'\n\n\n')

    f=open('SGD.txt','a')
    f.write(str(accuracy_score(y_true, y_pred))+'\n')
    f.close()

    pickle.dump(sgdmodel, open(filename1, 'wb'))

def mnb(df):
    pandasDF = df.toPandas()
    df2 = pd.DataFrame(ndf)
        
    df3 = pd.concat([pandasDF, df2], ignore_index = True)
    df3.reset_index()

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df3, test_size=0.2)

    X_train=train[['pddis','s[0]','s[1]','hour']].values
    y_train=train['label'].values
    X_val=test[['pddis','s[0]','s[1]','hour']].values
    y_val=test['label'].values


    if not os.path.exists(filename2):
        mnb=MultinomialNB(alpha=0.01, fit_prior=True, class_prior=None)
        #print('\n\n\n\nHEREEE\n\n\n')
    else:
        mnb= pickle.load(open(filename2, 'rb'))
        
    import numpy as np
    classes=np.arange(1.0,40.0)
    mnb.partial_fit(X_train,y_train,classes=classes)
    y_true, y_pred = y_val , mnb.predict(X_val)

    from sklearn.metrics import confusion_matrix

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    cm = confusion_matrix(y_pred, y_val)
    
    print('\n\n\n\nMNB\n',classification_report(y_pred, y_val))
    print("\nAccuracy of MNBClassifier : ", accuracy_score(y_true, y_pred),'\n\n\n')

    f=open('MNB.txt','a')
    f.write(str(accuracy_score(y_true, y_pred))+'\n')
    f.close()

    pickle.dump(mnb, open(filename2, 'wb'))




def kmeans(df):
    pandasDF = df.toPandas()
    df2 = pd.DataFrame(ndf)
        
    df3 = pd.concat([pandasDF, df2], ignore_index = True)
    df3.reset_index()

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df3, test_size=0.2)

    X_train=train[['pddis','s[0]','s[1]','hour']].values
    y_train=train['label'].values
    X_val=test[['pddis','s[0]','s[1]','hour']].values
    y_val=test['label'].values


    if not os.path.exists(filename3):
        kmns=MiniBatchKMeans(n_clusters=8, init='k-means++', max_iter=100, batch_size=len(X_train), verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
        print('\n\n\n\nHEREEE\n\n\n')
    else:
        kmns= pickle.load(open(filename3, 'rb'))
        
    kmns=kmns.partial_fit(X_train)
    y_true, y_pred = y_val , kmns.predict(X_val)
    print('K-means centers',kmns.cluster_centers_)

    f=open('KMNS.txt','a')
    f.write(str(kmns.cluster_centers_)+'\n')
    f.close()

    pickle.dump(kmns, open(filename3, 'wb'))