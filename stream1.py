#! /usr/bin/python3
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
from telnetlib import Telnet
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
from preprocess import preprocess
from preprocess import mlp
from preprocess import sgd
from preprocess import mnb
from preprocess import kmeans
import pyspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession
#from mlp import mlp
from sklearn.neural_network import MLPClassifier
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def convert_to_row(d: dict) -> Row:
    return Row(**OrderedDict(sorted(d.items())))

count=0

def j(rdd):
    df1=rdd.collect()
    #deptDF1 = ss.createDataFrame(rdd, schema = deptSchema)
    #deptDF1.show()

    if df1!=[]:
        #print('\n\n',df1,type(df1[0][1:-1]),'\n\n')
        d = json.loads(df1[0])
        #print(d)
        dictList= lambda x: d[x]
        df=sc.parallelize(list(map(dictList,d))).map(convert_to_row).toDF(['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])  
        #df.show()
        df1=preprocess(df)
        df1.show()
        mlp(df1)
        sgd(df1)
        mnb(df1)
        kmeans(df1)
        df1.unpersist()


sc=SparkContext('local[2]',appName="crime")
ss=SparkSession(sc) 
ssc=StreamingContext(sc,2)
dataStream=ssc.socketTextStream('localhost',6100)
words=dataStream.flatMap(lambda line : line.split('}\}'))

#deptSchema = 'Dates TIMESTAMP,Category STRING,Descript STRING,DayOfWeek STRING,PdDistrict STRING,Resolution STRING,Address STRING,X DOUBLE,Y DOUBLE'
  
    
words.foreachRDD(lambda x:j(x))

ssc.start()
ssc.awaitTermination(2000)
ssc.stop()