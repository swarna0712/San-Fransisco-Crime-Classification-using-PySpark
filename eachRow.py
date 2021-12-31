from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.functions import *
 
window = Window.partitionBy("label").orderBy('tiebreak')
x=df.withColumn('tiebreak', monotonically_increasing_id()).withColumn('rank', rank().over(window)).filter(col('rank') == 1).drop('rank','tiebreak').select('pddis','s[0]','s[1]','hour','minute','year','label').orderBy('label')
x.show(39,False)
y=x.toPandas()

import numpy as np
  
# Declare an empty dictionary
d = {}
z=list(y.columns)
for i in z:
    # Add key as column_name and
    # value as list of column values
    d[i] = y[i].values.tolist()