# Crime-Classification-using-PySpark
Big Data Project - SSML - Spark Streaming for Machine Learning

### San Francisco Crime Classification

An open-ended project to learn and then train a model to classify crime which is a single vector of size 9 in the dataset to its corresponding label/category. There are 878k records in the train and test dataset.
 
### Design Details

Streaming – To stream the batches, first run stream.py on a terminal and when it shows waiting run streamclient.py on the other terminal. This starts a spark context, session and then converts RDD to a dictionary and then to a dataframe.
 
Pre-processing - Since the dataset provided is already clean, there are no missing values. We use VectorAssembler for creating a vector of X & Y, and have normalized only the X and Y categories using MinMaxScaler  as they are the only numeric categories in the dataset. This shifts and rescales the data such that they are between 0 and 1.
We then splitted the Dates column into Day, month , year, hour, minute and second.
We have also removed Resolution and descript columns since they are not provided in the test dataset.



