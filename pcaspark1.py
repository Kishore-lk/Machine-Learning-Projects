#! /usr/bin/python3
#pcaspark1.py

#importing Packages

import pandas as pd
import numpy as np
import plotly.express as px
import pyspark.ml
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA

#Building Spark sessions and configurations
spark = SparkSession.builder.master("local[*]") .config("spark.executor.memory", "90g") .config("spark.driver.memory", "70g") .config("spark.memory.offHeap.enabled",True) .config("spark.memory.offHeap.size","32g"). appName("Pcaspark"). getOrCreate()
#spark = SparkSession.builder.appName("Pcaspark").getOrCreate()
sc = SparkContext.getOrCreate()
spark.conf.set("spark.sql.debug.maxToStringFields", 100000)

#Reading dataset
df = pd.read_csv("Foods.csv")

df['level'] = df['level'].astype(float)  #Datatype conversion
spark_df = spark.createDataFrame(df)     #Saving data as Spark dataframe

#Transforming the features as vector Assembler
assembler = VectorAssembler(inputCols=spark_df.columns, outputCol = "features")
output_dat = assembler.transform(spark_df).select("features")
dat = assembler.transform(spark_df)
output_dat.show(5, truncate = False)

#Fitting PCA model
pca = PCA(k=len(spark_df.columns),inputCol="features")
fit = pca.fit(output_dat)
transformed_feature = fit.transform(output_dat)

#Explained Variance generation
var_df = pd.DataFrame(
                np.round(100.00 * fit.explainedVariance.toArray(),3),
                        index = ["PC"+str(x) for x in range(1,len(spark_df.columns)+1)],
                                columns = ['Explained Variance']
                                )
print(var_df)

px.bar(var_df, x=var_df.index, y="Explained Variance")

#Loading PCA and converting to CSV for visualizations
principal_components = np.transpose(np.round(fit.pc.toArray(),3))
loadings_df = pd.DataFrame(
                principal_components,
                        columns = ['PC'+str(x) for x in range(1,len(spark_df.columns)+1)],
                                index = spark_df.columns
                                )
print(loadings_df)
#loadings_df.to_csv("test1.csv",index=False)


#Train, Test Split
train_df,test_df = dat.randomSplit([0.7,0.3])
#train_df.show(4)

#Model Building
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier

# Random Forest Model
rf = RandomForestClassifier(featuresCol='features',labelCol='level')
rf_model = rf.fit(train_df)
y_pred = rf_model.transform(test_df)

print("Random Forest Model")
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(labelCol='level',metricName='accuracy')
multi_evaluator.evaluate(y_pred)

from pyspark.mllib.evaluation import MulticlassMetrics
rf_metric = MulticlassMetrics(y_pred['level', 'prediction'].rdd)
print("Accuracy",rf_metric.accuracy)
print("Precision",rf_metric.precision(1.0))
print("Recall",rf_metric.recall(1.0))
print("F1Score",rf_metric.fMeasure(1.0))

# Decision tree Model
dt = DecisionTreeClassifier(featuresCol='features',labelCol='level')
dt_model = dt.fit(train_df)
y_pred = dt_model.transform(test_df)

print("Decision tree Model")
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(labelCol='level',metricName='accuracy')
print("Accracy", multi_evaluator.evaluate(y_pred))

#from pyspark.mllib.evaluation import MulticlassMetrics
#dt_metric = MulticlassMetrics(y_pred['level', 'prediction'].rdd)
#print("Accuracy",dt_metric.accuracy)

# Logistic Model
lr = LogisticRegression(featuresCol='features',labelCol='level')
lr_model = lr.fit(train_df)
y_pred = lr_model.transform(test_df)

print("Logistic Model")
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(labelCol='level',metricName='accuracy')
print("Accuracy", multi_evaluator.evaluate(y_pred))

#from pyspark.mllib.evaluation import MulticlassMetrics
#lr_metric = MulticlassMetrics(y_pred['level', 'prediction'].rdd)
#print("Accuracy",lr_metric.accuracy)
#print("Precision",lr_metric.precision(1.0))
#print("Recall",lr_metric.recall(1.0))
#print("F1Score",lr_metric.fMeasure(1.0))
