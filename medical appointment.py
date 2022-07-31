# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:03:15 2022

@author: LENOVO
"""

import findspark
findspark.init()
import pyspark
findspark.find()
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer

from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#---------------spark context for starting spark session----------------------
spark = SparkSession.builder.master("local").getOrCreate() 

sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

# ----------------------------- Read data ------------------------------------
data = spark.read.option("header","true").option("inferschema", "true").csv(
    "G:\medical appointment\medical appointment.csv")
data.show(5)

data.printSchema()

data.describe().show()

data.count()

data.schema.names
# ----------------------------Rename Header ----------------------------------
data1 = data.withColumnRenamed("Hipertension","Hypertension").withColumnRenamed(
    'Handcap', 'Handicap').withColumnRenamed('No-show', 'No_show')
data1.printSchema()

colms= data1.drop('PatientId','AppointmentID','ScheduledDay','AppointmentDay',
                  'Neighbourhood').schema.names
colms

data1 = data1.select(colms)
data1.show(3)

# -----------------------------lable encoding ---------------------------------
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_type") 
indexer1 = StringIndexer(inputCol="No_show", outputCol="No_show_con") 

pipeline = Pipeline(stages=[indexer, indexer1])

# Fit the pipeline for lable encoding.
pipelineFit = pipeline.fit(data1)
dataset = pipelineFit.transform(data1)
dataset.show(5)

# --------------------------- Remove columns ---------------------------------
colms= dataset.drop("Gender","No_show").schema.names
colms
dataset = dataset.select(colms)
dataset.show(5)

# -------------------------vectorAssembler------------------------------------
colms= dataset.drop("No_show_con").schema.names
colms

vectorAssembler = VectorAssembler(inputCols = colms, outputCol = 'features')
dataset = vectorAssembler.transform(dataset)
dataset.select("features").show(5)

dataset.show(5)
dataset.printSchema()

dataset1 = dataset.select(['features', 'No_show_con'])
dataset1.show(10)

data.printSchema()
dataset1.describe().show()

# -----------------------------Train Test spliting-----------------------------
(trainingData, testData) = dataset1.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " ,trainingData.count())
trainingData.show(10000)

print("Test Dataset Count: " ,testData.count())
testData.show(10000)

# --------------------Model----------------
rf = RandomForestClassifier(labelCol="No_show_con", featuresCol="features", numTrees=10)
model = rf.fit(trainingData)

# ----------------------------Make predictions -------------------------------
predictions = model.transform(testData)
predictions.show()

# -------------------Select example rows to display---------------------------
predictions.select("prediction","No_show_con","features").show(10)

# ---------Select (prediction, true label) and compute test error-------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="No_show_con", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test Error = %g" % (1.0 - accuracy))

print(accuracy)

rfModel = model.stages[2]
print(rfModel)  # summary only


sc.stop()