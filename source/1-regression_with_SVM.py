#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import log, log10

from Common import *

import sys
try:
    from pyspark.conf import SparkConf
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import SVMWithSGD
    from pyspark.ml.feature import Normalizer
    from pyspark.sql import SQLContext, Row
    import pyspark
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


def main(input_file_path):

    print('=====>>>>>')
    print('ddd')
    # sc = pyspark.SparkContext()
    data = sc.textFile(input_file_path)
    traning_data_RDD = data.filter(lambda line: line.split(',')[3] != '' and line.split(',')[0] != 'INDEX')
    unseen_data_RDD = data.filter(lambda line: line.split(',')[3] == '')



    parsedData = traning_data_RDD.map(parsePoint)
    parsedData.persist()
    print(parsedData.take(1))
    # Correct print: [LabeledPoint(1.0, [1.0,8.6662186586,6.98047693487])]
    SVMmodel = SVMWithSGD.train(parsedData, iterations=100)

    labelsAndPreds = parsedData.map(lambda lp: [lp.label, SVMmodel.predict(lp.features) + 1])
    Accuracy = labelsAndPreds.filter(lambda ele: int(ele[0]) == int(ele[1])).count() / float(parsedData.count())
    print("Training Accuracy on training data = " + str(Accuracy))



    parsedData.unpersist()
    print('=====>>>>>')
    print('=====>>>>>')
    print('=====>>>>>')
    print('=====>>>>>')



# if len(sys.argv) < 2:
#     print('please enter input file path')
#     sys.exit(1)

#main(sys.argv[1])
main('/Users/1002720/Documents/workspace/SNU-project/data/BDA2Project/1-GenderPrediction/masked_data_gender.csv')