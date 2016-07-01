#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import log, log10

from Common import *

import sys
try:
    from pyspark.conf import SparkConf
    from pyspark.mllib.classification import NaiveBayes
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

    traning_data_pddf = create_pddf(traning_data_RDD)
    traning_data_df = sqlContext.createDataFrame(traning_data_pddf)
    print(traning_data_df.head())

    parsed_data = rdd_to_labeled_point(traning_data_df.rdd)
    parsed_data.persist()
    # Correct print: [LabeledPoint(1.0, [1.0,8.6662186586,6.98047693487])]
    naiveBayesModel = NaiveBayes.train(parsed_data)

    labels_and_preds = parsed_data.map(lambda lp: [lp.label, naiveBayesModel.predict(lp.features)])
    Accuracy = labels_and_preds.filter(lambda ele: int(ele[0]) == int(ele[1])).count() / float(parsed_data.count())
    print("Training Accuracy on training data = " + str(Accuracy))

    unseen_data_pddf = create_pddf(unseen_data_RDD)
    unseen_data_df = sqlContext.createDataFrame(unseen_data_pddf)
    unseen_parsed_data = rdd_to_index_featurs(unseen_data_df.rdd)
    unseen_parsed_data.persist()

    file = open('/Users/1002720/Documents/workspace/SNU-project/data/BDA2Project/1-GenderPrediction/result.csv', 'w', encoding='utf-8')
    file.write('INDEX,GENDER\n')
    for data in unseen_parsed_data.collect():
        file.write(str(data[0])+','+str(naiveBayesModel.predict(data[1]) + 1)+'\n')
    # print(labels_and_preds.collect())




    parsed_data.unpersist()
    unseen_parsed_data.unpersist()
    print('=====>>>>>')
    print('=====>>>>>')
    print('=====>>>>>')
    print('=====>>>>>')



# if len(sys.argv) < 2:
#     print('please enter input file path')
#     sys.exit(1)

#main(sys.argv[1])
main('/Users/1002720/Documents/workspace/SNU-project/data/BDA2Project/1-GenderPrediction/masked_data_gender.csv')