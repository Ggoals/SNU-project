#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
try:
    from pyspark.conf import SparkConf
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    from pyspark.mllib.regression import LabeledPoint
    import pyspark
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


def unseenDataParsing(line):
    values = line.strip().split(',')
    def toFloat(v):
        try:
            if v != '':
                return float(v)
            else:
                return ''
        except ValueError:
            print(v)
    v = list(map(toFloat, values))
    features = v[7:16]
    return [v[0], [1] + features]


def parsePoint(line):
    """INDEX, -- 0
    DATE,
    REGION,
    GENDER,
    AGE,
    OCCUPATION, -- 5
    EDUCATION,
    INCOME,
    CHANNEL,
    CHANNEL_START,
    CHANNEL_END,  -- 10
    VIEWTIME,
    PROG_CODE,
    PROG_START,
    PROG_END,
    GENRE,  -- 15
    PROGRAM_TYPE"""
    v = line.strip().split(',')
    features = v[7:16]
    return LabeledPoint(v[3], [1] + features)


def main(input_file_path):

    print('=====>>>>>')
    print('ddd')
    sc = pyspark.SparkContext()
    data = sc.textFile(input_file_path)
    traning_data_RDD = data.filter(lambda line: line.split(',')[3] != '' and line.split(',')[0] != 'INDEX')
    unseen_data_RDD = data.filter(lambda line: line.split(',')[3] == '')


    parsedData = traning_data_RDD.map(parsePoint)
    parsedData.persist()
    # # Correct print: [LabeledPoint(1.0, [1.0,8.6662186586,6.98047693487])]
    logisticRegressionModel = LogisticRegressionWithLBFGS.train(parsedData, numClasses=3)

    labelsAndPreds = parsedData.map(lambda lp: [lp.label, logisticRegressionModel.predict(lp.features)])
    Accuracy = labelsAndPreds.filter(lambda ele: int(ele[0]) == int(ele[1])).count() / float(parsedData.count())
    print("Training Accuracy on training data = " + str(Accuracy))





    # unseen_parsed_data = unseen_data_RDD.map(unseenDataParsing)
    # labelsAndPreds = unseen_parsed_data.map(lambda lp: (lp[0], logisticRegressionModel.predict(lp[1])))
    # print('=====>>>>>')
    # print('=====>>>>>')
    # print('=====>>>>>')
    # print('=====>>>>>')
    #
    # file = open('/Users/1002720/Documents/workspace/SNU-project/data/BDA2Project/1-GenderPrediction/result.csv', 'w', encoding='utf-8')
    # file.write("INDEX,GENDER\n")
    # for label in labelsAndPreds.collect():
    #     file.write(str(int(label[0])) + ',' + str(label[1])+'\n')


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