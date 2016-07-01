#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import log, log10

import sys
try:
    from pyspark.conf import SparkConf
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import SVMWithSGD
    from pyspark.mllib.feature import Normalizer
    from pyspark.mllib.linalg import Vectors
    from pyspark.sql import SQLContext, Row
    import pyspark
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)


def getFeature(v):
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

    features = []
    features.append(v[4])

    features.append(v[7])
    features.append(v[8])
    features.append(int(v[10]) - int(v[9]))
    features.append(v[11])
    features.append(v[15])
    features.append(v[16])
    return features


def toFloat(v):
    try:
        if v != '':
            return float(v)
        else:
            return ''
    except ValueError:
        print('Error' + str(v))


def unseenDataParsing(line):
    values = line.strip().split(',')

    v = list(map(toFloat, values))
    return [v[0], [1] + getFeature(v)]
    # return v


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
    values = line.strip().split(',')
    v = list(map(toFloat, values))
    return LabeledPoint(v[3]-1, [1] + getFeature(v))
    # return v



# def toNomalizeRDD(data_arr, argv):
#     print('>>make normalization...')
#     normalizer2 = Normalizer(p=1.0)
#
#     for index in argv:
#         values = list(map(lambda data: data[index], data_arr))
#         v = Vectors.dense(values)
#         print(normalizer2.transform(v).collect().distinct())
#
#     print('>>end normalization...')
#     return None
#
# def createDF(data):
#     data = data.split(',')
#     column_names=[
#         "INDEX",
#         "DATE",
#         "REGION",
#         "GENDER",
#         "AGE",
#         "OCCUPATION",
#         "EDUCATION",
#         "INCOME",
#         "CHANNEL",
#         "CHANNEL_START",
#         "CHANNEL_END",
#         "VIEWTIME",
#         "PROG_CODE",
#         "PROG_START",
#         "PROG_END",
#         "GENRE",
#         "PROGRAM_TYPE"
#     ]
#     df = sqlContext.createDataFrame([(Vectors.dense(data))], column_names)
#     # row = Row(
#     #     INDEX=data[0],
#     #     DATE=data[1],
#     #     REGION=data[2],
#     #     GENDER=data[3],
#     #     AGE=data[4],
#     #     OCCUPATION=data[5],
#     #     EDUCATION=data[6],
#     #     INCOME=data[7],
#     #     CHANNEL=data[8],
#     #     CHANNEL_START=data[9],
#     #     CHANNEL_END=data[10],
#     #     VIEWTIME=data[11],
#     #     PROG_CODE=data[12],
#     #     PROG_START=data[13],
#     #     PROG_END=data[14],
#     #     GENRE=data[15],
#     #     PROGRAM_TYPE=data[16]
#     # )
#
#     return df
