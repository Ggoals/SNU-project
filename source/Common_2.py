#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import sys
import numpy as np
import pandas as pd

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
    features.append(v[2])
    features.append(v[5])
    features.append(v[7])
    features.append(v[15])
    features.append(v[15])
    features.append(v[15])

    features.append(v[8])
    features.append(int(v[10]) - int(v[9]))
    features.append(v[11])

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


def rdd_to_labeled_point(rdd):
    print('<><><><><>>><<><><><><><')
    print(list(rdd.take(1))[0])
    labeled_rdd = rdd.map(lambda v: LabeledPoint(int(list(v)[1]), [1] + list(v)[2:]))
    return labeled_rdd


def rdd_to_index_featurs(rdd):
    print('<><><><><>>><<><><><><><')
    print(list(rdd.take(1))[0])
    try:
        labeled_rdd = rdd.map(lambda v: [list(v)[0], [1] + list(v)[2:]])
        return labeled_rdd
    except ValueError:
        print(rdd.collect())



def create_pddf(csv_data):
    vectors_rdd = csv_data.map(lambda l: l.split(","))
    rows = vectors_rdd.map(lambda data: data)
    column_names = [
        "INDEX",
        "DATE",
        "REGION",
        "GENDER",
        "AGE",
        "OCCUPATION",
        "EDUCATION",
        "INCOME",
        "CHANNEL",
        "CHANNEL_START",
        "CHANNEL_END",
        "VIEWTIME",
        "PROG_CODE",
        "PROG_START",
        "PROG_END",
        "GENRE",
        "PROGRAM_TYPE"
    ]

    data = pd.DataFrame(rows.collect(), columns=column_names)

    data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m%d')
    data['DAY'] = data['DATE'].dt.weekday
    data['DAY'] = data['DAY'].astype('category')
    data.drop('DATE', axis=1, inplace=True)

    data['GENDER'] = data['GENDER'].astype('double').astype('int64')
    data['GENRE'] = data['GENRE'].astype('double')
    data['GENRE1'] = (data['GENRE'] / 100000000).astype('int64')
    data['GENRE2'] = ((data['GENRE'] - data['GENRE1'] * 100000000) / 100000).astype('int64')
    data['GENRE3'] = ((data['GENRE'] - data['GENRE1'] * 100000000 - data['GENRE2'] * 100000) / 100).astype('int64')
    data.drop('GENRE', axis=1, inplace=True)

    #data.set_index('INDEX', inplace=True)

    data['PROG_CODE'] = data['PROG_CODE'].astype('category')
    data['PROGRAM_TYPE'] = data['PROGRAM_TYPE'].astype('category')
    data['REGION'] = data['REGION'].astype('category')
    data['OCCUPATION'] = data['OCCUPATION'].astype('category')
    data['EDUCATION'] = data['EDUCATION'].astype('category')
    data['VIEWTIME'] = data['VIEWTIME'].astype('double').astype('int64')
    data['GENDER'] = data['GENDER'].astype('category')
    data['GENRE1'] = data['GENRE1'].astype('category')
    data['GENRE2'] = data['GENRE2'].astype('category')
    data['GENRE3'] = data['GENRE3'].astype('category')
    data['INCOME'] = data['INCOME'].astype('category')

    data.drop('PROG_CODE', axis=1, inplace=True)
    data.drop('PROG_START', axis=1, inplace=True)
    data.drop('PROG_END', axis=1, inplace=True)
    data.drop('CHANNEL', axis=1, inplace=True)
    data.drop('CHANNEL_START', axis=1, inplace=True)
    data.drop('CHANNEL_END', axis=1, inplace=True)

    # data['AGE'] = data['AGE'].astype('double').astype('int64')

    data['VIEWTIME'] = pd.cut(data['VIEWTIME'], 24)

    # data.drop('VIEWTIME', axis=1, inplace=True)
    # data.drop('EDUCATION', axis=1, inplace=True)
    # 'EDUCATION',
    categorical_cols = ['GENDER' ,'VIEWTIME', 'EDUCATION', 'DAY', 'OCCUPATION', 'INCOME', 'GENRE1', 'GENRE2', 'GENRE3', 'PROGRAM_TYPE', 'REGION']
    for cc in categorical_cols:
        dummies = pd.get_dummies(data[cc])
        dummies = dummies.add_prefix("{}#".format(cc))
        data.drop(cc, axis=1, inplace=True)
        data = data.join(dummies)

    return data
