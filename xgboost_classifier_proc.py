# -*- coding: utf-8 -*-
from msilib.schema import Condition
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from matplotlib import font_manager, rc
from sys import platform as _platform
import argparse
import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='xgb.log',
                format = '%(asctime)s:%(levelname)s:%(message)s',
                datefmt = '%m/%d/%Y %I:%M:%S %p')
cnnlogger = logging.getLogger("xgb")
cnnlogger.setLevel(logging.DEBUG)
warnings.filterwarnings('ignore')

if _platform == "linux" or _platform == "linux2":
    plt.rcParams["font.family"] = 'NanumGothicCoding'  # Ubuntu font
elif _platform == "darwin":
    plt.rcParams["font.family"] = 'AppleGothic'  # Mac
print('<meta charset="utf-8">')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('condition', type=str, help='fit pred')
    subparsers = parser.add_subparsers(help='sub-command help', dest='fun')

    # 'training' 명령을 위한 파서를 만든다.
    parser_fit = subparsers.add_parser('fit', help='fit help')
    parser_fit.add_argument('-data_dir', type=str, help='input path help', required=True)
    parser_fit.add_argument('-model_path', type=str, help='model path help', required=True)
    parser_fit.add_argument('-n_estimator', type=str, help='xgboost의 분류기 갯수')
    parser_fit.add_argument('-max_depth', type=str, help='max_depth help', required=True)
    parser_fit.add_argument('-learning_rate', type=str, help='learning rate help', required=True)

    # 'predict' 명령을 위한 파서를 만든다.
    parser_fit = subparsers.add_parser('predict', help='predict help')
    parser_fit.add_argument('-data_path', type=str, help='test file input path help', required=True)
    parser_fit.add_argument('-model_path', type=str, help='model path help', required=True)

    args = parser.parse_args(r'fit fit -data_dir .\datasets\interrojo_train.csv -model_path .\models\xgb_model.json -n_estimator 500 -max_depth 4 -learning_rate 0.01'.split())
    #args = parser.parse_args()

    n_estimator = None
    learning_rate = None
    max_depth = None
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    if args.condition == 'fit':
        data_dir = args.data_dir
        model_path = args.model_path
        n_estimator = int(args.n_estimator)
        max_depth = int(args.max_depth)
        learning_rate = float(args.learning_rate)


        # train 데이터 불러오기
        df_train = pd.read_csv('datasets/interrojo_train.csv')

        # train 데이터 전처리
        X_train = df_train.iloc[:, :-1]
        Y_train = df_train.iloc[:, [-1]]

        X_train = scaler.fit_transform(X_train)  # Scaling of Features part for optimisation

        X_train = pd.DataFrame(X_train, columns=df_train.columns[:-1])  # Converting Features into 2D DataFrame

        from imblearn.over_sampling import RandomOverSampler

        ROS = RandomOverSampler()
        X_samp, y_samp = ROS.fit_resample(X_train, Y_train)

        xgb = XGBClassifier(n_estimators=n_estimator, learning_rate=learning_rate, max_depth=max_depth)  # n_estimators에 s 빠져있음
        xgb.fit(
            X_samp, y_samp,
            eval_metric='mae', eval_set=[(X_samp, y_samp)])
        xgb.save_model(model_path)

    elif args.condition == 'predict':


        data_path = args.data_path
        model_path = args.model_path

        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # gpu 사용안하게..

        # test 데이터 불러오기
        df_test = pd.read_csv('datasets/interrojo_test.csv')

        # train 데이터 전처리
        X_test = df_test.iloc[:, :-1]


        X_test = scaler.transform(X_test)  # Scaling of Features part for optimisation

        X_test = pd.DataFrame(X_test, columns=df_test.columns[:-1])  # Converting Features into 2D DataFrame

        xgb = XGBClassifier(n_estimators=n_estimator, learning_rate=learning_rate, max_depth=4).load_model(
            model_path)
        pred = xgb.predict(X_test)
        df_results = df_test.iloc[:, :-1]
        df_results['PRED'] = pred
        df_result = df_results.astype({'PRED': 'bool'})
        print(df_result)
    else:
        print('Unknown arguments')






