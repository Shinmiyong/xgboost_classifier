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
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
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
xgblogger = logging.getLogger("xgb")
xgblogger.setLevel(logging.DEBUG)
warnings.filterwarnings('ignore')

if _platform == "linux" or _platform == "linux2":
    plt.rcParams["font.family"] = 'NanumGothicCoding'  # Ubuntu font
elif _platform == "darwin":
    plt.rcParams["font.family"] = 'AppleGothic'  # Mac
print('<meta charset="utf-8">')


class XGB():
    def __init__(self):
        pass
    
    
    # 데이터 리드 후 데이터 스케일링을 진행한다
    def data_split(self, data_dir):
        df_interX = pd.read_csv(data_dir + '/interrojo_X.csv')
        df_interY = pd.read_csv(data_dir + '/interrojo_Y.csv')
        print(f'x dataset: {df_interX.shape}\n \
                y dataset: {df_interY.shape}')
        
        scaler = StandardScaler()

        scaled_interX = scaler.fit_transform(df_interX)  # Scaling of Features part for optimisation
        X_train = pd.DataFrame(scaled_interX, columns=df_interX.columns[:-1])  # Converting Features into 2D DataFrame
        X_train, X_test, Y_train, Y_test = train_test_split(scaled_interX, df_interY, test_size = 0.3, random_state = 1)
        
        ROS = RandomOverSampler()
        X_samp, y_samp = ROS.fit_resample(X_train, Y_train)
        print(X_samp.shape, y_samp.shape)
        
    # def fit(self, data_dir='./interrojo_train.csv',
    #         model_path='.\xgb_model.json', n_estimators=500,
    #         max_depth=4, learning_rate=0.01):
    #     try:
    #         xgblogger.debug('[data_split]' +str(data_dir))
    #         self.data_split(image_dir=data_dir)

    #     except Exception as e:
    #         xgblogger.error(str(e))
    #         return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('condition', type=str, help='fit pred result')
    subparsers = parser.add_subparsers(help='sub-command help', dest='fun')

    # 'training' 명령을 위한 파서를 만든다.
    parser_fit = subparsers.add_parser('fit', help='fit help')
    parser_fit.add_argument('-data_dir', type=str, help='input path help', required=True)
    parser_fit.add_argument('-model_path', type=str, help='model path help', required=True)
    parser_fit.add_argument('-n_estimators', type=str, help='n_estimators help', required=True)
    parser_fit.add_argument('-max_depth', type=str, help='max_depth help', required=True)
    parser_fit.add_argument('-learning_rate', type=str, help='learning rate help', required=True)

    # 'predict' 명령을 위한 파서를 만든다.
    parser_fit = subparsers.add_parser('predict', help='predict help')
    parser_fit.add_argument('-data_path', type=str, help='test file input path help', required=True)
    parser_fit.add_argument('-model_path', type=str, help='model path help', required=True)

    # args = parser.parse_args(r'fit fit -data_dir .\interrojo_train.csv -model_path .\xgb_model.json -n_estimators 500 -max_depth 4 -learning_rate 0.01'.split())
    args = parser.parse_args()
    
    if 'fit' == args.condition:
        # data_dir = args.data_dir
        # model_path = args.model_path
        # n_estimators = int(args.n_estimators)
        # max_depth = int(args.max_depth)
        # learning_rate = float(args.learning_rate)
    
        data_dir = './datasets'
        xgb = XGB()
        xgb.data_split(data_dir)
        # xgb.fit(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
    elif 'predict' == args.condition:
        xgb = XGB()
        data_path = args.data_path
        model_path = args.model_path
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # gpu 사용안하게..
        xgb.predict(data_path=data_path, model_path=model_path)
    else:
        print('Unknown arguments')     
        
