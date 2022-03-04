import pandas as pd
import numpy as np
import os
import pymongo
from tscv import GapKFold,GapWalkForward
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.feature_selection import RFECV
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from utils import load_xy, ini_tree, ini_tree_choose, load_config, loc_zq_collection
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,Lasso
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

X = pd.read_excel('x_sw_coal_features.xlsx', index_col=0)
X.index = [i.strftime("%Y-%m-%d") for i in X.index]
Y = pd.read_excel('y_ppi.xlsx', index_col=0)['PPI:煤炭开采和洗选业:当月同比']
Y.index = [i.strftime("%Y-%m-%d") for i in Y.index]
x_data = X[:85]
y_data = Y[:85]
print('get the x_data and y_data!!!')

gcv = GapWalkForward(n_splits=5,gap_size=1,test_size=6)
tscv = TimeSeriesSplit(n_splits=5,test_size=12)
lre = LinearRegression()
ela = ElasticNet(alpha = 1)
las = Lasso(alpha = 1,max_iter=10)
rid = Ridge(alpha = 1000 )

# best_ting = {'alpha':[0.01,0.1,1,10,100,1000]}
# best_g = GridSearchCV(ElasticNet(),best_ting,cv=gcv)
# best_g.fit(x_data,y_data)
# print(best_g.best_params_)
# print(best_g.best_score_)
# print(best_g.best_index_)



