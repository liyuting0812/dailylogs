import pandas as pd
import numpy as np
import os
import math
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
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from statsmodels.stats.outliers_influence import variance_inflation_factor
# f = open('xy_cache/xy_data.pkl','rb')
# data = pickle.load(f)
# lidata = list(data)
# # print(type(data))
# print(len(lidata))
# print(lidata[0])
# print('\n')
# print(type(lidata[0]))
# print('\n')
# print(lidata[0].values())
# print(type(lidata[0].values()))

'''
1. 首先，导入四个模型，OLS、Elastic Net、Lasso、Ridge
'''

def get_df_by_dict(s: dict):
    re = pd.DataFrame(s['data'], index=s['index'], columns=s['columns'])
    re.sort_index(ascending=True, inplace=True)
    re.index = pd.to_datetime(re.index)
    print(s['columns'])
    return re.iloc[:, 0]

def save_pickle(paras, path):
    with open(path, 'wb') as fo:
        pickle.dump(paras, fo)
    return path

def get_x_by_db_name(db_name):
    meta_data = list(loc_zq_collection(db_name, 'metadata').find())
    map_id = {s['__id']: s['name'] for s in meta_data}
    data_li = list(loc_zq_collection(db_name, 'data').find())
    assert len(data_li) > 0

    df_data = pd.concat([get_df_by_dict(s) for s in data_li], axis=1)
    new_cols = [s for s in df_data.columns if s in map_id]
    df_data = df_data.loc[:, new_cols]
    df_data.columns = [map_id[s] for s in df_data.columns]
    return df_data

def get_y_by_db_name(db_name):
    meta_data = list(loc_zq_collection(db_name, 'metadata').find())
    map_id = {s['__id']: s['name'] for s in meta_data}
    data_li = list(loc_zq_collection(db_name, 'data').find())
    assert len(data_li) > 0

    df_data = pd.concat([get_df_by_dict(s) for s in data_li], axis=1)
    new_cols = [s for s in df_data.columns if s in map_id]
    df_data = df_data.loc[:, new_cols]
    df_data.columns = [map_id[s] for s in df_data.columns]
    return df_data

def load_xy(cfg: dict):
    try:
        os.mkdir('xy_cache')
    except:
        pass
    cache_path = f"xy_cache/xy_data.pkl"
    if not os.path.exists(cache_path):
        x_data = {}
        y_data = {}
        for y_dict in cfg['Y']:
            y_name = y_dict['y_name']
            db_name = y_dict['db_name']
            coll_meta = loc_zq_collection(db_name, 'metadata')
            coll_data = loc_zq_collection(db_name, 'data')
            meta_li = list(coll_meta.find({'name': y_dict['name']}))
            y_meta_dict = meta_li[0]
            y_data_li = list(coll_data.find({'__id': y_meta_dict['__id']}))
            try:
                y_data_dict = y_data_li[0]
            except:
                IndexError(f"数据库{y_name},集合data没有找到{y_dict['name']}")
            if y_dict['x_id'] == 'auto':
                x_id = y_meta_dict['x_id']
            else:
                x_id = y_dict['x_id']
            x_df = get_x_by_db_name(x_id)
            x_df = x_df[x_df.index > pd.to_datetime(y_dict['t_start'])]
            x_data[y_name] = x_df

            y_se = get_df_by_dict(y_data_dict)
            y_se = y_se[y_se.index > pd.to_datetime(y_dict['t_start'])]
            y_se.dropna(inplace=True)

            y_data[y_name] = y_se
        save_pickle((x_data, y_data), cache_path)
    else:
        x_data, y_data = pd.read_pickle(cache_path)
    return x_data, y_data

    X = pd.read_excel('x_sw_coal_features.xlsx',index_col=0)
    print(X.tail())
    X.index = [i.strftime("%Y-%m-%d") for i in X.index]
    print(X.index[0],'\n')
    Y = pd.read_excel('y_ppi.xlsx',index_col=0)['PPI:煤炭开采和洗选业:当月同比']
    Y.index = [i.strftime("%Y-%m-%d") for i in Y.index]
    print(Y.tail())

if __name__ == "__main__":
    X = pd.read_excel('x_sw_coal_features.xlsx', index_col=0)
    X.index = [i.strftime("%Y-%m-%d") for i in X.index]
    Y = pd.read_excel('y_ppi.xlsx', index_col=0)['PPI:煤炭开采和洗选业:当月同比']
    Y.index = [i.strftime("%Y-%m-%d") for i in Y.index]
    x_data = X[:85]
    y_data = Y[:85]
    print('get the x_data and y_data!!!')

# 时间序列交叉验证集
    y_data.plot()
    gcv = GapWalkForward(n_splits=5,gap_size=1,test_size=6)
    tscv = TimeSeriesSplit(n_splits=5,test_size=12)
    lre = LinearRegression()
    ela = ElasticNet(alpha = 1)
    las = Lasso(alpha = 1)
    rid = Ridge(alpha = 1000 )
    # cols = x_data.columns
    # list=[]
    # for k in range(x_data.shape[1]-40):
    #     x_data_ = x_data.iloc[:,k:k+40]
    #     mae_l =[]
    #     for train,test in tscv.split(x_data_):
    #         # print(x_data.iloc[train],x_data.iloc[test])
    #         # print(y_data.iloc[train],y_data.iloc[test])
    #         x_train = x_data_.iloc[train]
    #         x_test = x_data_.iloc[test]
    #         y_train = y_data.iloc[train]
    #         y_test = y_data.iloc[test]
    #         lre.fit(x_train,y_train)
    #         ela.fit(x_train,y_train)
    #         las.fit(x_train,y_train)
    #         y_pred = lre.predict(x_test)
    #         mae_ll = np.mean(mae(y_test,y_pred))
    #         mae_l.append(mae_ll)
    #         y_pred = ela.predict(x_test)
    #         mae_ll = np.mean(mae(y_test, y_pred))
    #         mae_l.append(mae_ll)
    #         y_pred = las.predict(x_test)
    #         mae_ll = np.mean(mae(y_test, y_pred))
    #         mae_l.append(mae_ll)
    #     print(mae_l)
    #     list.append(np.mean(mae_l))
    # print(list)
    # print(len(list))
    # print(min(list))

    # for i in range(20,30):
    #     scores = cross_val_score(lre, x_data.iloc[:, :i], y_data, cv=tscv, scoring='neg_mean_absolute_error')
    #     scores_gap = cross_val_score(lre,x_data.iloc[:, :i],y_data,cv=gcv,scoring='neg_mean_absolute_error')
    #     print("TSCV:")
    #     print(abs(np.mean(scores)))
    #     print("GapWalk:")
    #     print(abs(np.mean(scores_gap)))

    # lre = LinearRegression()
    # lre.fit(x_train,x_test)
    from sklearn.feature_selection import RFECV
    method = [lre,ela,las,rid]
    # method = [lre]
    k=0
    mae2 =[]
    cmm =[[],[],[],[]]
    for i in method:
        rfe = RFECV(estimator=i,step=1,cv=gcv).fit(x_data,y_data)
        k += 1
        sup = rfe.support_
        result = rfe.cv_results_
        col = x_data.columns[sup]

        x_new = x_data[col]
        cmm[k-1] = col
        # print(x_data.columns[sup])
        # print(x_new.head())
        print(rfe.estimator_)
        print(x_data[col].shape)


        mae1 = []
        for train,test in  gcv.split(range(x_new.shape[0])):
            x_train = x_new.iloc[train]
            x_test = x_new.iloc[test]
            y_train = y_data.iloc[train]
            y_test = y_data.iloc[test]

            for j in method:
                j.fit(x_train,y_train)
                y_pred = j.predict(x_test)
                mae1.append(mae(y_test,y_pred))
        mae2.append(np.mean(mae1))

    print(mae2.index(min(mae2)),min(mae2))
    feature = cmm[mae2.index(min(mae2))]
    x_select = x_data[feature]
    cor = x_select.corr()
    sns.heatmap(cor)
    # x_select.to_excel('x_select1.0.xlsx',index=True)
    print(x_select.shape)
    vif = [variance_inflation_factor(x_select.values, x_select.columns.get_loc(i)) for i in x_select.columns]
    print(list(zip(list(range(1, 21)), vif)))



        # print(rfe.support_)
        # print(rfe.ranking_)
        # print(rfe.n_features_)
        # print(rfe.n_features_in_)
        # print(rfe.feature_names_in_)

        # print('Results:')
        # print(np.mean(rfe.grid_scores_[sup]),'\n')

        # print(type(result))
        # print(result.keys())
        # print(np.mean(result['mean_test_score'][sup]))
        # print(result['split0_test_score'][sup])
        # print(result['split1_test_score'][sup])
        # print(result['split2_test_score'][sup])
        # print(result['split3_test_score'][sup])

        # print(len(result))
    # Plot number of features VS. cross-validation scores
    #     plt.figure(k)
    #     plt.subplot(2,2,k)
    #     plt.xlabel("Number of features selected")
    #     plt.ylabel("Cross validation score (accuracy)")
    #     plt.plot(
    #         range(1, len(rfe.grid_scores_) + 1),
    #         rfe.grid_scores_,
    #     )
    #
    # plt.show()












    # d = get_x_by_db_name('x_sw_coal_features')
    # d = d[d.index>'2015-01-01']
    # d = d.resample('M').last()
    # d.to_excel('x_sw_coal_features.xlsx')


    # d = get_x_by_db_name('y_ppi')
    # d = d[d.index>'2015-01-01']
    # d = d.resample('M').last()
    # d.to_excel('y_ppi.xlsx')
