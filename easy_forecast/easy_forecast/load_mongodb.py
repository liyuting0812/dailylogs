import pymongo
import pandas as pd
import os
from sklearn.cluster import KMeans


def load_my_ifind_tb_data():
    # 设置MongoDB连接信息
    client = pymongo.MongoClient('192.168.1.9', 27017)
    time_series = client['time_series']
    # names = time_series.collection_names()
    wind_metadata = time_series['wind_metadata']
    data = pd.DataFrame(list(wind_metadata.find()))
    return data


def load_ifind_wind_data(k='ifind'):
    # 设置MongoDB连接信息
    client = pymongo.MongoClient('192.168.1.9', 27017)
    print('connect 192.168.1.9')
    time_series = client['time_series']
    # names = time_series.collection_names()
    ifind = time_series[k]
    data = pd.DataFrame(list(ifind.find()))
    data.index = data['_id']
    data = data.iloc[:, 1:]
    dt = data.T
    dt.sort_index(inplace=True)
    return dt


def filter_data(data, k='ifind_data'):
    dt = data[data.table_name == k]  # 只要ifind数据
    dt = dt[dt.frequency == '日']  # 只要日度数据
    ft_name = [True if '价' in s else False for s in dt.name]
    dt = dt[ft_name]
    ft_name = [True if '期货' not in s else False for s in dt.name]
    dt = dt[ft_name]
    ft_name = [True if s > '2021-09-01' else False for s in dt.latest_date]
    dt = dt[ft_name]
    return dt


def filter_nan(data):
    nan_number = data.isnull().sum(axis=0)
    v = nan_number / len(data)
    v = v[v < 0.5]
    return v


def get_wind_data():
    print('加载wind数据表')
    data = load_my_ifind_tb_data()
    print('过滤不合适的数据')
    data = filter_data(data, 'wind_data')
    wind_abstract = data.copy()
    wind_abstract.index = wind_abstract.index_id
    name_dic = wind_abstract.name.to_dict()

    print('加载wind数据')
    wind_data = load_ifind_wind_data('wind')
    wind_data = wind_data.loc[:, data.index_id]
    wind_data = filter_nan(wind_data)
    wind_data = wind_data.loc[:, filter_data.index]
    wind_data = wind_data[wind_data.index > '2010-12-31']
    print('替换数据列名为字符串')
    wind_data.columns = [name_dic[s] for s in wind_data.columns]
    return wind_abstract, wind_data


def get_ifind_data():
    print('加载ifind数据表')
    data = load_my_ifind_tb_data()
    print('过滤不合适的数据')
    data = filter_data(data)
    ifind_abstract = data.copy()
    ifind_abstract.index = ifind_abstract.index_id
    name_dic = ifind_abstract.name.to_dict()

    print('加载同花顺数据')
    ifind_data = load_ifind_wind_data()
    ifind_data = ifind_data.loc[:, data.index_id]
    filter_data = filter_nan(ifind_data)
    ifind_data = ifind_data.loc[:, filter_data.index]
    ifind_data = ifind_data[ifind_data.index > '2010-12-31']
    print('替换数据列名为字符串')
    ifind_data.columns = [name_dic[s] for s in ifind_data.columns]
    return ifind_abstract, ifind_data


def k_cluster(dt):
    print('正在聚类')
    dt = dt.fillna(method='ffill').fillna(method='bfill')
    cols = dt.columns.to_list()
    km = KMeans(n_clusters=500, init='random', random_state=28)
    km.fit(dt.T)
    y_hat = km.predict(dt.T)
    y_cluster = {}
    for i, i_cluster in enumerate(y_hat):
        v = y_cluster.setdefault(i_cluster, [])
        v.append(i)
    ctr = []
    for i_cluster, v in y_cluster.items():
        vv = [cols[s] for s in v]
        se = pd.Series(vv, name=i_cluster)
        ctr.append(se)
    y_cluster_df = pd.concat(ctr, axis=1)
    return y_cluster_df


def choose_by_cluster(y_cluster_df, data):
    fit_li = []
    for i_cluster, se in y_cluster_df.iteritems():
        se = se.dropna()
        temp_data = data.loc[:, se]
        best = temp_data.isnull().sum(axis=0).sort_values().index[0]
        fit_li.append(temp_data[best])
    re = pd.concat(fit_li, axis=1)
    return re


def get_daily_yoy(data):
    print('计算同比数据')
    data = pd.DataFrame(data, dtype='float')
    data = data.interpolate(method='linear')
    data = data.fillna(method='bfill')
    se_dic = {}
    data.index = pd.to_datetime(data.index)
    for k, v in data.resample('M'):
        new_k = str(int(str(k)[:4])) + str(k.month)
        se_dic[new_k] = v.mean()
    day_dic = {}
    for k, v in data.iterrows():
        t_ll = str(int(str(k)[:4]) - 1) + str(k.month)
        if t_ll in se_dic:
            v_yoy = v / se_dic[t_ll] - 1
            day_dic[k] = v_yoy
    daily_data = pd.DataFrame(day_dic).T
    return daily_data


def get_ppi_columns():
    print('load local ppi features')
    cols = pd.read_excel('ppi/ppi_features.xlsx', index_col=0)
    return cols.iloc[:, 0].to_list()


if __name__ == "__main__":
    # abs_data, ifind_data = get_ifind_data()
    wind_abs, wind_data = get_wind_data()
    # if os.path.exists('ppi/ppi_features_ifind.xlsx'):
    #     ppi_c = get_ppi_columns()
    #     data = ifind_data.loc[:, ppi_c]
    # else:
    #     y_cluster_df = k_cluster(ifind_data)
    #     data = choose_by_cluster(y_cluster_df, ifind_data)
    # daily_yoy = get_daily_yoy(data)
    # daily_yoy.to_excel('ppi/ppi_ifind.xlsx')
