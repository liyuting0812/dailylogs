import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from tscv import GapWalkForward
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
import matplotlib.ticker as ticker



def loc_zq_collection(db_name, collection_name):
    client = MongoClient("mongodb://192.168.1.9:27017")
    coll = client[db_name][collection_name]
    return coll

def get_df_by_dict(s):
    re = pd.DataFrame(s['data'], index=s['index'], columns=s['columns'])
    re.sort_index(ascending=True, inplace=True)
    re.index = pd.to_datetime(re.index)
    print(s['columns'])
    return re.iloc[:, 0]


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

def get_y_data(db_name,y_name):
    data = get_x_by_db_name(db_name)
    data = data.loc[:,y_name]
    end_time = data[data.notnull()].index[-1]
    end_time = end_time.strftime("%Y-%m-%d")
    print(end_time)
    print('Y Done!')
    return data



def data_processing(x):

    x = x[x.index > '2015-01-01']
    '''
    可自行修改和添加初筛条件，包括起始时间，提取行数等
    '''
    x = x.resample('M').last()
    x.index = [i.strftime("%Y-%m-%d") for i in x.index]

    x_data = x.iloc[:72]

    print('X has been processed!')
    return x_data

def direction(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def bo_corr(seri, y):
    return seri[seri * y >= 0].count() / len(seri)

def zero_count(x):
    return x[x == 0].count()

def bo_zero(x_data,y_data):
    dire_list = [list(map(direction, list(x_data.iloc[i + 1] - x_data.iloc[i]))) for i in range(x_data.shape[0] - 1)]
    dire_data = pd.DataFrame(dire_list, columns=x_data.columns)
    y = [direction(y_data.iloc[i + 1] - y_data.iloc[i]) for i in range(y_data.shape[0] - 1)] #差值少一期
    zero_data = pd.Series([zero_count(dire_data.iloc[:, i]) for i in range(dire_data.shape[1])],index=dire_data.columns)
    bo_data = [bo_corr(dire_data.iloc[:, i], y) for i in range(dire_data.shape[1])]
    bo_data = pd.Series(bo_data, index=dire_data.columns)
    score = pd.DataFrame(index=dire_data.columns)
    score['bo_score'] = bo_data
    score['zero_num'] = zero_data
    sc = score[score['zero_num'] > 10]

    tick_spacing = 12
    fig = plt.figure()
    for i in range(len(sc)):
        ax = fig.add_subplot(int(len(sc)/2)+1,2,i+1)
        ax.plot(x_data.index,x_data[sc.index[i]],label = sc.index[i])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.tick_params(axis='x', labelsize=8, rotation=30)
        plt.legend()

    plt.savefig('M_data/fig.pdf')
    print('0值过多：',sc)

    return score








if __name__ == '__main__':

    data = get_x_by_db_name('x_sw_BasicChemical')

    data = data_processing(data)

    '''
    通过bo_zero找出剔除数据，del 剔除
    '''
    # del data['出厂价:煤制石脑油(组分柴油):陕西华航']
    # del data['出厂价:渣油:东营东明化工']
    # del data['最高零售价:汽油(标准品):天津']

    data.to_excel('M_data/x_sw_BC.xlsx')
    y_data = get_y_data('y_ppi','PPI:化学原料及化学制品制造业:当月同比')
    y_data = data_processing(y_data)
    y_data.to_excel('M_data/y_sw_BC.xlsx')

    score = bo_zero(data,y_data)
    score['bo_score'].to_excel('M_data/Chemical_bo_score.xlsx')
    print(score)
