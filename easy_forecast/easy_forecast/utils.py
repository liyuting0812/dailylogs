import yaml
import os
import pickle
import pandas as pd
from pandas import DataFrame
from tkinter import ttk
from pymongo import MongoClient


def save_pickle(paras, path):
    with open(path, 'wb') as fo:
        pickle.dump(paras, fo)
    return path


def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        cfg_data = yaml.load(f.read(), yaml.FullLoader)
        print(cfg_data.keys())
    print("loading_config ... >>> 读取本地配置")
    return cfg_data


def loc_zq_collection(db_name, collection_name):
    conn = MongoClient("mongodb://192.168.1.9:27017")
    coll_ = conn[db_name][collection_name]
    return coll_


def get_df_by_dict(s: dict):
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


def ini_tree(tree: ttk.Treeview, x: dict) -> dict:
    tree_dict = {}
    for first_i, (k, v) in enumerate(x.items()):
        tree_dict[k] = tree.insert("", first_i, k, text=k, values='1')
        for second_i, col in enumerate(v.columns):
            tree.insert(tree_dict[k], second_i, f"{k}>>{col}", text=col, values='2')
    return tree_dict


def ini_tree_choose(tree_choose: ttk.Treeview, x_df: DataFrame):
    tree_dict = {}
    for first_i, (k, v) in enumerate(x_df.iteritems()):
        tree_dict[k] = tree_choose.insert("", first_i, k, text=k, values='1')


if __name__ == "__main__":
    d = get_x_by_db_name('x_sw_sysh')
    d = d[d.index>'2015-01-01']

    d.to_excel('x_sw_sysh.xlsx')