"""
机器学习计算和可视化平台
"""
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib import gridspec

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error as mae
from utils import load_xy, ini_tree, ini_tree_choose, load_config

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示


class Gui:
    def __init__(self, ):
        super(Gui, self).__init__()
        self.cfg = load_config()
        self.root = tk.Tk()  # 创建主窗体
        self.root.title("欢迎")
        self.root.geometry("1850x1100")
        self.bg = "#F07C82"  # 设置背景色
        self.s_f = ('Microsoft YaHei', 15,)  # second_font, 二级字体

        self.fm_args = {"relief": "ridge", "bd": 3, "bg": self.bg}
        self.fig_show = tk.Frame(self.root, height=830, width=1500, **self.fm_args)
        self.fig_show.place(x=440, y=10)

        self.x, self.y = load_xy(self.cfg)

        self.tree = ttk.Treeview(self.root, height=40, )
        self.tree.place(x=10, y=130)
        self.tree_dict = ini_tree(self.tree, self.x)
        self.tree.bind("<<TreeviewSelect>>", self._tree_fun)

        self.y_name = list(self.x)[0]
        self.x_choose = pd.DataFrame()

        self.tree_choose = ttk.Treeview(self.root, height=40, )
        self.tree_choose.place(x=220, y=130)
        self.tree_dict_choose = ini_tree_choose(self.tree_choose, self.x_choose)
        self.tree_choose.bind("<<TreeviewSelect>>", self._choose_tree_fun)

        self.add = tk.Button(self.root, text="ADD", font=self.s_f, width=6, command=self._add_choose)
        self.add.place(x=300, y=10)
        self.clear = tk.Button(self.root, text="C", font=self.s_f, width=6, command=self._clear)
        self.clear.place(x=100, y=70)
        self.remove = tk.Button(self.root, text="Remove", font=self.s_f, width=6, command=self._remove)
        self.remove.place(x=300, y=70)
        self.fit = tk.Button(self.root, text="FIT", font=self.s_f, width=6, command=self._fit)
        self.fit.place(x=100, y=10)
        self.search_plus = tk.Button(self.root, text=" + + ", font=self.s_f, width=6, command=self._search_plus)
        self.search_plus.place(x=350, y=500)

        self.search_minus = tk.Button(self.root, text=" - ", font=self.s_f, width=6, command=self._search_minus)
        self.search_minus.place(x=350, y=400)

        self.fitted = False
        self.explain = False
        self._ini_fig()

    def _on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)

    def _ini_fig(self):
        self.fig = Figure(figsize=(18, 12), dpi=78)
        self.canvas = FigureCanvasTkAgg(self.fig, self.fig_show)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_show)
        self.toolbar.update()
        self.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.spec = gridspec.GridSpec(ncols=3, nrows=6, figure=self.fig)

    def _tree_fun(self, event):
        self.name = event.widget.selection()[0]
        self._fig_plot()

    def _choose_tree_fun(self, event):
        self.choose_name = event.widget.selection()[0]
        self._fig_plot()

    def _fig_plot(self):
        self.fig.clear()
        self._basic_plot()
        self.canvas.draw()

    def _basic_plot(self):
        ax_main = self.fig.add_subplot(self.spec[4:, :2])
        if '>>' in self.name:
            y_name, x_name = self.name.split('>>')
            idx = sorted([s for s in self.y[y_name].index if s in self.x[y_name].index])
            x_se = self.x[y_name].loc[idx, x_name]
            ax_main.plot(x_se, label=f'x >> {x_name}')
            y_se = self.y[y_name][idx]
            acc_tztd = self._cal_tztd_ratio(x_se, y_se)
            ax_main.set_title(f"{self.name} >>> {acc_tztd}")
        else:
            y_name = self.name
            y_se = self.y[y_name]
        ax_main2 = ax_main.twinx()
        ax_main2.plot(y_se, color='r', label=f'y >> {y_name}')
        ax_main.legend()

        if self.fitted:
            ax_fit = self.fig.add_subplot(self.spec[:2, :2])
            ax_fit.plot(self.y_data_train, label=f'y_train')
            ax_fit.plot(self.y_data_test, label=f'y_test')
            ax_fit.plot(self.y_pred, label=f'y_pred >> {self.y_name}')
            ax_fit.legend()

            ax_error = self.fig.add_subplot(self.spec[2:4, :2])
            error = self.y_data_ - self.y_pred
            ax_error.plot(error, label='error')
            ax_error.legend()

        if self.explain:
            ax_explain = self.fig.add_subplot(self.spec[:, 2:])
            ax_explain.barh(self.x_choose.columns.to_list(), self.h_s)

    def _cal_tztd_ratio(self, x: pd.Series, y: pd.Series) -> float:
        assert x.index[0]==y.index[0] and x.index[-1] == y.index[-1]
        acc_li = []
        for i in range(len(x)-1):
            if (x[i+1] - x[i]) * (y[i+1]-y[i]) >= 0:
                acc_li.append(1)
            else:
                acc_li.append(0)
        return np.mean(acc_li)

    def _add_choose(self):
        x = self.tree_choose.get_children()
        for item in x:
            self.tree_choose.delete(item)
        y_name, x_name = self.name.split('>>')
        if x_name not in self.x_choose.columns:
            if len(self.x_choose) == 0:
                self.x_choose = pd.DataFrame(self.x[y_name][x_name].copy(), columns=[x_name])
            else:
                self.x_choose = pd.concat([self.x[y_name][x_name], self.x_choose], axis=1)
        self.tree_dict_choose = ini_tree_choose(self.tree_choose, self.x_choose)
        self.x_names = self.x_choose.columns

    def _fit(self):
        self.y_name, _ = self.name.split('>>')
        x_data = self.x_choose

        self.y_data = self.y[self.y_name].copy()
        x_data = x_data.loc[self.y_data.index, :]
        self.y_data_ = self.y_data['2015-01-01':]
        self.y_data_train = self.y_data['2015-01-01':'2019-01-01']
        self.y_data_test = self.y_data['2019-01-01':]
        self.x_data = x_data.loc['2015-01-01':, :]
        self.x_data_train = x_data.loc['2015-01-01':'2019-01-01', :]
        self.x_data_test = x_data.loc['2019-01-01':, :]
        md = Ridge(fit_intercept=False)
        md.fit(X=self.x_data_train, y=self.y_data_train)
        print('x_data_train.shape', self.x_data_train.shape)
        print('the w of this model is ', md.coef_)
        print('the intercept of this model is ', md.intercept_)
        y_pred = md.predict(x_data)
        self.y_pred = pd.Series(y_pred, index=x_data.index, name='y_pred')
        self.w = list(md.coef_)
        self.b = md.intercept_
        y_last_month = x_data.iloc[-1, :].to_list()
        self.h_s = [ww * vv for ww, vv in zip(self.w, y_last_month)]

        self.smaller_mae = mae(self.y_pred, self.y_data)
        self.fitted = True
        self.explain = True
        self._fig_plot()

    def _clear(self):
        self.x_choose = pd.DataFrame()
        x = self.tree_choose.get_children()
        for item in x:
            print('item', item)
            self.tree_choose.delete(item)
        self.tree_dict_choose = ini_tree_choose(self.tree_choose, self.x_choose)

    def _remove(self):
        print('delete:', self.choose_name)
        old_cols = self.x_choose.columns.to_list()
        new_cols = [s for s in old_cols if s != self.choose_name]
        self.x_choose = self.x_choose.loc[:, new_cols]
        x = self.tree_choose.get_children()
        for item in x:
            self.tree_choose.delete(item)
        self.tree_dict_choose = ini_tree_choose(self.tree_choose, self.x_choose)

    def _search_plus(self):
        self.y_name, _ = self.name.split('>>')
        all_x = self.x[self.y_name]
        used_cols = self.x_choose.columns.to_list()
        all_cols = all_x.columns.to_list()
        not_used_cols = [s for s in all_cols if s not in used_cols]
        print('use cols', len(used_cols))
        for c in not_used_cols:
            temp_cols = used_cols + [c]
            all_less_x = all_x.loc[:, temp_cols]
            self.y_data = self.y[self.y_name].copy()
            self.y_data_ = self.y_data['2015-01-01':]
            self.y_data_train = self.y_data['2015-01-01':'2019-01-01']
            self.y_data_test = self.y_data['2019-01-01':]
            self.x_data = all_less_x.loc['2015-01-01':, :]
            self.x_data_train = all_less_x.loc['2015-01-01':'2019-01-01', :]
            self.x_data_test = all_less_x.loc['2019-01-01':, :]
            md = Ridge()
            md.fit(X=self.x_data_train, y=self.y_data_train)
            y_pred = md.predict(self.x_data)
            self.y_pred = pd.Series(y_pred, index=self.x_data.index, name='y_pred')
            temp_mae = mae(self.y_pred, self.y_data_)
            if temp_mae < self.smaller_mae:
                print(c, 'mse descending')
                self.smaller_mae = temp_mae
                self.x_choose = all_less_x
        self._fit()
        x = self.tree_choose.get_children()
        for item in x:
            self.tree_choose.delete(item)
        self.tree_dict_choose = ini_tree_choose(self.tree_choose, self.x_choose)

    def _search_minus(self):
        self.y_name, _ = self.name.split('>>')
        all_x = self.x[self.y_name]
        used_cols = self.x_choose.columns.to_list()
        for c in used_cols:
            temp_cols = [s for s in used_cols if s != c]
            all_less_x = all_x.loc[:, temp_cols]
            self.y_data = self.y[self.y_name].copy()
            self.y_data_ = self.y_data['2016-01-01':]
            self.y_data_train = self.y_data['2016-01-01':'2019-01-01']
            self.y_data_test = self.y_data['2019-01-01':]
            self.x_data = all_less_x.loc['2016-01-01':, :]
            self.x_data_train = all_less_x.loc['2016-01-01':'2019-01-01', :]
            self.x_data_test = all_less_x.loc['2019-01-01':, :]
            md = Ridge()
            md.fit(X=self.x_data_train, y=self.y_data_train)
            y_pred = md.predict(self.x_data)
            self.y_pred = pd.Series(y_pred, index=self.x_data.index, name='y_pred')
            temp_mae = mae(self.y_pred, self.y_data_)
            if temp_mae < self.smaller_mae:
                print(c, 'mse descending')
                self.smaller_mae = temp_mae
                self.x_choose = all_less_x
        self._fit()
        x = self.tree_choose.get_children()
        for item in x:
            self.tree_choose.delete(item)
        self.tree_dict_choose = ini_tree_choose(self.tree_choose, self.x_choose)


if __name__ == "__main__":
    G = Gui()
    G.root.mainloop()
    print('done')
