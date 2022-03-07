import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from tscv import GapWalkForward
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error as mae
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.ticker as ticker

class Features_Selection:
    def __init__(self,ypath,xpath):
        self.y_data = pd.read_excel(ypath,index_col=0)
        self.x_data = pd.read_excel(xpath,index_col=0)
        self.gcv = GapWalkForward(n_splits=5,gap_size=2,test_size=6)




    def gridSearch(self,i):
        best_ting = {'alpha': [0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100]}
        best_g = GridSearchCV(i,best_ting,cv=self.gcv)
        best_g.fit(self.x_data,self.y_data)
        return best_g.best_estimator_

    def alpha_select(self):
        method = [Lasso(),Ridge(),ElasticNet()]
        print([self.gridSearch(i) for i in method])
        return [self.gridSearch(i) for i in method]

    def get_feature_rfe(self):
        method = self.alpha_select()
        method.append(LinearRegression())

        k = 0
        mae2 = []
        cmm = [[], [], [], []]
        for i in method:
            rfe = RFECV(estimator=i,step=1,cv=self.gcv).fit(self.x_data,self.y_data)
            k += 1
            sup = rfe.support_
            col = self.x_data.columns[sup]
            x_new = self.x_data[col]
            cmm[k-1] = col
            print(rfe.estimator_)
            print(self.x_data[col].shape)

            mae1 = []
            for train, test in self.gcv.split(range(x_new.shape[0])):
                x_train = x_new.iloc[train]
                x_test = x_new.iloc[test]
                y_train = self.y_data.iloc[train]
                y_test = self.y_data.iloc[test]
                for j in method:
                    j.fit(x_train, y_train)
                    y_pred = j.predict(x_test)
                    mae1.append(mae(y_test, y_pred))
            mae2.append(np.mean(mae1))
        return mae2,cmm

    def maeVif(self):
        method = self.alpha_select()
        method.append(LinearRegression())
        mean = []
        mae,cmm = self.get_feature_rfe()
        for i in range(len(mae)):
            fig = plt.figure(figsize=(7, 7))
            fig.title = method[i]
            feature = cmm[i]
            x_select = self.x_data[feature]
            cor = x_select.corr()
            mask = np.zeros_like(cor)
            mask[np.tril_indices_from(mask)] = True

            vif = [variance_inflation_factor(x_select.values, x_select.columns.get_loc(i)) for i in x_select.columns]
            vif_table = pd.DataFrame(vif, index=x_select.columns)
            vif_table.columns = ['VIF']
            mean.append(np.mean(vif_table['VIF']))
            if np.mean(vif_table['VIF']) < 100:
                sns_plot = sns.heatmap(cor, annot=True, cmap='Blues', mask=mask.T)

        mae_vif = pd.DataFrame(mae)
        mae_vif['vif'] = mean
        mae_vif.columns = (['mae', 'vif'])
        mae_vif['score'] = mae_vif['mae'] * 0.5 + mae_vif['vif'] * 0.5
        sco = mae_vif['score']
        in_ = (sco[sco ==min(sco)].index)[0]

        print(mae_vif)
        print(cmm[in_])
        print(method[in_])
        return method[in_],cmm[in_]

    def Plot(self):
        model,features = self.maeVif()
        data = self.x_data[features]
        num = self.x_data.shape[0]-13
        x_train = self.x_data.iloc[:num]
        x_test = self.x_data.iloc[num:]
        y_train = self.y_data.iloc[:num]
        y_test = self.y_data.iloc[num:]
        model.fit(x_train,y_train)
        y_pre = model.predict(x_test)
        y_pre = pd.Series(y_pre, index=y_test.index)
        mae_ = mae(y_test,y_pre)
        y_con = pd.concat([y_train,y_pre],axis=0)
        print(self.y_data.index)

        bo_data = pd.read_excel('M_data/bo_score.xlsx', index_col=0)

        fig = plt.figure(figsize=(25, 45), dpi=80)
        tick_spacing = 12
        for i in range(data.shape[1] + 1):
            if i == data.shape[1]:
                ax = fig.add_subplot(7, 2, i + 1)
                ax.set_title(str(model) + '   >>>  MAE: ' + str(mae_))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                ax.tick_params(axis='x', labelsize=8, rotation=30)
                ax.plot(self.y_data.index, self.y_data.values, label='y')
                ax.plot(y_test.index, y_pre.values, label=model)


                plt.axvline(x=self.x_data.index[num], ls="--", c="green")
                plt.legend()

            else:

                ax = fig.add_subplot(7, 2, i + 1)
                ax.set_title(features[i] + '    >>>   ' +str((bo_data.loc[features[i]].values)[0]))
                ax.tick_params(axis='x', labelsize=8, rotation=30)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

                ax.plot(self.y_data.index, self.y_data.values, label='y')
                ax.plot(data.index, data.iloc[:, i])

                plt.legend()

        plt.savefig('M_data/sy_BC_feature '+str(model) + '.pdf', format='pdf')


if __name__ == '__main__':
    F = Features_Selection('M_data/y_sw_BC.xlsx','M_data/x_sw_BC.xlsx')

    F.Plot()
