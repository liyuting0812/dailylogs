import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_selection import RFECV
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

x = pd.read_excel(f'features/ppi.xlsx', sheet_name='x', index_col=0)
y = pd.read_excel(f'features/ppi.xlsx', sheet_name='y', index_col=0)

def fit_x_y(x, y, plot=False, iter_=0):
    start_t = '2012-01-31'
    end_t = '2019-07-31'

    test_end = '2021-08-31'
    m_x = x.resample("M").last()
    x_train = m_x.loc[start_t:end_t, :]
    y_train = y.loc[start_t:end_t]

    x_fit = m_x.loc[start_t:test_end, :]
    y_fit = y.loc[start_t: test_end]

    md_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1], cv=10, fit_intercept=False)
    cv = md_cv.fit(x_fit, y_fit)

    md = Ridge(alpha=cv.alpha_, fit_intercept=False)

    md.fit(x_train, y_train, )

    y_p = md.predict(m_x.loc[end_t:test_end])
    y_y = y.loc[end_t:test_end].values
    v_mae = mae(y_y, y_p)

    coef = pd.DataFrame(md.coef_, columns=x.columns)

    if plot:
        y_real = y_fit.iloc[:, 0]
        y_pred = pd.Series(md.predict(x_fit).squeeze(), index=y_real.index, name='y_pred')
        fig = Figure(figsize=(20, 10), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(y_real.to_list(), label='y_real')
        ax.plot(y_pred.to_list(), label='y_pred')
        ax.legend()
        ax.set_title(v_mae)
        fig.savefig(f"iter{iter_}.png")
    return v_mae, coef


cycle = 1
while 1:
    v_, coe = fit_x_y(x, y)
    old_num = x.shape[1]
    print('x.shape', x.shape)
    cols = coe[coe > 0].dropna(axis=1).columns.to_list()
    new_num = len(cols)
    x = x.loc[:, cols]
    print('after x.shape', x.shape, 'mae',v_)
    if new_num == old_num:
        v_, coe = fit_x_y(x, y, True, cycle)
        break
    else:
        cycle += 1
