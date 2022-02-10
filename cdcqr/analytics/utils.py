import matplotlib.pyplot as plt
from cdcqr.ct.utils import plot2, rollingcorr
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import os
from cdcqr.common.config import LOCAL_FIGURE_DIR
from scipy.stats import pearsonr



pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


def rollingcorr(df, window, col=None, min_periods=None, method='spearman'):
    if method =='pearson':
        return df.rolling(window).corr(col)
    elif method=='spearman':
        return rolling_rank_corr(df, window, y = col, min_periods=1)
    else:
        raise ValueError(f"method {method} not implemented")

pd.core.frame.DataFrame.rollingcorr=rollingcorr


def rolling_spearman_quick(seqa, seqb, window):
    seqa =seqa.values
    seqb =seqb.values
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)


def rolling_rank_corr(df, window, y = None, min_periods=1):
    if y is None:
        n, p = df.shape
        x = df.values

        # create rolling 3d array
        mat_3d = as_strided(x, shape=(n-window+1, window, p), strides=(x.strides[0], x.strides[0], x.strides[1]))

        # apply spearman corr
        res = np.array([libalgos.nancorr_spearman(x) for x in mat_3d])

        # produce missing array
        a = np.empty((window-1, p, p,))
        a[:] = np.nan

        # concat
        res2 = np.concatenate((a, res))

        # create index
        t = [[i]*p for i in df.index]
        flat_list1 = [item for sublist in t for item in sublist]
        t2 = [df.columns]*n
        flat_list2 = [item for sublist in t2 for item in sublist]
        arrays = [flat_list1, flat_list2]
        midx = pd.MultiIndex.from_arrays(arrays)

        return pd.DataFrame(res2.reshape(n*p, p), index=midx, columns=df.columns)
    else:
        ret_dict = {}
        for col in df.columns:
            ret_dict[col] = rolling_spearman_quick(df[col], y, window=window)
        return pd.DataFrame(ret_dict, index=df.index)


def ts_eda(dff0, x, xn, y, lbw = 50, save_fig=False):
    dff = dff0.copy()
    # double y plot

    dff['xn_pctrank'] = dff[xn].rolling(lbw).apply(pctrank)
    dff['xn_zscore'] = zscore(dff[xn], lbw)

    dff.plot2(x, y)
    #dff.plot2(xn, y)
    #dff.plot2('xn_pctrank', y)
    #dff.plot2('xn_zscore', y)

    n_f = 3
    n_fret = 3
    n_group = 4
    fig = plt.figure(num=1, figsize=(6 * n_f, 4 * n_fret*n_group))
    dff['yr'] = dff[y].pct_change()
    dff['yr+10'] = dff[y].pct_change(periods=10).shift(-10)
    dff['yr+100'] = dff[y].pct_change(periods=100).shift(-100)
    dff['yr+1000'] = dff[y].pct_change(periods=1000).shift(-1000)
    for i, feature in enumerate([xn,'xn_pctrank','xn_zscore']):
        for j, forward_ret in enumerate(['yr+10','yr+100','yr+1000']):
            
            lower_bound1 = dff[xn].quantile(0.05)
            upper_bound1 = dff[xn].quantile(0.95)
            lower_bound2 = dff[xn].quantile(0.01)
            upper_bound2 = dff[xn].quantile(0.99)


            upper95_99 = dff[((dff[xn] >= upper_bound1) & (dff[xn] < upper_bound2))]
            upper99 = dff[(dff[xn] >= upper_bound2)]
            lower1 = dff[(dff[xn] <= lower_bound2)]
            lower1_5 = dff[((dff[xn] > lower_bound2) & (dff[xn] <= lower_bound1))]
            #print(lower_bound1, lower_bound2, upper_bound1, upper_bound2)
            #print(lower1.shape, lower1_5.shape, upper95_99.shape, upper99.shape)
            group_dict={}
            group_dict[0] = 'lower1'
            group_dict[1] = 'lower1_5'
            group_dict[2] = 'upper95_99'
            group_dict[3] = 'upper99'
            for k, data in enumerate([lower1, lower1_5, upper95_99, upper99]):
                ax = fig.add_subplot(n_fret * n_f, n_group, i * n_fret * n_group + j * n_group + k + 1)
                #data.plot.scatter(x='f_lagged', y='yr', c='DarkBlue', s=2, ax=ax)

                dflr = data[[feature, forward_ret]].dropna()

                x1 = '{}_{}'.format(feature, group_dict[k])
                y1 = forward_ret
                f_ret = dflr[y1]
                dflr.columns = [x1, y1]
                if dflr.shape[0]>0:
                    lr=LinearRegression(fit_intercept=False).fit(dflr[[x1]],f_ret)
                    lri=LinearRegression(fit_intercept=True).fit(dflr[[x1]],f_ret)
                    sns.scatterplot(x=x1,y=y1,data=dflr, ax = ax)
                    ax.plot(dflr[x1],lr.predict(dflr[[x1]]),color='g')
                    ax.plot(dflr[x1],lri.predict(dflr[[x1]]),color='r')
                    plt.title(f"r {lr.coef_[0]:.3f}x R2={r2_score(dflr[[y1]],lr.predict(dflr[[x1]])):.3f} \
                        \n g {lri.coef_[0]:.3f}x+{lri.intercept_:.3f} R2={r2_score(dflr[[y1]],lri.predict(dflr[[x1]])):.3f} \
                        \n E(r)={dflr[y1].mean():.5f}")
                else:
                    sns.scatterplot(x=x1,y=y1,data=dflr, ax = ax)
                    plt.title('No data available')


                #plt.title('{} lag {} {} v.s. {}'.format(feature, lag, group_dict[k],'{} return'.format(y)), )

    fig.tight_layout()
    if save_fig:
        fig_name = '{}_{}'.format(x,y)
        fig_path = os.path.join(LOCAL_FIGURE_DIR, '{}.png'.format(fig_name))
        fig.savefig(fig_path)    
        print('{} saved'.format(fig_path))

    if True:
        # return scatter plots
        f = x 
        u = y
        dff[f] = dff[f].rolling(20, min_periods=1).mean()
        ret = dff.pct_change()
        ret = ret[ret[f].notna()]
        ret = ret[ret[f] != 0]
        

        #lag corr plot
        lag2mean_corr = {}
        for lag in range(0, 60):
            lag2mean_corr[lag] = ret[f].to_frame().rollingcorr(window=10, col=ret[u].shift(lag)).mean().values[0]
        pd.DataFrame(lag2mean_corr, index=['mean_corr']).T.plot()
        plt.title('avg rank corr v.s. signal lag')
        plt.xlabel('# lags')
        plt.ylabel('average rank corr')
        fig.tight_layout()
        return ret


def calculate_corr_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

