import logging
LOG_FILENAME = '/core/logs/backtest.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',filename=LOG_FILENAME)
from cdcqr.backtest.vbt.runsignal import runsignal
from cdcqr.common.utils import parallel_jobs
import numpy as np
from cdcqr.common.utils import LOCAL_DATA_DIR, dict_hash
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from cdcqr.common.utils import LOCAL_DATA_DIR
import json
from cdcqr.data.glassnode.features.feature_loaders import produce_feature_BTCSSR


pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]

# data 

# signal generation
signal = pd.read_pickle('/core/tmp/sample_signal.pickle')
signalid = 'long-BTCSSR_pctrank'
price = pd.read_pickle(os.path.join(LOCAL_DATA_DIR,'BTC-PERP@ftx.pickle'))
short=False
asset = 'BTC'
side = 'long'
feature = 'BTCSSR'


if __name__ == '__main__':
    
    feature_data_fname = 'gn_{}'.format(feature)
    feature_data_dir = os.path.join(LOCAL_DATA_DIR, 'feature_data', '{}.pickle'.format(feature_data_fname))
    logging.info('loading feature data {}'.format(feature_data_dir))
    produce_feature_BTCSSR()
    df4a = pd.read_pickle(feature_data_dir)

    backtest_info = {}
    backtest_info['signalid'] = signalid
    backtest_info['side'] = 'long'
    backtest_info_hashed = dict_hash(backtest_info)
    backtest_job_info = '{}_{}'.format(json.dumps(backtest_info), backtest_info_hashed)
    job_start_log_info = 'Starting backtest job {}'.format(backtest_job_info)
    logging.info(job_start_log_info)

    backtest_info = {}
    backtest_info['signalid'] = signalid
    backtest_info['side'] = 'long'
    backtest_info_hashed = dict_hash(backtest_info)

    qtl_params = [0.995, 0.99, 0.975, 0.96]
    lbw_params = [1000, 2000, 3000, 4000]

    tp_params = [3, 5, 7, 10]
    sl_params = [1, 2, 3 , 5 , 7, 10]
    n1_params = [1, 10, 100]
    n2_params = [101, 200, 500, 1000, 2000]


    def partial_runsignal(param_dict):
        tp = param_dict['tp']
        sl = param_dict['sl']
        n1 = param_dict['n1']
        n2 = param_dict['n2']
        qtl = param_dict['qtl']
        lbw = param_dict['lbw']
        price = param_dict['price']
        signal = param_dict['signal']
        side = param_dict['side']

        if side=='long':
            short=False
        elif side =='short':
            short=True
        else:
            raise('side is either long and short')
        ressig=runsignal(price=price,signal=signal,tp=tp,sl=sl,ts=sl,n1=n1,n2=n2,rsil=30,rsih=50,rsilag=14,
                    maf=5,mas=15,short=short,size=np.inf,fees=0.0007,freq='1Min',init_cash=10000,request=['pf','backtest'])
        for k in ressig:
            resd={"signalid":signalid,"short":short,"tp":tp,"sl":sl,'k':k, 'lbw':lbw, 'qtl':qtl,
                'sr':ressig[k].sharpe_ratio(),'n1':n1,'n2':n2,'tr':ressig[k].total_return(),'ntrades':ressig[k].trades.count()/46.5} # rescale to number of weeks
        return resd

    all_ret_list = []
    for qtl in qtl_params:
        for lbw in lbw_params:
            log_info = 'run signal for qtl={}, lbw={}'.format(qtl, lbw)
            logging.info(log_info)
            feature_pctrank = '{}_pctrank{}'.format(feature, lbw)
            df4a[feature_pctrank] = df4a[feature].rolling(lbw).apply(pctrank)
            df4b = df4a[df4a.index>='20210101']
            signal = df4b[feature_pctrank]>qtl # rolling max, keep the singal for a period (1 hour), good for combining with other signals
            signal = signal.reindex(price.index).ffill()
            backtest_configs = []
            for tp in tp_params:
                for sl in sl_params:
                    for n1 in n1_params:
                        for n2 in n2_params:
                            param_dict = {}
                            param_dict['tp'] = tp
                            param_dict['sl'] = sl
                            param_dict['n1'] = n1
                            param_dict['n2'] = n2
                            param_dict['qtl'] = qtl
                            param_dict['lbw'] = lbw
                            backtest_configs.append(param_dict)
                            
            ret = parallel_jobs(partial_runsignal, backtest_configs)
            all_ret_list.extend(ret.values())


    res_df = pd.DataFrame(all_ret_list)
    print(res_df)

    backtest_res_df_fname = '{}_{}_{}'.format(asset,  signalid, backtest_info_hashed)
    res_df.save(os.path.join(LOCAL_DATA_DIR, 'backtest_res', backtest_res_df_fname))
