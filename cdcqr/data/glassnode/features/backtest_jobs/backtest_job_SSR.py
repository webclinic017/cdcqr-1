import logging
LOG_FILENAME = '/core/logs/backtest.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',filename=LOG_FILENAME)
from cdcqr.common.utils import parallel_jobs2
from cdcqr.common.utils import LOCAL_DATA_DIR, dict_hash
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from cdcqr.common.utils import LOCAL_DATA_DIR
import json
from cdcqr.data.glassnode.features.feature_loaders import produce_feature_BTCSSR
from cdcqr.backtest.vbt.runsignal import wrapped_runsignal
from cdcqr.backtest.vbt.sample_params import param_dict
import itertools
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]


if __name__ == '__main__':
    # load price
    price = pd.read_pickle(os.path.join(LOCAL_DATA_DIR,'BTC-PERP@ftx.pickle'))
    
    #load feature
    asset = 'BTC'
    feature = 'BTCSSR'
    signalid = 'long-BTCSSR'
    feature_data_fname = produce_feature_BTCSSR()
    feature_data_dir = os.path.join(LOCAL_DATA_DIR, 'feature_data', '{}.pickle'.format(feature_data_fname))
    logging.info('loading feature data {}'.format(feature_data_dir))
    df4a = pd.read_pickle(feature_data_dir)

    param_dict_raw = param_dict.copy()
    qtl_params = [0.995, 0.99, 0.975, 0.96]
    lbw_params = [1000, 2000, 3000, 4000]

    test=True
    if test:
        tp_params = [3, 5] # 
        sl_params = [1, 2] # ts = sl 
        n1_params = [1, 10]
        n2_params = [101]
        rsil_params = [10]
        rsilag_params = [14]
        maf_params = [5]
        mas_params = [100, 200]
    else:
        tp_params = [3, 5, 7, 10] # 
        sl_params = [1, 2, 3 , 5] # ts = sl 
        n1_params = [1, 10, 100]
        n2_params = [101, 200, 500, 1000, 2000]
        rsil_params = [10, 20] # 2 runs with rsil rsilag
        rsilag_params = [14]
        maf_params = [5, 60] # 4 runs for maf/mas
        mas_params = [100, 200]

    backtest_info = {}
    backtest_info['asset'] = asset
    backtest_info['signalid'] = signalid
    backtest_info['side'] = 'long'
    backtest_info['qtl_params'] = qtl_params
    backtest_info['lbw_params'] = lbw_params
    backtest_info['tp_params'] = tp_params
    backtest_info['sl_params'] = sl_params
    backtest_info['n1_params'] = n1_params
    backtest_info['n2_params'] = n2_params
    
    backtest_info['rsil_params'] = rsil_params
    backtest_info['rsilag_params'] = rsilag_params
    backtest_info['maf_params'] = maf_params
    backtest_info['mas_params'] = mas_params


    backtest_info_hashed = dict_hash(backtest_info)
    backtest_job_info = '{}_{}'.format(json.dumps(backtest_info), backtest_info_hashed)
    job_start_log_info = 'Starting backtest job {}'.format(backtest_job_info)
    logging.info(job_start_log_info)


    logging.info('generating backtest configs')
    backtest_configs = []
    rsi_or_ma=['rsi', 'ma']
    for lbw in lbw_params:
        feature_pctrank = '{}_pctrank{}'.format(feature, lbw)
        df4a[feature_pctrank] = df4a[feature].rolling(lbw).apply(pctrank)
        df4b = df4a[df4a.index>='20210101']
        for qtl in qtl_params:
            signal = df4b[feature_pctrank]>qtl # rolling max, keep the singal for a period (1 hour), good for combining with other signals
            signal = signal.reindex(price.index).ffill()
            for tp in tp_params:
                for sl in sl_params:
                    for n1 in n1_params:
                        for n2 in n2_params:
                            for rsil in rsil_params:
                                rsih = 100-rsil
                                for maf in maf_params:
                                    for mas in mas_params:
                                        for railag in rsilag_params:
                                            
                                            param_dict = param_dict_raw.copy()
                                            param_dict['qtl'] = qtl
                                            param_dict['lbw'] = lbw
                                            param_dict['signal'] = signal
                                            param_dict['price'] = price.astype(np.float32)
                                            param_dict['tp'] = tp
                                            param_dict['sl'] = sl
                                            param_dict['n1'] = n1
                                            param_dict['n2'] = n1 + n2
                                            param_dict['rsil'] = rsil
                                            param_dict['rsih'] = rsih
                                            param_dict['maf'] = maf
                                            param_dict['mas'] = mas
                                            param_dict['railag'] = railag
                                            
                                            backtest_configs.append(param_dict)


    logging.info('running {} backtest jobs'.format(len(backtest_configs)))
    ret = parallel_jobs2(wrapped_runsignal, backtest_configs, num_process=10)
    all_ret_list = list(itertools.chain.from_iterable(ret.values()))

    logging.info('saving res df')
    res_df = pd.DataFrame(all_ret_list)

    backtest_res_df_fname = '{}_{}_{}'.format(asset,  signalid, backtest_info_hashed)
    res_df.save(os.path.join(LOCAL_DATA_DIR, 'backtest_res', backtest_res_df_fname))
