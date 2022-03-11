import logging
LOG_FILENAME = '/core/logs/backtest.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',filename=LOG_FILENAME)
from cdcqr.common.utils import parallel_jobs2
from cdcqr.common.utils import LOCAL_DATA_DIR, dict_hash
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from cdcqr.common.utils import LOCAL_DATA_DIR
import json
from cdcqr.data.glassnode.features.feature_loaders import produce_feature_BTCSSR
from cdcqr.backtest.vbt.runsignal import wrapped_runsignal
from cdcqr.backtest.vbt.sample_params import param_dict
import itertools
import numpy as np
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
from cdcqr.data.glassnode import glassnode_data as gnd


def getsignal3(glass, price0, liq0, qtl=0.99, liql=20, priceq=0.0, qtl_w=0):
    glass['99q'] = glass['v'].rolling(60*24).quantile(qtl)
    glass['signal'] = (glass['v']>=glass['99q'])
    price0['glass_signal'] = glass['signal']
    price0['glass_signal'] = price0['glass_signal'].fillna(False)
    
    liq = liq0.resample("1Min").last().fillna(0)
    liq['spikes'] = liq[liq.v>liql]['v']
    liq['spikes'] = liq['spikes'].fillna(0).rolling(100).max().fillna(0)
    price0['liq_signal'] = liq['spikes'][liq['spikes']>liql]
    price0['liq_signal'] = price0['liq_signal'].fillna(False)
    
    if qtl_w == 0:
        price0['cq_signal'] = True
    else:
        price0['cq'] = price0['c'].rolling(qtl_w).quantile(priceq).resample("1Min").last().ffill()
        price0['cq_signal'] = (price0['c'] >= price0['cq']).fillna(False)

    price0['s'] = price0['glass_signal'] & price0['liq_signal'] & price0['cq_signal']
    return price0[['c','glass_signal','liq_signal','cq_signal','s']]


if __name__ == '__main__':
    # load data
    #for asset in ['LINK','CRV','SUSHI','AAVE','MKR','MATIC']:
    for asset in ['SUSHI']:
 
        signalid = 'short-{}liq'.format(asset)

        logging.info('backtesting  {}'.format(signalid))

        price0 = pd.read_pickle('/core/tmp/ppt_prc_{}_1m_20210101_20220228.pkl'.format(asset))
        liq0 = pd.read_pickle('/core/tmp/ppt_liqb_{}_1m_20210101_20220228.pkl'.format(asset))

        df = pd.read_pickle('/core/tmp/glass_balexch_{}_10m_20210101_20220131.pickle'.format(asset))
        glass = df.copy().resample("1Min").last().fillna(0)
        glass.columns = ['v']

        param_dict_raw = param_dict.copy()
        qtl_params = [0.995, 0.99, 0.975]
        #liql_params =  [5, 10, 20, 50]
        liql_params =  [75, 100, 200]
        priceq_params = [0]
        qtlw_params = [0]
        
        test=False
        if test:
            tp_params = [3, 5] # 
            sl_params = [1, 2] # ts = sl 
            n1_params = [1, 10]
            n2_params = [101]
            rsil_params = [10]
            rsilag_params = [14]
            maf_params = [5]
            mas_params = [15]
        else:
            tp_params = [3, 5, 7, 10] # 
            sl_params = [1, 2, 3 , 5] # ts = sl 
            n1_params = [None]
            n2_params = [None]
            rsil_params = [30] # 2 runs with rsil rsilag
            rsilag_params = [14]
            maf_params = [5, 60] # 4 runs for maf/mas
            mas_params = [15, 100, 200]

        backtest_info = {}
        backtest_info['asset'] = asset
        backtest_info['signalid'] = signalid
        backtest_info['side'] = 'short'
        backtest_info['qtl_params'] = qtl_params
        backtest_info['priceq_params'] = priceq_params
        backtest_info['qtlw_params'] = qtlw_params
        backtest_info['liql_params'] = liql_params
        backtest_info['tp_params'] = tp_params
        backtest_info['sl_params'] = sl_params
        backtest_info['n1_params'] = n1_params
        backtest_info['n2_params'] = n2_params
        
        backtest_info['rsil_params'] = rsil_params
        backtest_info['rsilag_params'] = rsilag_params
        backtest_info['maf_params'] = maf_params
        backtest_info['mas_params'] = mas_params

        request = ['pf','backtest']

        backtest_info_hashed = dict_hash(backtest_info)
        backtest_job_info = '{}_{}'.format(json.dumps(backtest_info), backtest_info_hashed)
        job_start_log_info = 'Starting backtest job {}'.format(backtest_job_info)
        logging.info(job_start_log_info)

        short = True
        logging.info('generating backtest configs')
        backtest_configs = []
        price = price0
        for qtl in qtl_params:
            for liql in liql_params:
                for priceq in priceq_params:
                    for qtlw in qtlw_params:
                        #print(qtl, liql, priceq, qtlw)
                        signal = getsignal3(glass, price, liq0, qtl,liql,priceq,qtlw)['s']
                        num_weeks = (signal.index.max() - signal.index.min()).days/7
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
                                                        param_dict['liql'] = liql
                                                        param_dict['priceq'] = priceq
                                                        param_dict['qtlw'] = qtlw
                                                        param_dict['price'] = price.astype(np.float32)
                                                        param_dict['signal'] = signal
                                                        param_dict['qtl'] = qtl
                                                        param_dict['tp'] = tp
                                                        param_dict['sl'] = sl
                                                        param_dict['n1'] = n1
                                                        param_dict['n2'] = n2
                                                        param_dict['rsil'] = rsil
                                                        param_dict['rsih'] = rsih
                                                        param_dict['maf'] = maf
                                                        param_dict['mas'] = mas
                                                        param_dict['railag'] = railag
                                                        param_dict['num_weeks']= num_weeks
                                                        param_dict['request'] = request
                                                        param_dict['short'] = short
                                                        
                                                        backtest_configs.append(param_dict)

        logging.info('running {} backtest jobs'.format(len(backtest_configs)))
        ret = parallel_jobs2(wrapped_runsignal, backtest_configs, num_process=4)
        all_ret_list = list(itertools.chain.from_iterable(ret.values()))

        logging.info('saving res df')
        res_df = pd.DataFrame(all_ret_list)

        backtest_res_df_fname = '{}_{}_{}'.format(asset,  signalid, backtest_info_hashed)
        res_df.save(os.path.join(LOCAL_DATA_DIR, 'backtest_res', backtest_res_df_fname))
