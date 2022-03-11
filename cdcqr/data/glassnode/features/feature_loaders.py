from cdcqr.data.glassnode import glassnode_data as gnd
import numpy as np
from cdcqr.common.utils import LOCAL_DATA_DIR, timeit
import os
import logging
LOG_FILENAME = '/core/logs/backtest.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',filename=LOG_FILENAME)


@timeit
def produce_feature_BTCSSR():
    feature = 'BTCSSR'
    feature_data_fname = 'gn_{}'.format(feature)
    feature_data_dir = os.path.join(LOCAL_DATA_DIR, 'feature_data','{}.pickle'.format(feature_data_fname))
    logging.info(feature_data_dir)
    if os.path.isfile(feature_data_dir):
        logging.info('feature data {} exists'.format(feature))
    else:

        logging.info('producing feature data {}'.format(feature))
        gn = gnd.GlassnodeData()
        gn.get_feature_best_resolutions('balance_exchanges')
        fs = ['price_usd_close', 'balance_exchanges']
        assets = ['BTC','USDT', 'USDC', 'BUSD']
        df = gn.load_features(fs, assets, resolution='10m')

        df1 = df[df.index>='20200101'].ffill().drop(['USDT-price_usd_close','USDC-price_usd_close','BUSD-price_usd_close'],axis=1)
        df1['stable-balance_exchanges'] = df['USDT-balance_exchanges']+df['USDC-balance_exchanges']+df['BUSD-balance_exchanges']
        df1 = df1[['BTC-price_usd_close', 'stable-balance_exchanges','BTC-balance_exchanges']]
        df1['BTC-balance_exchanges_MC'] = df1['BTC-price_usd_close'] * df1['BTC-balance_exchanges']
        df1['BTCSSR'] = df1['BTC-balance_exchanges_MC']/df1['stable-balance_exchanges']
        df2 = df1.drop('BTC-balance_exchanges_MC', axis=1).dropna()

        df3 = df2.copy()

        df3['stable_bal_chg'] = df3['stable-balance_exchanges'].diff()
        df3['stable_bal_chg_lag1'] = df3['stable_bal_chg'].shift(1)
        df3['BTCSSR_lag1'] = df3['BTCSSR'].shift(1)
        df3['BTCSSR_chg'] = df3['BTCSSR'].diff()
        df3['BTCSSR_chg_lag1'] = df3['BTCSSR_chg'].shift(1)
        df4 = df3.dropna()[['stable_bal_chg', 'stable_bal_chg_lag1', 'BTCSSR', 'BTCSSR_chg', 'BTCSSR_chg_lag1']]
        df4a = df4.where(~np.isinf(df4), 0)
        df4a.save(feature_data_dir)
    return feature_data_fname

@timeit
def produce_feature_PERPFundingRate():
    feature = 'BTCPERPFundingRate'
    feature_data_fname = 'gn_{}'.format(feature)
    feature_data_dir = os.path.join(LOCAL_DATA_DIR, 'feature_data','{}.pickle'.format(feature_data_fname))
    logging.info(feature_data_dir)
    logging.info(os.path.isfile(feature_data_dir))
    if os.path.isfile(feature_data_dir):
        logging.info('feature data {} exists'.format(feature))
    else:
        logging.info('producing feature data {}'.format(feature))
        gn = gnd.GlassnodeData()
        fs =  ['price_usd_close','FuturesFundingRatePerpetual']
        a = ['BTC']
        #i = '10m'
        df = gn.load_features(fs,a)

        df1 = df[df.index>='20200101']
        df3 = df1.copy()

        df4 = df3.dropna()[['BTC-FuturesFundingRatePerpetual']]
        df4a = df4.where(~np.isinf(df4), 0)
        df4a.columns = [feature]
        df4a.save(feature_data_dir)
    return feature_data_fname
