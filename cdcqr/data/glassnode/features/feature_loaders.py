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
    df3['ret'] = df3['BTC-price_usd_close'].pct_change()
    df3['r+5'] = df3['BTC-price_usd_close'].pct_change(periods=5).shift(-5)
    df3['r+10'] = df3['BTC-price_usd_close'].pct_change(periods=10).shift(10)
    df3['r+100'] = df3['BTC-price_usd_close'].pct_change(periods=100).shift(100)

    df3['stable_bal_chg'] = df3['stable-balance_exchanges'].diff()
    df3['stable_bal_chg_lag1'] = df3['stable_bal_chg'].shift(1)
    df3['BTCSSR_lag1'] = df3['BTCSSR'].shift(1)
    df3['BTCSSR_chg'] = df3['BTCSSR'].diff()
    df3['BTCSSR_chg_lag1'] = df3['BTCSSR_chg'].shift(1)
    df4 = df3.dropna()[['stable_bal_chg', 'stable_bal_chg_lag1', 'BTCSSR', 'BTCSSR_chg', 'BTCSSR_chg_lag1', 'ret', 'r+5', 'r+10', 'r+100']]
    df4a = df4.where(~np.isinf(df4), 0)
    
    feature_data_fname = 'gn_{}'.format(feature)
    feature_data_dir = os.path.join(LOCAL_DATA_DIR, 'feature_data', feature_data_fname)
    df4a.save(feature_data_dir)
    return feature_data_fname
