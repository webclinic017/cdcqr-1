import pandas as pd
from cdcqr.common.config import LOCAL_DATA_DIR
import os
from cdcqr.analytics.derivatives.vol_fitting.smoothing.utils import optchain2local_vol_res_df
import argparse



if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='start date')
    parser.add_argument('startdate', type=str, help='eg: 20211201')
    parser.add_argument('enddate', type=str, help='eg: 20211201')
    args = parser.parse_args()
    
    start_date = args.startdate
    end_date = args.enddate

    all_dates = pd.date_range(start_date, end_date)
    expire_dates = ['MAR22','JUN22']

    for date_ in all_dates:
        for exp_date in expire_dates:
            print('-------------',date_, '-------------', exp_date, '-------------')
            fname = 'bt_spike{}_{}.pkl'.format(date_.strftime('%Y-%m-%d'), exp_date)
            res_fname = 'bt_spike{}_{}_res.pkl'.format(date_.strftime('%Y-%m-%d'), exp_date)
            if os.path.isfile(os.path.join(LOCAL_DATA_DIR, res_fname)):
                pass
            else:
                try:
                    optchain_file1 = pd.read_pickle(os.path.join(LOCAL_DATA_DIR, fname))
                    res1 = optchain2local_vol_res_df(optchain_file1)
                    res1['dt'] = pd.to_datetime(res1['t'], unit='ms')
                    res1.to_pickle(os.path.join(LOCAL_DATA_DIR, res_fname))
                except:
                    print('can not find file', fname)
                    pass

