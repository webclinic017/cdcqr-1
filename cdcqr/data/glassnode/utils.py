from cdcqr.ct.utils import dt2ts
from cdcqr.common.utils import timeit, camel_case2snake_case
from cdcqr.common.config import CDQ_REPO_DIR
from datetime import datetime as dt
import time
import requests
import pandas as pd
import sys
sys.path.append(CDQ_REPO_DIR)
from ct.utils import glassnode


@timeit
def get_glassnode_data(category, feature, underly, freq, start_time, api_key):
    url ='https://api.glassnode.com/v1/metrics/{}/{}'.format(category, feature)
    return glassnode(url, underly, freq, start_time, Nretry=10, API_KEY=api_key)


def quantile_info(dff):
    q01 = dff.quantile(0.01).values[0]
    q02 = dff.quantile(0.02).values[0]
    q05 = dff.quantile(0.05).values[0]
    q95 = dff.quantile(0.95).values[0]
    q98 = dff.quantile(0.98).values[0]
    q99 = dff.quantile(0.99).values[0]
    print(q01, q02, q05, q95, q98, q99)
    return q01, q02, q05, q95, q98, q99


def url_parser(url):
    att_dict = {}
    for pair in url.split('&'):
        [k,v] = pair.split('=')
        att_dict[k] = v
        
    feature_camel_case = att_dict['m'].split('.')[1]
    att_dict['feature'] = camel_case2snake_case(feature_camel_case)
    att_dict['category'] = att_dict['m'].split('.')[0]
    return att_dict
