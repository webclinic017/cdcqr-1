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
from cdcqr.common.config import GLASSNODE_API_KEY as GLASSKEY


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


def glassnode1(url='https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_mean',a='BTC',i='10m',s=dt(2021,8,1),Nretry=0,price=False, priceohlc=False,thresh=6,periods=1,log=False):
    """
    a BTC,ETH
    i 1month,1w,24h,1h,10m
    s dt(2021,8,1), '2021-08-01'
    url https://api.  ,   /metrics/transactions/transfers_volume_to_exchanges_mean , ~metrics~transactions~
    
    """
    try:
        timestamp=dt2ts(s,delta='1s')
    except:
        timestamp=dt2ts(dt.fromisoformat(s),delta='1s')
    
    if '~' in url:
        url=url.replace('~','/')
    
    if 'http' not in url:
        url='https://api.glassnode.com/v1/'+url
    #print(f"url={url} i={i} s={s}")
    
    try:
        res = requests.get(url,params={'a': a,'i':i,'s':timestamp,'api_key': GLASSKEY}) #, 's':1629500000
        return res
    except Exception as e:
        print(e)
    
#         df = pd.read_json(res.text, convert_dates=['t']).set_index('t') 
#     except Exception as e:
#         #print(e,res.text)
#         if Nretry==0:
#             raise e
#         time.sleep(120)
#         return glassnode(url=url,a=a,i=i,s=s,Nretry=Nretry-1)
    
   
    
#     return df