import pandas as pd
import requests
from cdcqr.common.config import GLASSNODE_API_KEY
from functools import reduce
from fuzzywuzzy import fuzz
from cdcqr.common.utils import camel_case2snake_case


class GlassnodeData:
    def __init__(self):
        self.meta_data = GlassnodeData.get_meta_data()
        self.categories = sorted(list(self.meta_data['category'].unique()))
        self.feature_dict = self.meta_data.groupby('category')['f'].apply(list).to_dict()
        
    def get_feature_suggestions(self, f):
        f = camel_case2snake_case(f)
        fs = list(set(self.meta_data['f'].values))
        f2r = {}
        for f in fs:
            r = fuzz.ratio('nupl',f)
            f2r[f] = r
        suggestions = list(pd.DataFrame(f2r,index=[0]).T.sort_values([0]).reset_index()['index'].iloc[-2:].values)
        return suggestions


    def get_feature_category(self, f):
        f = camel_case2snake_case(f)
        fs = list(set(self.meta_data['f'].values))
        assert f in fs, "feature '{}' not found, do you mean one of {}".format(f, self.get_feature_suggestions(f))
        return self.meta_data[self.meta_data['f']==f]['category'].values[0]

    def get_feature_resolutions(self, f):
        f = camel_case2snake_case(f)
        fs = list(set(self.meta_data['f'].values))
        assert f in fs, "feature {} not found, do you mean one of {}".format(f, self.get_feature_suggestions(f))
        return self.meta_data[self.meta_data['f']==f]['resolutions'].values[0]

    def get_feature_best_resolutions(self, f):
        resolutions = self.get_feature_resolutions(f)
        for resolution in ['10m','1h','24h','1w','1month']:
            if resolution in resolutions:
                return resolution
        return None

    def get_feature_assets(self, f):
        f = camel_case2snake_case(f)
        asset_list = self.meta_data[self.meta_data['f']==f]['assets'].values[0]
        assets = []
        for asset_info in asset_list:
            assets.append(asset_info['symbol'])
        
        return assets

    @staticmethod
    def get_meta_data():
        url = 'https://api.glassnode.com/v2/metrics/endpoints'
        res = requests.get(url,
                           params={'api_key': GLASSNODE_API_KEY})
        df = pd.read_json(res.text)
        df[[1, 2, 3, 'category', 'f']] = df['path'].str.split('/', 4, expand=True)
        df = df.drop(labels=[1, 2, 3, 'path', 'formats'], axis=1)
        return df

    def load_features(self, f_list, asset, resolution='24h'):
        dfs = []
        for f in f_list:
            df = self._get_feature_df(f, asset, resolution)
            dfs.append(df)
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['t'],
                                            how='outer'), dfs).set_index('t')
        return df_merged


    def _get_feature_df(self, f, asset, resolution):
        f = camel_case2snake_case(f)
        category = self.get_feature_category(f)

        assets = self.get_feature_assets(f)
        assert asset in assets, "asset '{}' not in available asset {} for {}".format(asset, assets, f)
        
        resolutions = self.get_feature_resolutions(f)
        assert resolution in resolutions, "resolution '{}' not in {}".format(resolution, resolutions)
        
        print('loading', category, f, asset, resolution)
        res = requests.get('https://api.glassnode.com/v1/metrics/{}/{}'.format(category, f),
            params={'a': asset, 'i': resolution, 'api_key': GLASSNODE_API_KEY})
        
        df = pd.read_json(res.text, convert_dates=['t']).rename(columns={'v':'{}_{}'.format(asset, f)})
        
        return df
