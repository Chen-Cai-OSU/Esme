from collections import OrderedDict
import sys

import numpy as np
import pandas as pd

from Esme.db.util import get_db
from Esme.shape.util import prince_cat

def format_(v, idx = 0):
    # idx 0 for training error, idx 1 for test error
    from Esme.helper.format import precision_format
    res = (1 - list(list(v.values())[0][idx].values())[0][1]) * 100
    res = precision_format(res, 1)
    return res

def sacred_to_df(db_runs, mongo_query=None, ):
    """
    db_runs is usually db.runs
    returns a dataframe that summarizes the experiments, where
    config and info fields are flattened to their keys.
    Summary DF contains the following columns:
    _id, experiment.name, **config, result, **info, status, start_time
    """
    # get all experiment according to mongo query and represent as a pandas DataFrame
    df = pd.DataFrame(list(db_runs.find(mongo_query)))

    # Take only the interesting columns
    df = df.loc[:, '_id, experiment, config, result, info, status, stop_time, start_time'.split(', ')]

    # print(df['config'])
    # sys.exit()
    def _summerize_experiment(s):
        o = OrderedDict()
        o['_id'] = s['_id']
        o['name'] = s['experiment']['name']
        o['status'] = s['status']

        try:
            tmp = s['stop_time'] - s['start_time']
            o['t'] = int(abs(s['stop_time'] - s['start_time']) / np.timedelta64(1, 's'))
        except TypeError:
            o['t'] = '0'

        res = s.get('result', None)# ['json://1']['py/tuple']
        if res != None and type(res) is dict:

            for k, v in res.items():
                # print(k, format_(v, idx=0))  # train error
                # print(k, format_(v, idx=1)) # test error
                train_error = format_(v, idx=0)
                test_error = format_(v, idx=1)
            o['trE'] = train_error
            o['teE'] = test_error

        config = s['config']
        from Esme.db.util import slice_dict
        config = slice_dict(config, ['graph', 'test_size', 'norm', 'permute', 'clf',  'std', 'idx', 'seg', 'vec', 'n_iter'])
        o.update(config)

        cat_dict = prince_cat()
        for k, v in cat_dict.items():
            if 'idx' not in o.keys():
                o['idx'] = -1

            if int(o['idx']) >= int(k[0]) and int(o['idx']) <= int(k[1]):
                o['cat'] = v
        if 'cat' not in o.keys(): o['cat'] ='NoFound'
        return pd.Series(o)

    sum_list = []
    for ix, s in df.iterrows():
        # print(_summerize_experiment(s))
        sum_list.append(_summerize_experiment(s))
    df_summary = pd.DataFrame(sum_list).set_index('_id')
    # df_summary = df_summary.drop_duplicates()
    return df_summary

def gen_row(row):
    return  '$' + str(row['mean_teE']) + '±' + str(row['std']) + '$'

if __name__ == '__main__':
    db = get_db('tda_shape')

    res = list(db.runs.find(None))
    # print(res[0]['captured_out'])

    query = '_id > 0 and status=="COMPLETED"'
    df = sacred_to_df(db.runs).query(query)
    # print(df.head(10))
    print(df.tail(20))
    print(f'before filter {len(df)}')

    idx_Series = df['teE'] < 30
    # for i in range(1, len(df_summary) -1):
    #     if i % 2 ==0 and idx_Series[i] == True:
    #         idx_Series[i+1] = True
    # print(idx_Series)
    df = df[idx_Series]
    print(f'after filter {len(df)}')

    mean_df = df.groupby(['cat', 'permute'])['teE', 'idx'].mean().round(1)
    std_df = df.groupby(['cat', 'permute'])['teE'].std().round(1)
    std_df = std_df.rename( columns={'teE': 'std'})
    res = pd.concat([mean_df, std_df], axis=1)
    res = res.rename(columns={'teE':'mean_teE', 0:'std'})


    res['final'] = res.apply(lambda row: gen_row(row), axis=1)

    # https://stackoverflow.com/questions/43729268/convert-a-python-dataframe-with-multiple-rows-into-one-row-using-python-pandas
    res_df = res.groupby(['cat'])['final'].apply(lambda x: pd.DataFrame(x.values)).unstack().reset_index()
    res_df.columns = res_df.columns.droplevel()

    idx_df = res.groupby(['cat'])['idx'].apply(lambda x: pd.DataFrame(x.values)).unstack().reset_index()
    idx_df.columns = idx_df.columns.droplevel()
    idx_df = idx_df.drop(columns=[1])

    res = pd.merge(res_df, idx_df, on='')
    res = res.rename(columns={'0_x': 'True(P)' , 1:  'False(NP)', '0_y':'idx'})
    res = res.sort_values(['idx'], ascending=True)
    res = res.drop(columns=['idx'])
    print(res)

    tex_str = res.to_latex()
    tex_str = tex_str.replace('cat', 'Categories')
    tex_str = tex_str.replace('toprule', 'hline')
    tex_str = tex_str.replace('midrule', 'hline')
    tex_str = tex_str.replace('bottomrule', 'hline')
    tex_str = tex_str.replace('\$', '$')
    tex_str = tex_str.replace('{}', '& {}')

    # dirty hack for remvoing idx
    for i in range(22,-1,-1):
        tex_str = tex_str.replace(str(i) + ' &', '')
        tex_str = tex_str.replace(str(i) + '  &', '')
    tex_str.replace('& {}', '')
    print(tex_str)


