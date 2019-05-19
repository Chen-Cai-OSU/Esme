# https://github.com/yuvalatzmon/SACRED_HYPEROPT_Example/blob/master/mongo_queries.ipynb

import pandas as pd
import pymongo
from pprint import pprint
import numpy as np
import sys
# Connect to client
from pymongo import MongoClient
client = MongoClient('localhost', 27017)

from collections import OrderedDict
import pandas as pd
import re

def slice_dict(d, keys):
    """ Returns a dictionary ordered and sliced by given keys
        keys can be a list, or a CSV string
    """

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    if isinstance(keys, str):
        keys = keys[:-1] if keys[-1] == ',' else keys
        keys = re.split(', |[, ]', keys)

    keys = intersection(keys, d.keys())
    return dict((k, d[k]) for k in keys)

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
        # o['st'] = s['heartbeat']
        try:
            tmp = s['stop_time'] - s['start_time']
            o['t'] = int(abs(s['stop_time'] - s['start_time']) / np.timedelta64(1, 's'))
        except TypeError:
            o['t'] = '0'

        tmp = s.get('result', None)
        o['result'] = None
        if type(tmp) is dict:
            # print(tmp)
            o['result'] = tmp['test']['mean']

        config = s['config']
        config = slice_dict(config, ['graph', 'fil', 'norm', 'permute', 'ss','epd', 'flip', 'feat', 'n_cv', 'clf', 'feat_kwargs'])
        o.update(config)

        # for key, val in s['info'].items():
        #     if key != 'metrics':
        #         o[key] = val

        # o.update(slice_dict(s.to_dict(), 'result, status'))
        return pd.Series(o)

    sum_list = []
    for ix, s in df.iterrows():
        # print(_summerize_experiment(s))
        sum_list.append(_summerize_experiment(s))
    df_summary = pd.DataFrame(sum_list).set_index('_id')

    return df_summary

def fail_query(db, id = 371):
    try:
        fail = db.runs.find()[id]['fail_trace']
        print(' The fail trace of id %s '%(id))
        for line in fail:
            print(line)
    except KeyError:
        print('The status of id %s is %s'%(id, db.runs.find()[id]['status'] ))
        pprint(db.runs.find()[id]['captured_out'])

def permute(db, graph):
    """
    output the table comparing permute versus not permute
    fix norm=True, epd=False, ss=True, n_cv=1.0, clf=svm
    :param graph: "reddit_5K"
    :return: df
    """
    query_fix = ' norm==True and epd==False and ss==True and n_cv==1.0 and clf=="svm" and '
    query = query_fix + '_id>=560 and graph=="' + graph +  '" and permute==True'
    df_permute_true = sacred_to_df(db.runs).query(query).groupby(['fil'])['result'].max()

    query_fix = ' norm==True and epd==False and ss==True and n_cv==1.0 and clf=="svm" and '
    query = query_fix + '_id>=560 and graph=="' + graph + '" and permute==False'
    df_permute_false = sacred_to_df(db.runs).query(query).groupby(['fil'])['result'].max()

    df = pd.concat([df_permute_true, df_permute_false], axis=1)
    df.columns = [graph + '.True', graph + '.False']

    print('This is the permutation test for graph %s'%graph)
    print(df)
    return df

def df_format(df):
    df = df.round(3)
    df = df.multiply(100)
    return df


# for i in range(150, 155):
#     fail_query(i)

# client.drop_database('sacred_mnist_example') # delete database
# client.drop_database('hyperopt') # delete database
