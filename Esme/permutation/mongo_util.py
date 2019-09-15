"""
auxiliary functions to query mongodb
"""
# https://github.com/yuvalatzmon/SACRED_HYPEROPT_Example/blob/master/mongo_queries.ipynb

import re
import sys
import warnings
from collections import OrderedDict
from pprint import pprint

import numpy as np
import pandas as pd
# Connect to client
from pymongo import MongoClient
from sacred import Experiment
from sacred.observers import MongoObserver

from Esme.helper.format import print_line
from Esme.helper.time import timefunction


def set_ex():
    client = MongoClient('localhost', 27017)
    EXPERIMENT_NAME = 'PD'
    YOUR_CPU = None  # None is the default setting and will result in using localhost, change if you want something else
    DATABASE_NAME = 'tda'
    ex = Experiment(EXPERIMENT_NAME)
    ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    return ex


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

@timefunction
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
        o['std'] = -1
        if type(tmp) is dict:
            try:
                o['result'] = tmp['test']['mean']
                o['std'] =  tmp['test'].get('std',-1) # todo added by Chen
            except:
                o['result'] = -1


        config = s['config']
        config = slice_dict(config, ['graph', 'fil', 'norm', 'permute', 'ss', 'epd', 'flip', 'feat', 'n_cv', 'clf', 'feat_kwargs', 'ntda', 'std'])
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
    # df_summary = df_summary.drop_duplicates()
    return df_summary


def fail_query(db, id=371):
    try:
        all_info = db.runs.find()[id]
        fail = all_info['fail_trace']
        print(' The fail trace of id %s ' % (id))
        for line in fail:
            print(line)
    except KeyError:
        print('The status of id %s is %s' % (id, db.runs.find()[id]['status']))
        pprint(db.runs.find()[id]['captured_out'])

    # other auxliary info
    config = all_info['config']
    start_time = all_info['start_time']
    print(config, start_time)


def query_db(db, q):
    # given a query on db, return a df
    df_filter = sacred_to_df(db.runs).query(q)
    if df_filter.shape[0] == 0:
        tmp = df
        # tmp.loc[1:] = -1
        tmp = tmp.fillna(-3.14)
        print(tmp)
        return tmp
    else:
        return df_filter


def permute(db, graph, to_latex_f=False, feat='sw'):
    """
    output the table comparing permute versus not permute
    fix norm=True, epd=False, ss=True, n_cv=1.0, clf=svm
    :param graph: "reddit_5K"
    :return: df
    """

    query_fix = 'norm==True and epd==False and ss==True and n_cv==1.0 and clf=="svm" and ' + '_id>=560 and graph=="' + graph + '" and feat==' + feat + '"'
    query = query_fix + '" and permute==True'
    # df_filter = query_db(db, query)
    df_filter = sacred_to_df(db.runs).query(query)
    groupby = df_filter.groupby(['fil'])['result']  # SeriesGroupBy object
    df_permute_true, df_permute_true_std = groupby.max(), groupby.std()

    query = query_fix + '" and permute==False'
    df_filter = sacred_to_df(db.runs).query(query)
    # df_filter = query_db(db, query)
    groupby = df_filter.groupby(['fil'])['result']  # SeriesGroupBy object
    df_permute_false, df_permute_false_std = groupby.max(), groupby.std()

    df = pd.concat([df_permute_true, df_permute_true_std, df_permute_false, df_permute_false_std], axis=1)
    tmp_colnames = ['T1', 'std1', 'F2', 'std2']
    df.columns = tmp_colnames
    df = (100 * df).round(1)
    df['P'] = df.apply(lambda row: str(float(row.T1)) + '(' + str(float(row.std1)) + ')', axis=1)
    df['NP'] = df.apply(lambda row: str(float(row.F2)) + '(' + str(float(row.std2)) + ')', axis=1)
    df = df.drop(tmp_colnames, axis=1)
    caption = query_fix

    if to_latex_f:
        print(df.to_latex(longtable=False).replace('\n', '\n\\caption{' + caption + '}\\\\\n', 1))

    print('This is the permutation test for graph %s' % graph)
    print('Fixed params are ' + query_fix)
    print(df)
    return df


def time_summary(db):
    # look at the runtime of different method
    query = 'n_cv==1.0 and clf=="svm" and _id>=0 '
    df_filter = sacred_to_df(db.runs).query(query)
    df_filter.t = df_filter.t.astype(float)
    groupby = df_filter.groupby(['graph'])['t']
    res = pd.concat([groupby.max(), groupby.std()], axis=1)
    res.columns = ['t', 't_std']
    print(res)


def df_format(df):
    df = df.round(3)
    df = df.multiply(100)
    return df


def get_tda_db():
    client = MongoClient('localhost', 27017)
    # print('databases:', client.list_database_names())
    db = client['tda']
    # print('Collections of {} db: {}'.format(db.name, db.list_collection_names()))
    return db


@timefunction
def all_configs(db):
    # get all configs and results for duplication check

    configs = {}
    accuracies = {}
    n = db.runs.count()
    for id in range(1, n):
        tmp = db.runs.find()[id]  # tmp is dict
        config = tmp['config']  # {'clf': 'svm', 'epd': True, 'feat': 'sw', 'feat_kwargs': {'bw': 1, 'n_d': 10}, 'fil': 'ricci', 'flip': False, 'graph': 'imdb_multi', 'n_cv': 1, 'norm': True, 'permute': True, 'seed': 864373169, 'ss': True}
        # config.pop('seed', None) # remove seed to make matching easy
        accuracy = tmp.get('result', None)  # {'test': {'mean': 0.609, 'n_cv': 1, 'std': 0.0}, 'train': {'param': {'C': 0.01, 'kernel': 'precomputed'}, 'score': 50.7}}
        configs[id] = config
        accuracies[id] = accuracy
    return configs, accuracies


def check_duplicate(db, param):
    """
    check the duplicate of tda db
    :param db:
    :param param: # param = {...,(more) 'clf': 'svm', 'epd': True, 'feat': 'sw', 'fil': 'random', 'graph': 'imdb_binary', 'n_cv': 1, 'norm': True, 'permute': False, 'ss': True}

    :return: true if there is duplicate else false
    """

    if param.get('graph', None) in ['frankenstein', 'reddit_binary', 'reddit_5K', 'reddit_12K', 'nci1', 'nci109', 'collab']:
        pass  # do nothing when graph is large
    if param.get('feat', None) in ['pf']:
        pass
    else:
        return False  # when graph is small, it doesn't save time to check duplication. Just run exp again.

    configs, accuracies = all_configs(db)
    for k, v in configs.items():
        if param.items() <= v.items() and accuracies[k] != None:
            print('Found duplicate', k, v, accuracies[k]['test']['mean'])
            print_line()
            return True
    print('No duplicate found')
    return False


if __name__ == '__main__':
    db = get_tda_db()
    query = '_id>=2100 and graph!="dfhr," '  # status=="FAILED" and
    df_summary = sacred_to_df(db.runs).query(query)
    print(df_summary.tail(20))

    sys.exit()
    db = get_tda_db()
    query = 'None'
    print(list(db.runs.find())[-10]['captured_out'])

    sys.exit()
    all_configs_ = all_configs(db)
    print(len(all_configs_))
    sys.exit()

    fail_query(db, id=3175 - 1)
    sys.exit()

    # automate generate permutation test table. Half done.
    dfs = []
    for graph in ['imdb_binary', 'dd_test']:  # ['reddit_5K','imdb_binary', 'nci1']:
        df = permute(db, graph)
        print(df)

    # pd.concat(dfs, axis=1)
