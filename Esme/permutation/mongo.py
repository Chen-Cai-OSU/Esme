import pandas as pd
import pymongo
from pprint import pprint
import numpy as np
import sys
# Connect to client
from pymongo import MongoClient
from collections import OrderedDict
import pandas as pd
import re
from Esme.permutation.mongo_util import permute, df_format, sacred_to_df

def permute_table(db):
    """
    generate latex table for permutation test
    :param db:
    :return:
    """
    df1 = permute(db, 'reddit_5K')
    df2 = permute(db, 'protein_data')
    df = pd.concat([df_format(df1), df_format(df2)], axis=1)
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split('.')) for c in df.columns])
    print(df.to_latex())


def one_feat(db, feat = 'sw'):
    """
    examine one featuralization method for different graph and filtration
    fix permute = False, flip = False, epd = False
    """
    query = 'permute == False and flip == False and _id >=560 and feat=="' + feat + '"'
    df = sacred_to_df(db.runs).query(query)
    grouped = df.groupby(['graph', 'fil'], as_index=False)
    df = grouped['result'].aggregate(max)
    df = df.pivot(index='graph', columns='fil', values='result')
    df = df_format(df)
    print('Here is the summary for %s featuralization\n'%feat)
    print(df)
    print(df.to_latex())

def one_graph(db, graph='mutag'):
    """
    examine the performance of different methods on one graph
    fix permute = False, flip = False, epd = False
    """
    query = 'permute == False and flip == False and _id >=560 and graph=="' + graph + '"'
    df = sacred_to_df(db.runs).query(query)
    grouped = df.groupby(['feat', 'fil'], as_index=False)
    df = grouped['result'].aggregate(max)

    df = df.pivot(index='feat', columns='fil', values='result')
    df = df_format(df)
    print('Here is the summary of different feats for graph %s\n' % graph)
    print(df)
    print(df.to_latex())


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    print('databases:', client.list_database_names())
    db = client['tda']



    print('Collections of {} db: {}'.format(db.name, db.list_collection_names()))
    one_graph(db, graph='nci1')

    # Get the COMPLETED experiments with dim<=100 and val. accuracy > 85%
    # query = 'fil=="deg" and n_cv ==1 and permute==False and _id > 20 and status=="COMPLETED" and (graph == "imdb_binary" or graph=="imdb_multi" or graph=="nci1")'
    query = '_id>=560 '

    # coarse grain
    df_summary = sacred_to_df(db.runs).query(query)
    print(df_summary)
    sys.exit()




    # finer grain
    df_summary = sacred_to_df(db.runs).query(query).groupby(['graph', 'fil', 'permute'])['result'].max()
    # df_summary = df_summary.sort_values('graph')
    df_summary = df_summary.round(3)
    df_summary = df_summary.multiply(100, 'result')
    print(df_summary.to_latex())
    print(df_summary)
    sys.exit()

    idx = df_summary.groupby(['fil', 'graph', 'feat'])['result'].transform(max) == df_summary['result']

    print(df_summary[idx])
    # df_summary = df_summary.sort_values('_id', ascending=False) # Sort them in descending order (best performer is first).
    # print(df_summary)
    sys.exit()

    for i, exp in enumerate(list(db.runs.find())[-2:]):
        # pprint(exp)
        # continue
        pprint(exp['config'])
        pprint(exp['result'])
