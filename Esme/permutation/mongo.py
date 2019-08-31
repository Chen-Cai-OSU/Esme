import sys

# Connect to client
import pandas as pd

from Esme.permutation.mongo_util import permute, df_format, sacred_to_df
from Esme.permutation.mongo_util import get_tda_db
from Esme.permutation.mongo_util import permute
from Esme.helper.time import timefunction
from Esme.graph.dataset.stoa import perslay

ID_THRESHOLD = 560
# ID_THRESHOLD = 9000
PERMUTE = False
FLIP = False
FEAT = 'sw'
GRAPH = 'bzr'
status = 'COMPLETED'


def probe_db():
    query = f'permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}"'
    df = sacred_to_df(db.runs).query(query)
    print(df.to_string())


def permute_table(db, latex_f = False):
    """
    generate latex table for permutation test
    :param db:
    :return:
    """
    df1 = permute(db, 'reddit_5K')
    df2 = permute(db, 'protein_data')
    df = pd.concat([df_format(df1), df_format(df2)], axis=1)
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split('.')) for c in df.columns])
    print(df)
    if latex_f:
        return df.to_latex()

def examine_one_feat(db, feat ='sw', latex_f = False):
    """
    examine one featuralization method for different graph and filtration
    fix permute = False, flip = False, epd = False
    """

    query = f'permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and feat=="{feat}"'
    df = sacred_to_df(db.runs).query(query)
    grouped = df.groupby(['graph', 'fil'], as_index=False)
    df = grouped['result'].aggregate(max)
    df = df.pivot(index='graph', columns='fil', values='result')
    df = df_format(df)

    print('Here is the summary for %s featuralization\n'%feat)
    print(df)
    print()
    if latex_f:
        print(df.to_latex())

@timefunction
def examine_one_graph(db, graph='mutag', latex_f = False, n_cv = 1, print_flag = False):

    """
    examine the performance of different methods on one graph
    fix permute = False, flip = False, epd = False
    """
    # query = 'permute == False and flip == False and _id >=560 and graph=="' + graph + '"'
    query = f'n_cv=={n_cv} and permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and ntda!=True'
    df = sacred_to_df(db.runs).query(query)

    # add sw with permutation as True
    query = f'n_cv=={n_cv} and permute == True and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="sw" and ntda!=True'
    df_ = sacred_to_df(db.runs).query(query)
    df_.feat = 'sw_p' # df_.rename(columns={'feat': 'population'}, inplace=True)
    df = pd.concat([df, df_])

    # add filvec
    query = f'n_cv=={n_cv} and permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="pervec" and ntda==True'
    df_ = sacred_to_df(db.runs).query(query)
    df_.feat = 'filvec'  # df_.rename(columns={'feat': 'population'}, inplace=True)
    df = pd.concat([df, df_])
    allowed_feats = ['pervec', 'sw', 'sw_p', 'filvec']

    if True: # turned on sometimes
        query = f'permute == True and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="pss"'
        df_ = sacred_to_df(db.runs).query(query)
        df_.feat = 'pss_p'  # df_.rename(columns={'feat': 'population'}, inplace=True)
        df = pd.concat([df, df_])

        query = f'permute == True and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="wg"'
        df_ = sacred_to_df(db.runs).query(query)
        df_.feat = 'wg_p'  # df_.rename(columns={'feat': 'population'}, inplace=True)
        df = pd.concat([df, df_])

        # filter out pervector and pss
        allowed_feats = ['pervec', 'sw', 'sw_p', 'pss', 'pss_p', 'wg', 'wg_p']
        df = df[df.feat.isin(allowed_feats)]

    df = df[df.feat.isin(allowed_feats)]
    if print_flag: print(df.to_string())

    grouped = df.groupby(['feat', 'fil'], as_index=False)
    df = grouped['result'].aggregate(max)
    df = df.pivot(index='feat', columns='fil', values='result')
    drop_cols = ['random']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(drop_cols, axis=1)

    df = df_format(df)
    df['mean'] = df.mean(axis=1) # add mean column

    caption = f'Here is the summary of different feats for graph {graph} with n_cv {n_cv}\n' + query
    print(caption)
    print(df)
    print('-'*150)
    if latex_f:
        print(df.to_latex(longtable=False).replace('\n', '\n\\caption{' + caption + '}\\\\\n', 1))
    return df

def examine_homology(db, latex_f = False, stoa = {}):
    """
        examine the performance of different methods on one graph
        fix permute = False, flip = False, epd = False

        stoa is a dict
    """

    # query = 'permute == False and flip == False and _id >=0 and graph=="' + graph + '"'
    query = f'permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and feat=="{FEAT}"'
    df = sacred_to_df(db.runs).query(query)
    grouped = df.groupby(['epd', 'graph'], as_index=False)
    df = grouped['result'].aggregate(max)
    df = df.pivot(index='epd', columns='graph', values='result')
    df = df_format(df)
    df.drop(['dhfr,', 'reddit_12K'], axis=1) # drop certain columns
    stoa = {k:v for k,v in stoa.items() if k in df.columns}
    df = df.append(stoa, ignore_index=True)


    caption = f'Here is the summary of whether to add 1 homolgy for different graphs' + query
    print(caption)
    print(df)
    print()
    if latex_f:
        print(df.to_latex(longtable=False).replace('\n', '\n\\caption{' + caption + '}\\\\\n', 1))

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n_entry", default=20, type=int, help='number recent entries"')

if __name__ == '__main__':
    db = get_tda_db()
    args = parser.parse_args()
    # look at imdb wired case
    if False:
        query = '_id>5000 and graph == "imdb_binary" and feat=="pervec" and fil=="random"'
        df_summary = sacred_to_df(db.runs).query(query)
        print(df_summary)
        sys.exit()

    # coarse grain
    query = '_id>=2100 and graph!="dfhr," ' # status=="FAILED" and
    df_summary = sacred_to_df(db.runs).query(query)
    print(df_summary.tail(args.n_entry))
    sys.exit()
    # examine_one_graph(db, 'reddit_binary', latex_f=True, print_flag=False)
    # sys.exit()

    # examine one graph (aggregation results)
    dfs = []
    graph_names = ['collab','protein_data', 'nci1'] # ['cox2',  'dfhr', 'imdb_binary', 'imdb_multi', 'dd_test' ] # ['cox2'  'dfhr' 'imdb_binary' 'imdb_multi' 'dd_test' ] # ['bzr', 'cox2', 'dd_test', 'dhfr', 'frankenstein', 'imdb_binary', 'imdb_multi', 'nci1', 'protein_data', 'ptc', 'reddit_5K'] # ['dhfr', 'frankenstein', 'cox2' ]
    for graph in graph_names:
        # graph = 'protein_data'
        # _ = examine_one_graph(db, graph, latex_f=False, n_cv=10)
        individual_df = examine_one_graph(db, graph, n_cv=1, latex_f=False)
        dfs.append(individual_df)
    sys.exit()

    concatdf = pd.concat(dfs, axis=0, keys=graph_names)
    concatdf['mean'] = concatdf.mean(axis=1) #
    # concatdf.loc['mean'] = concatdf.mean(axis=0)
    print(concatdf)
    print(concatdf.to_latex())
    sys.exit()
    # examine one feat sw
    examine_homology(db, latex_f=True, stoa=perslay())
    examine_one_feat(db, feat='pervec')

    # examine_homology
    FEAT = 'sw'
    examine_homology(db, latex_f = True)
    FEAT = 'pervec'
    examine_homology(db, latex_f=True)
    sys.exit()

    # permutation test table
    dfs = []
    # graph_names = ['cox2', 'frankenstein', 'bzr', 'protein_data', 'dd_test', 'imdb_multi', 'imdb_binary', 'mutag', 'ptc'] # , ]
    for graph in graph_names:
        individual_df = permute(db, graph)
        dfs.append(individual_df)
    concatdf = pd.concat(dfs, axis=0, keys=graph_names)
    print(concatdf)
    print(concatdf.to_latex())

    sys.exit()

    # Get the COMPLETED experiments with dim<=100 and val. accuracy > 85%
    # query = 'fil=="deg" and n_cv ==1 and permute==False and _id > 20 and status=="COMPLETED" and (graph == "imdb_binary" or graph=="imdb_multi" or graph=="nci1")'

    # finer grain
    df_summary = sacred_to_df(db.runs).query(query).groupby(['graph', 'fil', 'permute'])['result'].max()
    # df_summary = df_summary.sort_values('graph')
    df_summary = df_summary.round(3)
    df_summary = df_summary.multiply(100, 'result')
    print(df_summary.to_latex())
    print(df_summary)
    sys.exit()

    for i, exp in enumerate(list(db.runs.find())[-2:]):
        # pprint(exp)
        # continue
        pprint(exp['config'])
        pprint(exp['result'])


    # fail query
    from Esme.permutation.mongo_util import fail_query
    fail_query(db, 4167)
