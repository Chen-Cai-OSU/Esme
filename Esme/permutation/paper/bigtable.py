import pandas as pd
import sys

from Esme.permutation.mongo_util import df_format, sacred_to_df
from Esme.permutation.mongo_util import get_tda_db
from Esme.helper.time import timefunction
ID_THRESHOLD =  560 # used for most of the time
# ID_THRESHOLD = 10000 # test how much speed for for the query
PERMUTE = False
FLIP = False
FEAT = 'sw'
GRAPH = 'bzr'
status = 'COMPLETED'

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser("scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--n_entry", default=20, type=int, help='number recent entries"')

@timefunction
def examine_one_graph(meta_df, graph='mutag', latex_f = False, n_cv = 1, print_flag = False):

    """
    examine the performance of different methods on one graph
    fix permute = False, flip = False, epd = False
    """

    # query = 'permute == False and flip == False and _id >=560 and graph=="' + graph + '"'
    query = f'n_cv=={n_cv} and permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and ntda!=True'
    df = meta_df.query(query)

    # add sw with permutation as True
    query = f'n_cv=={n_cv} and permute == True and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="sw" and ntda!=True'
    df_ = meta_df.query(query)
    df_.feat = 'sw_p' # df_.rename(columns={'feat': 'population'}, inplace=True)
    df = pd.concat([df, df_])

    # add filvec
    query = f'n_cv=={n_cv} and permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="pervec" and ntda==True'
    df_ = meta_df.query(query)
    df_.feat = 'filvec'
    df = pd.concat([df, df_])

    allowed_feats = ['pervec', 'sw', 'sw_p', 'filvec']

    if False: # turned on sometimes
        query = f'permute == True and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="pss"'
        df_ = meta_df.query(query)
        df_.feat = 'pss_p'  # df_.rename(columns={'feat': 'population'}, inplace=True)
        df = pd.concat([df, df_])

        query = f'permute == True and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="wg"'
        df_ = meta_df.query(query)
        df_.feat = 'wg_p'  # df_.rename(columns={'feat': 'population'}, inplace=True)
        df = pd.concat([df, df_])

        # filter out pervector and pss
        allowed_feats = ['pervec', 'sw', 'sw_p', 'pss', 'pss_p', 'wg', 'wg_p', 'pf']
        df = df[df.feat.isin(allowed_feats)]

    df = df[df.feat.isin(allowed_feats)]
    if print_flag: print(df.to_string())

    grouped = df.groupby(['feat', 'fil'], as_index=False)
    df = grouped['result'].aggregate(max)
    df = df.pivot(index='feat', columns='fil', values='result')
    drop_cols = ['random', 'fiedler', 'hks_100']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(drop_cols, axis=1)

    df = df_format(df)
    df['mean'] = df.mean(axis=1) # add mean column
    df = df.round({'mean': 2})

    caption = f'Here is the summary of different feats for graph {graph} with n_cv {n_cv}\n' + query
    print(caption)
    print(df)
    print('-'*150)
    if latex_f:
        print(df.to_latex(longtable=False).replace('\n', '\n\\caption{' + caption + '}\\\\\n', 1))
    return df

if __name__ == '__main__':
    db = get_tda_db()
    args = parser.parse_args()
    meta_df = sacred_to_df(db.runs)
    print(meta_df.tail(20))
    # sys.exit()


    dfs = []
    # graph_names = ['protein_data','bzr']
    graph_names = ['bzr','cox2', 'dd_test', 'dhfr', 'frankenstein', 'imdb_binary', 'imdb_multi', 'nci1', 'protein_data', 'reddit_binary'] # ['protein_data', 'imdb_binary'] # # ['mutag'] # ['collab','protein_data', 'nci1'] #  # ['cox2'  'dfhr' 'imdb_binary' 'imdb_multi' 'dd_test' ] # ['bzr', 'cox2', 'dd_test', 'dhfr', 'frankenstein', 'imdb_binary', 'imdb_multi', 'nci1', 'protein_data', 'ptc', 'reddit_5K'] # ['dhfr', 'frankenstein', 'cox2' ]
    for graph in graph_names:
        individual_df = examine_one_graph(meta_df, graph, n_cv=1, latex_f=False) # todo change n_cv back to 1
        if graph == 'imdb_binary': individual_df = individual_df.drop(['', 'dd'], axis=1)
        dfs.append(individual_df)

    concatdf = pd.concat(dfs[:-1], axis=0, keys=graph_names)
    concatdf['mean'] = concatdf.mean(axis=1) #
    concatdf = concatdf.round({'mean': 2})
    print(concatdf)

    tex_str = concatdf.to_latex()

    # format
    tex_str = tex_str.replace('toprule', 'hline')
    tex_str = tex_str.replace('midrule', 'hline')
    tex_str = tex_str.replace('bottomrule', 'hline')
    tex_str = tex_str.replace('{llrrrrr}', '{| c |  c | c  c  c  c  | c |}')
    for g in ['bzr', 'cox2', 'dd\_test', 'dhfr', 'frankenstein', 'imdb\_binary', 'imdb\_multi', 'nci1', 'protein\_data']:
        tex_str = tex_str.replace(g, "\\texttt{\\uppercase{" + g + "}}")
    tex_str = tex_str.replace('imdb\_multi', 'imdb-m')
    tex_str = tex_str.replace('imdb\_binary', 'imdb-b')
    tex_str = tex_str.replace('protein\_data', 'protein')
    tex_str = tex_str.replace('dd\_test', 'DD')
    tex_str = tex_str.replace('ricci', 'Ricci')
    tex_str = tex_str.replace('\\\\\n\\texttt', '\\\\\\hline\n\\texttt')
    tex_str = tex_str.replace('{} & feat &', '% {} & feat &')
    tex_str = tex_str.replace('  & fil', ' graph & fil')
    print(tex_str)


