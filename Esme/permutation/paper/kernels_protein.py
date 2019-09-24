import pandas as pd

from Esme.permutation.mongo_util import df_format, sacred_to_df
from Esme.permutation.mongo_util import get_tda_db

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

def tex_format(tex_str):
    tex_str = tex_str.replace('toprule', 'hline')
    tex_str = tex_str.replace('midrule', 'hline')
    tex_str = tex_str.replace('bottomrule', 'hline')
    return tex_str

def examine_one_graph(meta_df, graph='mutag', latex_f = False, n_cv = 1, print_flag = False):

    """
    examine the performance of different methods on one graph
    fix permute = False, flip = False, epd = False
    """
    
    # meta_df = sacred_to_df(db.runs)
    # query = 'permute == False and flip == False and _id >=560 and graph=="' + graph + '"'
    query = f'n_cv=={n_cv} and permute == {PERMUTE} and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and ntda!=True'
    df = meta_df.query(query)

    # add pf with permutation as False
    query = f'n_cv=={n_cv} and permute == False and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="pf" and ntda!=True'
    df_ = meta_df.query(query)
    df_.feat = 'pf'
    df = pd.concat([df, df_])

    allowed_feats = ['pervec', 'sw', 'sw_p', 'filvec', 'pf']

    query = f'n_cv=={n_cv} and permute == False and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="pss"'
    df_ = meta_df.query(query)
    df_.feat = 'pss'  # df_.rename(columns={'feat': 'population'}, inplace=True)
    df = pd.concat([df, df_])

    query = f'n_cv=={n_cv} and permute == False and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{graph}" and feat=="wg"'
    df_ = meta_df.query(query)
    df_.feat = 'wg'  # df_.rename(columns={'feat': 'population'}, inplace=True)
    df = pd.concat([df, df_])

    # filter out pervector and pss
    allowed_feats = ['sw', 'pss', 'wg', 'pf']
    df = df[df.feat.isin(allowed_feats)]

    df = df[df.feat.isin(allowed_feats)]
    if print_flag: print(df.to_string())

    grouped = df.groupby(['feat', 'fil'], as_index=False)
    df = grouped['result'].aggregate(max)
    df = df.pivot(index='feat', columns='fil', values='result')
    drop_cols = ['random', 'hks_100','fiedler']
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
        tex_str = df.to_latex(longtable=False).replace('\n', '\n\\caption{' + caption + '}\\\\\n', 1)
        tex_str = tex_format(tex_str)
        print(tex_str)
    return df


if __name__ == '__main__':
    db = get_tda_db()
    args = parser.parse_args()
    meta_df = sacred_to_df(db.runs)

    dfs = []
    for graph in ['protein_data']:
        individual_df = examine_one_graph(meta_df, graph, n_cv=1, latex_f=True) # todo change n_cv back to 1
        dfs.append(individual_df)

