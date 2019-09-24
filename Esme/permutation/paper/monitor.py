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

if __name__ == '__main__':
    db = get_tda_db()
    meta_df = sacred_to_df(db.runs)

    # test the pf query
    query = f'n_cv=={1} and permute == False and flip == {FLIP} and _id >={ID_THRESHOLD} and graph=="{"protein_data"}" and feat=="pf" and ntda!=True'
    df_ = meta_df.query(query)
    print(df_)

    # coarse grain
    query = '_id>=2100 and graph!="dfhr," and status=="COMPLETED" ' # status=="FAILED" and
    df_summary = meta_df.query(query)
    print(df_summary.tail(20))