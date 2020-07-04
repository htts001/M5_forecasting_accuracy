import pandas as pd
import numpy as np
import IPython
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing, metrics
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm


def onehot_encoder(df, cols=[], set_index=None):
    """
    One hot the features in `cols` list, append the features along index and remove the `cols` for dataframe.
    Returns: Updated & complete dataframe.
    """
    ohe = OneHotEncoder(handle_unknown='ignore')
    transformed = ohe.fit_transform(df[cols]).toarray()
    flattened = pd.DataFrame(data=transformed, columns=ohe.get_feature_names(cols))
    print({'categories': ohe.categories_, 'features names': ohe.get_feature_names(cols)})

    updated_df = pd.concat([df.drop(columns=cols), flattened], axis=1)
    if set_index:
        updated_df = updated_df.set_index(set_index).reset_index()
    print('OHE complete !!\n')
    return updated_df


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def sizeof_fmt(num, suffix='B'):
    """
    Usage: print("{:>20}: {:>8}".format('Reduced df size: ',sizeof_fmt(df.memory_usage(index=True).sum())))
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def display(*dfs):
    for df in dfs:
        IPython.display.display(df)
# display(sales_train_val.head(3))


