
# Problem: reference: https://www.kaggle.com/c/m5-forecasting-accuracy
import pandas as pd
import numpy as np
import seaborn, utils, gc, warnings
from datetime import datetime
from features import m5_feature_engg
from utils import onehot_encoder, encode_categorical, reduce_mem_usage
from models import run_lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

def read_data():
    print('\n\nRunning read_data')
    calendar_df = pd.read_csv('./input/calendar.csv') #date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, snap_CA, snap_TX, snap_WI 
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    # id, submission_id are only unique, # 30k
    # 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd_1' ... 'd_1941'
    sales_train_evaluation_df = pd.read_csv('./input/sales_train_evaluation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_evaluation_df.shape[0], sales_train_evaluation_df.shape[1]))

    # no uniques, # 6M
    sell_prices_df = pd.read_csv('./input/sell_prices.csv')  #store_id, item_id, wm_yr_wk, sell_price
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    submission_df = pd.read_csv('./input/sample_submission.csv')

    calendar_df = encode_categorical(calendar_df, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)
    sales_train_evaluation_df = encode_categorical(sales_train_evaluation_df, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]).pipe(reduce_mem_usage)
    sell_prices_df = encode_categorical(sell_prices_df, ["item_id", "store_id"]).pipe(reduce_mem_usage)

    # PICKLES
    calendar_df.to_pickle('./data/calendar_df.pkl.compress', compression="gzip")
    sales_train_evaluation_df.to_pickle('./data/sales_train_evaluation_df.pkl.compress', compression="gzip")
    sell_prices_df.to_pickle('./data/sell_prices_df.pkl.compress', compression="gzip")
    return calendar_df, sell_prices_df, sales_train_evaluation_df, submission_df


def melt_and_merge(calendar, sell_prices, sales_train_evaluation, submission, nrows = 55000000, merge = False):
    print('\n\n Running melt and merge\n')
    # melt sales data, get it ready for training
    sales_train_evaluation = pd.melt(sales_train_evaluation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_evaluation.shape[0], sales_train_evaluation.shape[1]))
    sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)
    sales_train_evaluation = sales_train_evaluation.iloc[-nrows:,:]
    
    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]
    
    # change column names
    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
                      'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
    
    # get product table
    product = sales_train_evaluation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    # merge with product table
    test1 = test1.merge(product, how = 'left', on = 'id')
    test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test2 = test2.merge(product, how = 'left', on = 'id')
    test2['id'] = test2['id'].str.replace('_validation','_evaluation')
    
    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    
    sales_train_evaluation['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'
    
    data = pd.concat([sales_train_evaluation, test1, test2], axis = 0)
    del sales_train_evaluation, test1, test2
    print(data.shape)
    
    # drop some calendar features
    print(calendar.columns)
    calendar.drop(['weekday'], inplace = True, axis = 1)
    
    # delete test2 for now, test2 to be predicted when full data is available. For now, predict on test1.
    data = data[data['part'] != 'test1']
    
    if merge:
        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
        data.drop(['d', 'day'], inplace = True, axis = 1)
        # get the sell price data (this feature should be very important)
        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    else: 
        print('Merge failed!')
    gc.collect()
    return data


def predict(test, submission, data):
    print('\n\nCreating prediction file !')
    predictions = test[['id', 'date', 'demand']]
    
    # Revert back - Vertical to horizontal transformation
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    # For concating evaluation rows from submission file, as it is
    evaluation_rows = [row for row in submission['id'] if 'validation' in row]
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv('submission.csv', index = False)

    # COMPARING RESULTS
    # data[data.demand>0].sort_values('demand', ascending=False).head()
    print('\nprevious and new results comparison for one of the entries')
    print('\\'*10)
    print(data[data.id == 'FOODS_3_090_CA_3_evaluation'][['id', 'demand']].head())
    print(predictions[predictions.id == 'FOODS_3_090_CA_3_evaluation'].head())
    print('\\'*10)


def model_and_predict(data):
    print('\n\nRunning transform_train_and_eval')
    sell_prices_df = pd.read_pickle("./data/sell_prices_df.pkl.compress", compression="gzip")
    calendar_df = pd.read_pickle("./data/calendar_df.pkl.compress", compression="gzip")
    test = run_lgb(data, calendar_df, sell_prices_df)
    submission_df = pd.read_csv('./input/sample_submission.csv')
    predict(test, submission_df, data)

start = datetime.now()
# calendar_df, sell_prices_df, sales_train_evaluation_df, submission_df = read_data()
# data = melt_and_merge(calendar_df, sell_prices_df, sales_train_evaluation_df, submission_df, nrows = 27500000, merge = True) # Use one year data only
# data.to_pickle('./data/data.pkl.compress', compression='gzip')
# data = pd.read_pickle("./data/data.pkl.compress", compression="gzip")
# print('data.pkl.compress pickle loaded \n')
# print(data.head())
# data = m5_feature_engg(data)
# data = reduce_mem_usage(data)
# data.to_pickle('./data/m5_feature_engg.pkl.compress', compression='gzip')

data = pd.read_pickle("./data/m5_feature_engg.pkl.compress", compression="gzip")
print('m5_feature_engg pickle loaded \n')
print(data.head(3))

# RUN PROCESS
model_and_predict(data)
print(datetime.now(), start)
time_spent = datetime.now() - start
print('\nProcess complete in {} minutes\n'.format(time_spent.total_seconds()/60))

