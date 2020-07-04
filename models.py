import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import gc
from scipy.sparse import csr_matrix
from sklearn import preprocessing, metrics

# Custom Evaluation metric: Incorporated from Tsuru's (girmdshinsei) kernal
# https://www.kaggle.com/girmdshinsei/for-japanese-beginner-with-wrmsse-in-lgbm

NUM_ITEMS, DAYS_PRED = 30490, 28 #Values & days to predict from submission file.

# Load some old data for faster execution
sales_train_val  = pd.read_pickle("./data/sales_train_evaluation_df.pkl.compress", compression="gzip")
product = sales_train_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
data = pd.read_pickle("./data/m5_feature_engg.pkl.compress", compression="gzip")

weight_mat = np.c_[np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                   pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                   pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values,
                   np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
                   ].T
weight_mat_csr = csr_matrix(weight_mat)


def weight_calc(data, product):
    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    sales_train_val = pd.read_csv('./input/sales_train_evaluation.csv')
    d_name = ['d_' + str(i+1) for i in range(1941)]
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values

    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1942),(weight_mat_csr.shape[0],1)))
    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
    flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1942),(weight_mat_csr.shape[0],1)))<1
    
    sales_train_val = np.where(flag,np.nan,sales_train_val)
    print('sales_train_val')
    print(sales_train_val.shape)
    print(sales_train_val)
    
    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1941-start_no)
    print('weight1')
    print(weight1)

    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp[product.id].values

    print('df_tmp')
    print(df_tmp)
    
    weight2 = weight_mat_csr * df_tmp
    weight2 = weight2/np.sum(weight2)

    print('weight1', weight1)
    print('weight2', weight2)

    return weight1, weight2
weight1, weight2 = weight_calc(data, product)


def wrmsse(preds, data):
    # this function is calculate for last 28 days to consider the non-zero demand period
    
    y_true = data.get_label()
    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    num_col = DAYS_PRED
    
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    
    train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        , axis=1) / weight1) * weight2)
    
    return 'wrmsse', score, False


def run_lgb(data, calendar, prices):
    print('\nRunning lightgbm\n')
    features = [
        "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI", "sell_price", \
        # demand features.
        "shift_t28", "rolling_std_t7", "rolling_std_t30", "rolling_std_t90", "rolling_std_t180", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", \
        # price features
        "price_change_t1", "price_change_t365", "rolling_price_std_t7", \
        # time features.
        "year", "month", "dayofweek", \

        # "rolling_mean_t90", "rolling_mean_t180", "rolling_skew_t30", "rolling_kurt_t30", "rolling_price_std_t30", \
        # "is_month_end", "is_month_start", "is_weekend", "wday", \
        # "price_max", "price_min", "price_std", "price_mean", "price_norm", "price_nunique", \
        # "item_nunique", "price_momentum", "price_momentum_m", "price_momentum_y", 
    ]

    # going to evaluate with the last 28 days
    x_train = data[data['date'] <= '2016-04-24']
    y_train = x_train['demand']
    x_val = data[(data['date'] > '2016-04-24') & (data['date'] <= '2016-05-22')]
    y_val = x_val['demand']
    test = data[(data['date'] > '2016-05-22')]

    print('\nPrint values for one of the entries')
    print(data[data.id == 'FOODS_3_090_CA_3_evaluation'][['id', 'demand']].head())

    del data
    gc.collect()
    
    params = {
        # 'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'poisson',
        'n_jobs': -1,
        'seed': 20,
        'learning_rate': 0.1,
        'alpha': 0.1,
        'lambda': 0.1,
        'bagging_fraction': 0.66,
        'bagging_freq': 2, 
        'colsample_bytree': 0.77
        }

    train_set = lgb.Dataset(x_train[features], y_train)
    val_set = lgb.Dataset(x_val[features], y_val)
    
    del x_train, y_train

    # model = lgb.train(params, train_set, num_boost_round = 5, early_stopping_rounds = 5
    # , valid_sets = [train_set, val_set], verbose_eval = 1, feval=wrmsse)
    model = lgb.train(
        params, train_set, num_boost_round = 1000, early_stopping_rounds = 250,
        valid_sets = [train_set, val_set], verbose_eval = 20
        )

    # model = lgb.train(
    #     params, train_set, num_boost_round = 1000, early_stopping_rounds = 200, 
    #     valid_sets = [train_set, val_set], verbose_eval = 20, feval=wrmsse
    #     )
    print('Saving model\n')
    joblib.dump(model, './data/lgbm_0.sav')
    # m = joblib.load('./data/lgbm_0.sav')
    # zipped = zip(features, m.feature_importance())
    # print([(k,v) for k,v in sorted(zipped, key=lambda x: x[1])])
    
    val_pred = model.predict(x_val[features], num_iteration=model.best_iteration)
    val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print('val_pred', val_pred)
    print('val_score', val_score)
    print(f'Our val wrmsse score is {val_score}')
    y_pred = model.predict(test[features], num_iteration=model.best_iteration)
    test['demand'] = y_pred

    return test
