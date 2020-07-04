
import pandas as pd
import numpy as np

def m5_feature_engg(data):
    # data.columns = 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'demand',
    #    'part', 'date', 'wm_yr_wk', 'wday', 'month', 'year', 'event_name_1',
    #    'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX',
    #    'snap_WI', 'sell_price'

    # rolling demand features
    print('\n\nRunning m5 Feat engg')
    for val in [28, 29, 30]:
        data[f"shift_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
    for val in [7, 30, 60, 90, 180]:
        data[f"rolling_std_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).std())
    for val in [7, 30, 60, 90, 180]:
        data[f"rolling_mean_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).mean())

    # measure spread
    data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform( lambda x: x.shift(28).rolling(30).skew())
    data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).kurt())
    
    # price features
    for a in ['min', 'max', 'std', 'mean']:
        data['price_%s' % a] = data.groupby(['store_id','item_id'])['sell_price'].transform(a)
    data['price_norm'] = data['sell_price']/data['price_max']
    data['price_nunique'] = data.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)

    # time features
    data['date'] = pd.to_datetime(data['date'])
    attrs = [
        "year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start"
        , "is_quarter_end", "is_quarter_start", "is_month_end", "is_month_start"
        ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        data[attr] = getattr(data['date'].dt, attr).astype(dtype)
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)
    return data
    