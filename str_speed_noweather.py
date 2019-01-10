import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
import time
import pickle

x_cols = ['pickup_hour',  'pickup_dayofweek', 'PULocationID']
y_col = ['speed']
train_file = '../data/my_Train_Set.csv'
val_file = '../data/my_Val_Set.csv'
val_df = pd.read_csv('../data/my_Val_Set.csv')
val_df  = val_df.dropna()
gbm=None
params = {
        'task': 'train',
        'application': 'regression',
        'boosting_type': 'gbdt',
        'learning_rate': 0.2,
        'num_leaves': 31,
        'tree_learner': 'serial',
        'min_data_in_leaf': 100,
        'metric': ['mape'],  # l1:mae, l2:mse
        'max_bin': 255,
        'num_trees': 300
    }

i=1
mape_train_list = []
mape_val_list = []
for sub_data in pd.read_csv(train_file, chunksize=900000):
    x_data = sub_data[x_cols]
    y_data = sub_data[y_col]
    lgb_train = lgb.Dataset(x_data, y_data)

    lgb_eval = lgb.Dataset(val_df[x_cols], val_df[y_col], reference=lgb_train)
    gbm = lgb.train(params = params,
                    train_set= lgb_train,
                    # num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=gbm,             
                    feature_name=x_cols,
                    early_stopping_rounds=40,

                    verbose_eval=False,
                    keep_training_booster=True)

    score_train = dict([(s[1], s[2]) for s in gbm.eval_train()])
    score_valid = dict([(s[1], s[2]) for s in gbm.eval_valid()])
    mape_train_list.append(score_train['mape'])
    mape_val_list.append(score_valid['mape'])
    print('Training Score：mape=%.4f'%(score_train['mape']))
    print('Validation Score：mape=%.4f' % (score_valid['mape']))
    i += 1
filename = '../model/noweather_incre_gbm_speed.sav'
pickle.dump(gbm, open(filename, 'wb'))
print(mape_train_list)
print(mape_val_list)
