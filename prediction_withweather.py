import pandas as  pd
import os
import numpy as np
import pickle

path = '../model/withweather'
model_file = os.listdir(path)
test_df = pd.read_csv('../data/testset_2150.csv')
print(len(test_df))
test_df = test_df.dropna()
print(len(test_df))
x_col = ['pickup_hour', 'Condition', 'Precip', 'Precip_Accum', 'pickup_dayofweek', 'PULocationID']
test_x = test_df[x_col]
test_y_s =test_df['speed']
test_y_c =test_df['count']

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 


for name in model_file:
    if name.endswith('speed.sav'):
        model = pickle.load(open(path+'/'+name,'rb'))
        pred_y = model.predict(test_x)
        pred_df = test_df[['pickup_date','pickup_hour']]
        # print(len(pred_y),len(pred_df))
        pred_df['speed']=  pred_y
        pred_df['speed']=test_y_s
        mape = mean_absolute_percentage_error(test_y_s,pred_y)
        print('The mape of '+name.replace('.sav','')+' is %f' % (mape))
        pred_df.to_csv('../prediction/'+name.replace('.sav','')+'_pre.csv')
    if name.endswith('count.sav'):
        model = pickle.load(open(path+'/'+name,'rb'))
        pred_y = model.predict(test_x)
        pred_df = test_df[['pickup_date', 'pickup_hour']]
        pred_df['count']=  pred_y
        pred_df['true']=test_y_c
        print(len(pred_y),len(test_y_c))
        mape = mean_absolute_percentage_error(test_y_c,pred_y)
        print('The mape of '+name.replace('.sav','')+' is %f' % (mape))
        pred_df.to_csv('../prediction/'+name.replace('.sav','')+'_pre.csv')





