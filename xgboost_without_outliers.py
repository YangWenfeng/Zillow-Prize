# Version 1: XGBoost without outlier.

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

OUTLIER_UPPER_BOUND = 0.4
OUTLIER_LOWER_BOUND = -0.4


print('Reading training data, properties and test data.')
train = pd.read_csv("../data/train_2016_v2.csv")
properties = pd.read_csv('../data/properties_2016.csv')
test = pd.read_csv('../data/sample_submission.csv')

print('Converting float64 to float32.')
for column, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[column] = properties[column].astype(np.float32)

print('Encoding missing data.')
for column in properties.columns:
    properties[column] = properties[column].fillna(-1)
    if properties[column].dtype == 'object':
        label_encoder = LabelEncoder()
        list_value = list(properties[column].values)
        label_encoder.fit(list_value)
        properties[column] = label_encoder.transform(list_value)

print('Combining training data with properties.')
train_with_properties = train.merge(properties, how='left', on='parcelid')
print('Original training data with properties shape: {}'
      .format(train_with_properties.shape))

print('Dropping out outliers.')
train_with_properties = train_with_properties[
    train_with_properties.logerror > OUTLIER_LOWER_BOUND]
train_with_properties = train_with_properties[
    train_with_properties.logerror < OUTLIER_UPPER_BOUND]
print('New training data with properties without outliers shape: {}'
      .format(train_with_properties.shape))

print('Creating training and test data for xgboost.')
x_train = train_with_properties\
    .drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_with_properties['logerror'].values

print('Splitting training data into train and valid parts.')
split = 80000
x_train, y_train, x_valid, y_valid = \
    x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Training the model')
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
params = {'eta': 0.06, 'objective': 'reg:linear', 'eval_metric': 'mae',
          'max_depth': 4, 'silent': 1}
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,
                verbose_eval=10)

print('Building test set.')
test['parcelid'] = test['ParcelId']
df_test = test.merge(properties, how='left', on='parcelid')
d_test = xgb.DMatrix(df_test[x_train.columns])

print('Predicting on test data.')
p_test = clf.predict(d_test)
test = pd.read_csv('../data/sample_submission.csv')
for column in test.columns[test.columns != 'ParcelId']:
    test[column] = p_test

print('Writing to csv.')
test.to_csv('../data/xgb_starter.csv', index=False, float_format='%.4f')

print('Congratulation!!!')
