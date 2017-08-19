# Add month column to baseline.
#
# Based on the "How does log error change with time" section in
# https://www.kaggle.com/philippsp/exploratory-analysis-zillow, it is sensitive
# on month, but the final result is worse than baseline.
#
# Training result: [192] train-mae:0.066781 valid-mae:0.065218
# Public score: 0.0658957

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


print('Reading training data, properties and test data.')
train = pd.read_csv("../data/train_2016_v2.csv")
properties = pd.read_csv('../data/properties_2016.csv')
test = pd.read_csv('../data/sample_submission.csv')

print('Encoding missing data.')
for column in properties.columns:
    if properties[column].dtype == 'object':
        properties[column].fillna(-1, inplace=True)
        label_encoder = LabelEncoder()
        list_value = list(properties[column].values)
        label_encoder.fit(list_value)
        properties[column] = label_encoder.transform(list_value)

print('Creating training and validation data for xgboost.')
train_with_properties = train.merge(properties, how='left', on='parcelid')
for i in xrange(len(train_with_properties)):
    date = train_with_properties.iloc[i]['transactiondate']
    # add month column
    train_with_properties.set_value(i, 'month', int(date[5:7]))

train_index = []
valid_index = []
for i in xrange(len(train_with_properties)):
    if i % 10 != 0:
        train_index.append(i)
    else:
        valid_index.append(i)
train_dataset = train_with_properties.iloc[train_index]
valid_dataset = train_with_properties.iloc[valid_index]

x_train = train_dataset.drop(
    ['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_dataset['logerror'].values
x_valid = valid_dataset.drop(
    ['parcelid', 'logerror', 'transactiondate'], axis=1)
y_valid = valid_dataset['logerror'].values

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

print('Training the model.')
# xgboost params
params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
}

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,
                  verbose_eval=10)

print('Building test set.')
test['parcelid'] = test['ParcelId']
test_1610 = test
test_1611 = test
test_1612 = test
df_test_1610 = test_1610.merge(properties, how='left', on='parcelid')
df_test_1610['month'] = 10
d_test_1610 = xgb.DMatrix(df_test_1610[x_train.columns])
df_test_1611 = test_1611.merge(properties, how='left', on='parcelid')
df_test_1611['month'] = 11
d_test_1611 = xgb.DMatrix(df_test_1611[x_train.columns])
df_test_1612 = test_1612.merge(properties, how='left', on='parcelid')
df_test_1612['month'] = 12
d_test_1612 = xgb.DMatrix(df_test_1612[x_train.columns])

print('Predicting on test data.')
p_test_1610 = model.predict(d_test_1610)
p_test_1611 = model.predict(d_test_1611)
p_test_1612 = model.predict(d_test_1612)
test = pd.read_csv('../data/sample_submission.csv')
# 2017 test cases are in the private board, which won't affect the ranking
# in public board.
for column in test.columns[test.columns != 'ParcelId']:
    if column == '201610' or column == '201710':
        test[column] = p_test_1610
    elif column == '201611' or column == '201711':
        test[column] = p_test_1611
    elif column == '201612' or column == '201712':
        test[column] = p_test_1612

print('Writing to csv.')
test.to_csv('../data/xgb_starter.csv', index=False, float_format='%.4f')

print('Congratulation!!!')
