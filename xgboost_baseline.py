# XGBoost baseline for feature engineering.
# 1. Read data.
# 2. Encode missing data.
# 3. Split training data to two parts for training and validation.
# 4. Predict the test data.
# 5. Output to file.
#
# Training result: [230] train-mae:0.066641 valid-mae:0.065194
# Public score: 0.0660345

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


print('Reading training data, properties and test data.')
train = pd.read_csv("../data/train_2016_v2.csv")
properties = pd.read_csv('../data/properties_2016.csv')
test = pd.read_csv('../data/sample_submission.csv')

print('Encoding missing data.')
for column in properties.columns:
    properties[column] = properties[column].fillna(-1)
    if properties[column].dtype == 'object':
        label_encoder = LabelEncoder()
        list_value = list(properties[column].values)
        label_encoder.fit(list_value)
        properties[column] = label_encoder.transform(list_value)

print('Creating training and validation data for xgboost.')
train_with_properties = train.merge(properties, how='left', on='parcelid')

train_index = []
valid_index = []
for i in range(len(train_with_properties)):
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
df_test = test.merge(properties, how='left', on='parcelid')
d_test = xgb.DMatrix(df_test[x_train.columns])

print('Predicting on test data.')
p_test = model.predict(d_test)
test = pd.read_csv('../data/sample_submission.csv')
for column in test.columns[test.columns != 'ParcelId']:
    test[column] = p_test

print('Writing to csv.')
test.to_csv('../data/xgb_starter.csv', index=False, float_format='%.4f')

print('Congratulation!!!')
