# Version 1: XGBoost without outlier.
# 1. Encode the missing data with LabelEncoder directly. Should try
# OneHotEncoder later.
# 2. Drop useless properties' feature inspired by kernel. Should dig out more in
# the following version.
# 3. Drop out outlier. The lower and upper bounds are gotten from kernel.
# [-0.3425, 0.4637] from Simple Exploration doesn't get better result.
# Maybe three categories assumption is not correct?
# 4. Use cross-validation whose parameters are gotten from kernel.
# 5. Let's try LightGBM in the following version.

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4
FOLDS = 5


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
x_train = train_with_properties.drop(
    ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
     'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = train_with_properties['logerror'].values
y_train_mean = np.mean(y_train)

print('Training the model with cross validation.')
d_train = xgb.DMatrix(x_train, y_train)

# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_train_mean,
    'silent': 1
}

# cross validation.
cv_result = xgb.cv(
    xgb_params, d_train, nfold=FOLDS, num_boost_round=350,
    early_stopping_rounds=50, verbose_eval=10, show_stdv=False)
num_boost_rounds = int(round(len(cv_result) * np.sqrt(FOLDS/(FOLDS-1))))
model = xgb.train(
    dict(xgb_params, silent=1), d_train, num_boost_round=num_boost_rounds)

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
