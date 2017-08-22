# LightGBM baseline for feature engineering.
# 1. Read data.
# 2. Encode missing data.
# 3. Split training data to two parts for training and validation.
# 4. Predict the test data.
# 5. Output to file.
#
# baseline + without outliers + cross-validation
# Best CV score: ~0.05252

import numpy as np
import pandas as pd
import lightgbm as lgb
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
    if properties[column].dtype == 'object':
        properties[column].fillna(-1, inplace=True)
        label_encoder = LabelEncoder()
        list_value = list(properties[column].values)
        label_encoder.fit(list_value)
        properties[column] = label_encoder.transform(list_value)

print('Creating training and validation data for xgboost.')
train_with_properties = train.merge(properties, how='left', on='parcelid')

print('Dropping out outliers.')
train_with_properties = train_with_properties[
    train_with_properties.logerror > OUTLIER_LOWER_BOUND]
train_with_properties = train_with_properties[
    train_with_properties.logerror < OUTLIER_UPPER_BOUND]
print('New training data with properties without outliers shape: {}'
      .format(train_with_properties.shape))

x_train = train_with_properties.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_with_properties['logerror'].values
d_train = lgb.Dataset(x_train, label=y_train)

print('Training the model.')
# lightgbm params
params = dict()
params['max_bin'] = 10
params['learning_rate'] = 0.0021   # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'            # or 'mae'
params['sub_feature'] = 0.5        # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.85  # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512         # num_leaf
params['min_data'] = 500           # min_data_in_leaf
params['min_hessian'] = 0.05       # min_sum_hessian_in_leaf
params['verbose'] = 0

# cross valication
# https://www.kaggle.com/aharless/lightgbm-outliers-remaining-cv
print("Cross-validating LightGBM model ...")
cv_results = lgb.cv(params, d_train, num_boost_round=10000, early_stopping_rounds=100,
                    verbose_eval=10, nfold=FOLDS)
print('Current parameters:\n', params)
print('\nBest num_boost_round:', len(cv_results['l1-mean']))
print('Best CV score:', cv_results['l1-mean'][-1])
print('Best CV score Standard Variance:', cv_results['l1-stdv'][-1])

# train model with all train data
num_boost_rounds = int(round(len(cv_results['l1-mean']) * np.sqrt(FOLDS/(FOLDS-1.))))
model = lgb.train(params, d_train, num_boost_round=num_boost_rounds, verbose_eval=10)

print('Building test set.')
test['parcelid'] = test['ParcelId']
df_test = test.merge(properties, how='left', on='parcelid')
d_test = df_test[x_train.columns]  # not lgb.Dataset

print('Predicting on test data.')
p_test = model.predict(d_test)

test = pd.read_csv('../data/sample_submission.csv')
for column in test.columns[test.columns != 'ParcelId']:
    test[column] = p_test

print('Writing to csv.')
test.to_csv('../data/lgb_baseline.csv', index=False, float_format='%.4f')

print('Congratulation!!!')
