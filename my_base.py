import pandas as pd
import numpy as np

import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import *

from xgboost import XGBClassifier

import xgboost as xgb
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

from numba import jit

n_splits = 5

# Evaluate Result

@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

## from zehhan
def gini_ze(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb_ze(pred, y):
    y = y.get_label()
    return 'gini', gini_ze(y, pred)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)


###Ensemble Generation
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T, id):
        t_X = X
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                #                y_holdout = y[test_idx]

                start = time.time()
                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                if str(clf).__contains__("XGB"):
                    print("goto xgb fit")
                    clf.fit(X_train, y_train)
                elif str(clf).__contains__("LGB"):
                    print("goto lgb fit")
                    clf.fit(X_train, y_train)
                else:
                    clf.fit(X_train, y_train)

                # cross_score = cross_val_score(clf, X_train, y_train, cv=n_splits, scoring='roc_auc')
                # print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
                print('using time : %5.1f min' % ((time.time() -start ) / 60) )
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=n_splits, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        pred = self.stacker.predict_proba(S_train)[:,1]
        tmp = pd.DataFrame()
        tmp['target'] = pred
        tmp['id'] = id
        tmp.to_csv('mybase_valid.csv', float_format='%.6f', index=False)
        print( "  Total Gini = ", eval_gini(y, pred) )

        res = self.stacker.predict_proba(S_test)[:,1]

        return res


###Data Preprocessing
train = pd.read_csv('../train/train.csv')
test = pd.read_csv('../test/test.csv')

id_train = train['id'].values
id_test = test['id'].values
target_train = train['target'].values
y_train = train['target']

print(y_train.head())

##remove not use feature
# remove target and id
train = train.drop(['target', 'id'], axis=1)
test = test.drop(['id'], axis=1)

# remove feature whith start with ps_calc_ and

train_calc_09 = train['ps_calc_09']
test_calc_09 = test['ps_calc_09']
train_calc_05 = train['ps_calc_05']
test_calc_05 = test['ps_calc_05']

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)
test = test.drop(col_to_drop, axis=1)


train = train.drop(['ps_ind_14'], axis=1)
test = test.drop(['ps_ind_14'], axis=1)

train = pd.concat([train, train_calc_09], axis=1)
test = pd.concat([test, test_calc_09], axis=1)
train = pd.concat([train, train_calc_05], axis=1)
test = pd.concat([test, test_calc_05], axis=1)

# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
#    ('ps_reg_01', 'ps_car_04_cat'),
]

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    train[name1] = train[f1].apply(lambda x: str(x)) + "_" + train[f2].apply(lambda x: str(x))
    test[name1] = test[f1].apply(lambda x: str(x)) + "_" + test[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train[name1].values) + list(test[name1].values))
    train[name1] = lbl.transform(list(train[name1].values))
    test[name1] = lbl.transform(list(test[name1].values))

##Missing data handle
# train = train.replace(-1, np.nan)
# test = test.replace(-1, np.nan)

##Outlier


##Categorical variable


cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]), prefix=column)
        #One-Hot Encoding:convert category to dummy/indicator variables
    train = pd.concat([train, temp], axis=1)
    train = train.drop([column], axis=1) #remove original one

for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]), prefix=column) #One-Hot Encoding
    test = pd.concat([test, temp], axis=1)
    test = test.drop([column], axis=1)




# Enocode data
# temp = train
# f_cats = [f for f in train.columns if "_cat" in f]
# for f in f_cats:
#     train[f + "_avg"], temp[f + "_avg"], test[f + "_avg"] = target_encode(
#                                                     trn_series=train[f],
#                                                     val_series=temp[f],
#                                                     tst_series=test[f],
#                                                     target=y_train,
#                                                     min_samples_leaf=200,
#                                                     smoothing=10,
#                                                     noise_level=0
#                                                     )
#
#     test = test.drop([f], axis=1)
#     train = train.drop([f], axis=1)
#    temp = temp.drop([f], axis=1)

car_age = train['ps_car_15']
car_age = 17 - car_age * car_age
train = pd.concat([train, car_age.rename('ps_car_15_age')], axis=1)
# train = train.drop(['ps_car_15'], axis=1)
car_age_t = test['ps_car_15']
car_age_t = 17 - car_age_t * car_age_t
test = pd.concat([test, car_age_t.rename('ps_car_15_age')], axis=1)
# test = test.drop(['ps_car_15'], axis=1)

car_age = train['ps_car_12']
car_age = car_age * car_age
train = pd.concat([train, car_age.rename('ps_car_12_square')], axis=1)
# train = train.drop(['ps_car_15'], axis=1)
car_age_t = test['ps_car_12']
car_age_t = car_age_t * car_age_t
test = pd.concat([test, car_age_t.rename('ps_car_12_square')], axis=1)
# test = test.drop(['ps_car_15'], axis=1)

car_age = train['ps_car_14']
car_age = car_age * car_age
train = pd.concat([train, car_age.rename('ps_car_14_square')], axis=1)
# train = train.drop(['ps_car_15'], axis=1)
car_age_t = test['ps_car_14']
car_age_t = car_age_t * car_age_t
test = pd.concat([test, car_age_t.rename('ps_car_14_square')], axis=1)
# test = test.drop(['ps_car_15'], axis=1)

car_age = train['ps_reg_03']
car_age = car_age * car_age
train = pd.concat([train, car_age.rename('ps_reg_03_square')], axis=1)
# train = train.drop(['ps_car_15'], axis=1)
car_age_t = test['ps_reg_03']
car_age_t = car_age_t * car_age_t
test = pd.concat([test, car_age_t.rename('ps_reg_03_square')], axis=1)
# test = test.drop(['ps_car_15'], axis=1)

cat_features = [a for a in train.columns if a.endswith('cat')]
print(cat_features)
# cat_features.remove('ps_car_11_cat')

# for column in cat_features:
#     temp = pd.get_dummies(pd.Series(train[column]))
#         #One-Hot Encoding:convert category to dummy/indicator variables
#     train = pd.concat([train, temp], axis=1)
#     train = train.drop([column], axis=1) #remove original one
#
# for column in cat_features:
#     temp = pd.get_dummies(pd.Series(test[column])) #One-Hot Encoding
#     test = pd.concat([test, temp], axis=1)
#     test = test.drop([column], axis=1)


print(train.values.shape, test.values.shape)

###Freture Engineering
onezero_featrue = {
    'ps_ind_01',
    'ps_ind_03',
#     'ps_ind_14',
    'ps_ind_15',
    'ps_reg_01',
    'ps_reg_02',
    'ps_car_11',
    'ps_car_12',
    'ps_car_13',
    'ps_car_15',
    'ps_car_15',
    'ps_car_12_square',
    'ps_car_14_square',
    'ps_reg_03_square'
}
for lable in onezero_featrue:
    train[lable] = minmax_scale(train[lable])
    test[lable] = minmax_scale(test[lable])
    # train[lable] = np.round(minmax_scale(train[lable]), 4)
    # featrue = train[lable].value_counts().sort_index()
    # featrue.plot(kind='bar',title=lable)
    # plt.show()

normalize_featrue = {
    'ps_calc_05',
    'ps_calc_09',
    'ps_car_14',
    'ps_reg_03'
}

scaler = StandardScaler()

# for lable in normalize_featrue:
#
#     train_t = scaler.fit_transform(train_t).toarray()
#     test_t = scaler.fit_transform(test_t).toarray()
#
#     train = pd.concat([train, train_t], axis=1)
#     train = train.drop([lable], axis=1)
#
#     test = pd.concat([test, test_t], axis=1)
#     test = test.drop([lable], axis=1)
#
#     # train[lable] = np.round(minmax_scale(train[lable]), 4)
    # featrue = train[lable].value_counts().sort_index()
    # featrue.plot(kind='bar',title=lable)
    # plt.show()

##Missing data handle
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

###Model Selection

## get this from zeeshan
params = {'eta': 0.02,
          'max_depth': 4,
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'seed': 99,
          'gpu_id' : 0,
          'max_bin' : 16,
          'tree_method' : 'gpu_hist',
          'silent': True}

x1, x2, y1, y2 = model_selection.train_test_split(train, y_train, test_size=0.25, random_state=99)

col = [c for c in x1.columns if c not in ['id','target']]

print("goto XGB Boost Part !!!")
# watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
# model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb_ze, maximize=True, verbose_eval=50, early_stopping_rounds=200)
# print("goto Predict ")
# # test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
# test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
# test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
# test_1 = pd.concat((test['target'], id_test['id']), axis=1)
# test_1[['id','target']].to_csv('xgb_submission.csv', index=False, float_format='%.5f')



# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99
lgb_params['device'] = 'gpu'
lgb_params['gpu_platform_id'] = 0
lgb_params['gpu_device_id'] = 0



lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99
lgb_params2['device'] = 'gpu'
lgb_params2['gpu_platform_id'] = 0
lgb_params2['gpu_device_id'] = 0


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['random_state'] = 99
lgb_params3['device'] = 'gpu'
lgb_params3['gpu_platform_id'] = 0
lgb_params3['gpu_device_id'] = 0

# from ul-hassan
lgb_params4 = {}
lgb_params4['n_estimators'] = 1450
lgb_params4['max_bin'] = 20
lgb_params4['max_depth'] = 6
lgb_params4['learning_rate'] = 0.25 # shrinkage_rate
lgb_params4['boosting_type'] = 'gbdt'
lgb_params4['objective'] = 'binary'
lgb_params4['min_data'] = 500         # min_data_in_leaf
lgb_params4['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params4['verbose'] = 0
lgb_params3['device'] = 'gpu'
lgb_params3['gpu_platform_id'] = 0
lgb_params3['gpu_device_id'] = 0

#xgboost
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 1000
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9
xgb_params['min_child_weight'] = 10
xgb_params['gamma'] = 0
xgb_params['nthread'] = 4
xgb_params['gpu_id'] = 0
xgb_params['max_bin'] = 16
xgb_params['tree_method'] = 'gpu_hist'
xgb_params['scale_pos_weight'] = 1
xgb_params['seed'] = 27


lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

lgb_model4 = LGBMClassifier(**lgb_params4)

xgb_model = XGBClassifier(**xgb_params)

#cat_model = CatBoostClassifier(**cat_params)

#rgf_model = RGFClassifier(**rgf_params)

#gb_model = GradientBoostingClassifier(max_depth=5)

#ada_model = AdaBoostClassifier()

xgb1 = XGBClassifier(
        learning_rate =0.05,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        gpu_id=0,
        max_bin=16,
        tree_method='gpu_hist',
        scale_pos_weight=1,
        seed=27)

log_model = LogisticRegression()

stack = Ensemble(n_splits=n_splits,
                 stacker = log_model,
                 base_models = (lgb_model,lgb_model2,lgb_model3, lgb_model4 ))

# stacker = log_model,
# base_models = (lgb_model, lgb_model2, lgb_model3, xgb_model))

y_pred = stack.fit_predict(train, target_train, test, id_train)

###Output resault

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('mybase_submit.csv', index=False)
