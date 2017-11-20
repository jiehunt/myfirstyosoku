import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from glob import glob
from numba import jit
import seaborn as sns
import matplotlib.pyplot as plt
import os

df_train = pd.read_csv('./input/train.csv')
y_train = df_train['target'].values
#=========================================================
# I. Define metrics functions
#=========================================================
@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)
#__________________________
def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True
#__________________________
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

kfold = 5
#=========================================================
# II. Load models results
#=========================================================
files = "model_submit0/*"
testres_files = "model_testres/*"
# glob_files = sys.argv[1]

valid_files = []
testres_files = []

df_model = []
path_model = []
# for i, file in enumerate(glob(files)):
#     file_name = str(file)
#     if file.endswith("valid.csv"):
#         valid_files.append(file)
#         # new_name = file_name.replace("valid.csv", "submit.csv")
#         # testres_files.append(new_name)
#         res = pd.read_csv(file)
#         path_model.append(str(file))
#         df_model.append(res)

for i, file in enumerate(glob(files)):
    file_name = str(file)
    res = pd.read_csv(file)
    path_model.append(str(file))
    df_model.append(res)

print(valid_files)
print(testres_files)


# for i, valid_file in enumerate(glob(valid_files)):
#     print("parsing:", valid_file, "file num is ", str(i))
#     res = pd.read_csv(valid_file)
#     path_model.append(str(valid_file))
#     df_model.append(res)

print("get Model Result")
print(path_model)

NB_models = len(path_model)

#__________________________________________
# print("Individual gini's scores:")
# for i, model in enumerate(path_model):
#     print('{:<20} -> {:8.5f}'.format(model, gini_normalized(y_train,
#            list(df_model[i]['target']) )))
#
# print("Individual auc's scores:")
# for i, model in enumerate(path_model):
#     print('{:<20} -> {:8.5f}'.format(model, fast_auc(y_train,
#            list(df_model[i]['target']) )))

#=========================================================
# II.II  check feature's corr
#=========================================================
# submissions_path = "./model_submit1/*"
# all_files = os.listdir(submissions_path)
#
# # Read and concatenate submissions
# outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)\
#         for f in all_files]
concat_df = pd.concat(df_model, axis=1)
concat_df = concat_df.drop(['id'], axis=1)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

cor1 = concat_df.corr(method='pearson')
# cor1.to_csv("pearson_corr.csv", index=False)
print(cor1.head())
pd.scatter_matrix(cor1, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cor1, mask=np.zeros_like(cor1, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

cor2 = concat_df.corr(method='kendall')
# cor1.to_csv("pearson_corr.csv", index=False)
print(cor2.head())
pd.scatter_matrix(cor2, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cor2, mask=np.zeros_like(cor2, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

cor3 = concat_df.corr(method='spearman')
# cor1.to_csv("pearson_corr.csv", index=False)
print(cor3.head())
pd.scatter_matrix(cor3, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cor3, mask=np.zeros_like(cor3, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

plt.show()

# print("Pearson's correlation score: %0.5f" % concat_df.corr(method='pearson'))
# print("Kendall's correlation score: %0.5f" % concat_df.corr(method='kendall'))
# print("Spearman's correlation score: %0.5f" % concat_df.corr(method='spearman'))
# print(concat_df.head())
# concat_df.to_csv("victor_temp.csv", index=False)


#=========================================================
# III. Determine optimal weights
#=========================================================
# coefficients = np.zeros((kfold, NB_models))
# valid_sum = 0
# #_______________________________________________________________________
# sss = StratifiedKFold(n_splits = kfold, shuffle = True, random_state=33)
# for j, (trn_idx, vld_idx) in enumerate(sss.split(y_train, y_train)):
#     print(50*"-"+'\n[Fold {}/{}]'.format(j + 1, kfold))
#     target_train = y_train[trn_idx]
#     target_valid = y_train[vld_idx]
#     #____________________
#     train, valid = [], []
#     for k, model in enumerate(path_model):
#         train.append(df_model[k].loc[trn_idx])
#         valid.append(df_model[k].loc[vld_idx])
#     #____________________
#     def function_gini(x):
#         x = x / np.linalg.norm(x)
#         x = [max(val, 0) for val in x]
#         y_test = np.zeros(len(target_train))
#         for k in range(NB_models):
#             y_test += x[k]*np.array(train[k]['target'])
#
#         fct = gini_normalized(target_train, y_test)
#         return -fct
#     #_________________________________
#     x0 = [1 for _ in range(NB_models)]
#
#     res = minimize(function_gini, x0, method='nelder-mead', options={'xtol': 1e-2, 'disp': True})
#     coeffs = [max(val, 0) for val in res.x]
#     coefficients[j,:] = coeffs / np.linalg.norm(coeffs)
#     print('\nfinal vector: {}'.format(coefficients[j,:]))
#
#     y_test = np.zeros(len(target_valid))
#     for k in range(NB_models):
#         y_test += coefficients[j, k]*np.array(valid[k]['target'])
#
#     valid_sum += gini_normalized(target_valid, y_test)
#     print("valid's gini: {:<7.4f} \n".format(gini_normalized(target_valid, y_test)))
# #____________________________________________________________________
# print(50*'='+"\nOverall's gini: {:<8.5f}".format(valid_sum / kfold))
#
# #=========================================================
# # IV. Create submission file
# #=========================================================
# print(50*'-' + '\n Initial set of weigths: \n', coefficients)
# t = [coefficients[:,j].mean() for j in range(NB_models)]
# print(50*'-' + "\n Weights averaged over folds: \n {}".format(t) )
# #_____________________________
# y_test = np.zeros(len(y_train))
# for k in range(NB_models):
#     y_test += t[k]*np.array(df_model[k]['target'])
# #_____________________________________________________________________________________________________
# print("\n valid's gini with mean weights =====> {:<8.6f} \n".format(gini_normalized(y_train, y_test)))
# #___________________________________
# sub = [_ for _ in range(NB_models)]
# for j in range(NB_models):
#     fichier = testres_files[j]
#     # fichier = path_model[j]+'_pred_test.csv'.format(j+1, kfold)
#     print("Add res file ", fichier)
#     sub[j] = pd.read_csv(fichier)
# #__________________________
# id_test = sub[0]['id']
# avg = np.zeros(len(id_test))
# for i in range(NB_models):
#     avg += t[i] * sub[i]['target']
# #______________________
# submit = pd.DataFrame()
# submit['id'] = id_test
# submit['target'] = avg
# submit.to_csv('merge_submission.csv', index=False)
# print(submit.describe())

