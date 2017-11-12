import pandas as pd 
import numpy as np
from scipy.stats.mstats import hmean
from glob import glob

#Download csv files from the following links. Make sure kernal's ID is the same as the one in the link.

#Author: Andy Harless
#File: xgb_submit.csv
#Link: https://www.kaggle.com/aharless/xgboost-cv-lb-284?scriptVersionId=1673404

#Author: Vladimir Demidov
#File: stacked_1.csv
#Link: https://www.kaggle.com/yekenot/simple-stacker-lb-0-284?scriptVersionId=1665392

#Author: Keui Shen Nong
#File: Froza_and_Pascal.csv
#Link: https://www.kaggle.com/kueipo/base-on-froza-pascal-single-xgb-lb-0-284?scriptVersionId=1679911

#Author: areeves87
#File: median_rank_submission.csv
#Link: https://www.kaggle.com/areeves87/aggregate-20-kernel-csvs-by-median-rank-lb-285


#Read csv files

files = "model_submit/*"
testres_files = "model_submit/*"
# glob_files = sys.argv[1]

testres_files = []

df_model = []
path_model = []
for i, file in enumerate(glob(files)):
    file_name = str(file)
    testres_files.append(file_name)
        # res = pd.read_csv(file_name)
        # path_model.append(str(file))
        # df_model.append(res)

print(testres_files)

# stacked_1 = pd.read_csv('../input/stacked1/stacked_1.csv')
# xgb_submit = pd.read_csv('../input/xgbsubmit/xgb_submit_1.csv')
# Froza_and_Pascal = pd.read_csv('../input/forza-and-pascal/Froza_and_Pascal.csv')
# median_rank_submission = pd.read_csv('../input/median-rank-submission/median_rank_submission.csv')

stacked_1 = pd.read_csv('./model_valid/andy_xgb_submit.csv')
# xgb_submit = pd.read_csv('./model_valid/mybase_submit.csv')
# Froza_and_Pascal = pd.read_csv('./model_valid/Keui_submit.csv')
# median_rank_submission = pd.read_csv('./model_valid/ievgen_submit.csv')

#concatenate target columns on the same Dataframe
# preds = pd.concat([stacked_1['target'], xgb_submit['target'],
#         Froza_and_Pascal['target'], median_rank_submission['target']])

# stacked_1 = pd.read_csv('./model_valid/andy_xgb_submit.csv')
for i, file in enumerate(testres_files):
    print("Parsing:  ",str(i), "  ", str(file))
    res = pd.read_csv(file)
    if i == 0:
        preds = res['target']
        continue
    preds = pd.concat([preds,res['target']])

# preds.to_csv("temp.csv",index=False)
print(preds.describe())

#Apply harmonic mean
preds = preds.groupby(level=0).apply(hmean)

# Create submission 
print(preds.head)
sub = pd.DataFrame()
sub['id'] = stacked_1['id']
sub['target'] = preds
print(sub.describe())
	
sub.to_csv('emsamble_harmonic.csv', index = False)
