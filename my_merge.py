import pandas as pd
import numpy as np
import time
from collections import defaultdict, Counter
from glob import glob
import sys
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier

# glob_files = sys.argv[1]
# loc_outfile = sys.argv[2]
loc_outfile = "mean_merge_submission.csv"

# files = "model_valid/*"
files = "submit/*"

def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
    j = 0
    for i, glob_file in enumerate(glob(glob_files)):
        if glob_file.endswith("csv"):
            print("parsing:", glob_file, "file num is ", str(j))
            res = pd.read_csv(glob_file)
            if j == 0:
                base = res
                j += 1
                continue
            base['target'] = base['target'] + res['target']
            j += 1
    if j > 0:
        print("J is ", j)
        base['target'] = base['target'] / (j + 1)
        base.to_csv(loc_outfile, index=False)


kaggle_bag(files, loc_outfile)

test = pd.read_csv(loc_outfile)
print(test.describe())
