# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.special import expit # sigmoid

# All submission files were downloaded from different public kernels
# See the description to see the source of each submission file
# submissions_path = "../input/kaggleportosegurosubmissions"
submissions_path = "./model_submit0"
# submissions_path = "./model_submit"
all_files = os.listdir(submissions_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(submissions_path, f), index_col=0)\
        for f in all_files]
for file in all_files:
    print(str(file))
concat_df = pd.concat(outs, axis=1)
concat_df.to_csv("victor_temp.csv", index=False)
cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# Apply ranking, normalization and averaging
# concat_df["target"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)
# concat_df.drop(cols, axis=1, inplace=True)

## this is from Andy
logits = concat_df.applymap(lambda x: np.log(x/(1-x)))
stdevs = logits.std()
w = .2/stdevs
wa = (w*logits).sum(axis=1)/w.sum()

# Convert back to probabilities
result = wa.apply(expit)
print(result.describe())

pd.DataFrame(result,columns=['target']).to_csv("ensamble_logit.csv",float_format='%.6f')

# print(concat_df.describe())
# Write the output
# concat_df.to_csv("./ensamble_rank.csv")