from timestamp import *

import pandas as pd
from sklearn.utils import shuffle

# load data
file_errands = "../data/errands.txt"
file_errands_v2 = "../data/errands_v2.txt"
file_non_errands = "../data/amazon_cells_labelled.txt"
data_errands = pd.read_csv(file_errands, names=['sentence', 'label'], sep='~')
data_errands_v2 = pd.read_csv(file_errands_v2, names=['sentence', 'label'], sep='~')
data_non_errands = pd.read_csv(file_non_errands, names=['sentence', 'label'], sep='~')


# combined data
df = pd.concat([data_errands, data_non_errands, data_errands_v2])



# shuffle and save combined data set
df = shuffle(df)
df.to_csv("../tmp/combined_data " + str(date_str) + " .txt")
