import pandas as pd
from sklearn.utils import shuffle

# load data
file_errands = "../data/errands.txt"
file_non_errands = "../data/amazon_cells_labelled.txt"
data_errands = pd.read_csv(file_errands, names=['sentence', 'label'], sep='~')
data_non_errands = pd.read_csv(file_non_errands, names=['sentence', 'label'], sep='~')

# save combined data set
df = pd.concat([data_errands, data_non_errands])
df = shuffle(df)
df.to_csv('../data/combined.txt')