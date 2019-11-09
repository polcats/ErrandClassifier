import pandas as pd
# file_errands = "../data/errands.txt"
# file_non_errands = "../data/amazon_cells_labelled.txt"
file_combined ="../data/combined.txt"
file_stop_words ="../data/stopwords.txt"

# data_errands = pd.read_csv(file_errands, names=['sentence', 'label'], sep=',')
# data_non_errands = pd.read_csv(file_non_errands, names=['sentence', 'label'], sep=',')
data_combined = pd.read_csv(file_combined, names=['sentence', 'label'], sep=',')
data_stop_words = pd.read_csv(file_stop_words)

# data_set = data_errands+data_non_errands

# df = pd.concat(map(pd.read_csv, [file_errands, file_non_errands]))
# from sklearn.utils import shuffle

# df = pd.concat([data_errands, data_non_errands])
# df = shuffle(df)
# df.to_csv('../data/combined.txt')

# with open('../data/combined.txt', 'a') as f:
# 	f.write(str(df))

print(data_combined)