import pandas as pd
file_errands = "../data/errands.txt"
file_non_errands = "../data/amazon_cells_labelled.txt"
file_stop_words ="../data/stopwords.txt"

data_errands = pd.read_csv(file_errands, names=['sentence', 'label'], sep=',')
data_non_errands = pd.read_csv(file_non_errands, names=['sentence', 'label'], sep=',')
data_stop_words = pd.read_csv(file_stop_words)

# print (data_errands)
# print (data_non_errands)
# print (data_stop_words)