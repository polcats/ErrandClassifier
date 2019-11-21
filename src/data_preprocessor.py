from input import get_input
import pandas as pd
from nltk.corpus import stopwords

file_training ="../data/combined_v2.txt"
data_training = pd.read_csv(file_training, names=['sentence', 'label'], sep='~')

stop = stopwords.words('english')
data_training['sentence'] = data_training.sentence.str.replace(r"[^\w\s]", "").str.lower()
data_training['sentence'] = data_training['sentence'].apply(
    lambda x: ' '.join([item for item in x.split() if item not in stop]))
# data_training.to_csv('../data/preprocessed_v2.txt')
