import pandas as pd
from nltk.corpus import stopwords

file_training ="../data/combined.txt"
data_training = pd.read_csv(file_training, names=['sentence', 'label'], sep='~')

# print(data_training)

stop = stopwords.words('english')
data_training['sentence'] = data_training.sentence.str.replace("[^\w\s]", "").str.lower()
data_training['sentence'] = data_training['sentence'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
# data_training.to_csv('../data/preprocessed.txt')

# print(data_training)


#to do
'''
replace symbols with nothing
switch all to lowercase
'''