from data_preprocessor import * 
from model import *

# data
sentences = data_training['sentence'].values
labels = data_training['label'].values

generate_model(sentences, labels)

