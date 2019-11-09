from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

epochs = 10
batch_size = 10
embedding_dim = 50
maxlen = 100
output_file = 'output.txt'

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid')) #ito yung outputa later
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # binary_crossentropy
                  metrics=['accuracy'])
    return model
