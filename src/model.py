import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

from timestamp import *

from keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


epochs = 10
batch_size = 10
maxlen = 100
embedding_dim = 50
num_filters = 32
kernel_size = 3
current_model = ""

def generate_model(data_training, save_threshold = 0):
    model, train_acc, test_acc = train_model(data_training);
    print_accuracy(train_acc, test_acc)
    save_model(model, train_acc, test_acc, save_threshold)

    return avg_accuracy(train_acc, test_acc)

def train_model(data_training, model = None, show_progress = 0):
    sentences = data_training['sentence'].values
    labels = data_training['label'].values

    # training data split
    sentences_train, sentences_test, label_train, labels_test = train_test_split(
        sentences, labels, test_size=0.25, random_state=1000)

    # tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    tokens_train = tokenizer.texts_to_sequences(sentences_train)
    tokens_test = tokenizer.texts_to_sequences(sentences_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences with zeros
    tokens_train = pad_sequences(tokens_train, padding='post', maxlen=maxlen)
    tokens_test = pad_sequences(tokens_test, padding='post', maxlen=maxlen)

    # Create the model
    if model == None:
        model = create_model(vocab_size)

    hist = model.fit(tokens_train, label_train, epochs=epochs, batch_size=batch_size, verbose = show_progress)

    train_acc = (hist.history['accuracy'][epochs - 1])
    test_acc = test_model(model, tokens_test, labels_test)

    return model, train_acc, test_acc

def _create_model():
    model = Sequential()
    model.add(Dense(150, input_dim=maxlen, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='../model.png')

    return model

def create_model(vocab_size, show_summary = False):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if show_summary == True:
        model.summary()

    plot_model(model, show_shapes=True, to_file='../model.png')

    return model

def test_model(model, tokens_test, labels_test):
    _, test_acc = model.evaluate(tokens_test, labels_test)
    return test_acc

def save_model(model, train_acc, test_acc, save_threshold = 0):
    global current_model
    model_name = get_acc(train_acc) + " " + get_acc(test_acc) + " " + str(date_str);
    model_dir="../models/"

    test_train_avg = avg_accuracy(train_acc, test_acc)
    if float(test_train_avg) >= save_threshold:
        model_dir += "threshold_target/"
        filename = str(model_dir) + "" + str(test_train_avg) + " " + str(model_name) + ".hdf5"
        model.save(filename)
        current_model = filename
        print("Current model : " + str(current_model))

def retrain_model(data_training, times = 1, file_name = None):
    file_name = get_current_model(file_name)

    model = load_handler(file_name)
    if model == False:
        return
    model.summary()
    train_acc = 0
    test_acc = 0

    while times > 0:
        times = times - 1
        data_training = shuffle(data_training)
        model, train_acc, test_acc = train_model(data_training)
        print_accuracy(train_acc, test_acc)

    save_model(model, train_acc, test_acc)

def model_summary(file_name = None):
    file_name = get_current_model(file_name)

    model = load_handler(file_name)
    if model == False:
        return
    model.summary()

def load_handler(file_name):
    try:
        model = load_model(file_name)
        return model
    except:
        print("Error: Cannot load the model " + str(file_name))
        return False

def get_current_model(file_name):
    global current_model
    if file_name == None:
        return current_model
    else:
        return file_name

def avg_accuracy(train_acc, test_acc):
     return str(((train_acc + test_acc) * 100) / 2.0)[0:4]

def print_accuracy(train_acc, test_acc):
    print("\n")
    print("Avg Acc: " + avg_accuracy(train_acc, test_acc))
    print('Test Acc:    ' + get_acc(test_acc))
    print("Train Acc: " + get_acc(train_acc))

def get_acc(acc):
    return str(acc * 100)[0:4]
