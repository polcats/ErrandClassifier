import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

from datetime import datetime

from keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

epochs = 10
batch_size = 10
embedding_dim = 50
maxlen = 100
output_file = 'output.txt'

def generate_model(sentences, labels):
	model, train_acc, test_acc = train_model(sentences, labels);
	print_accuracy(train_acc, test_acc)
	save_model(model, test_acc, train_acc)

def create_model():
	model = Sequential()
	model.add(Dense(250, input_dim=maxlen, activation='relu'))
	model.add(Dense(300, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	plot_model(model, show_shapes=True, to_file='../model.png')

	return model

def train_model(sentences, labels, model = None):
	# training data split
	sentences_train, sentences_test, label_train, labels_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

	# tokenize words
	tokenizer = Tokenizer(num_words=5000)
	tokenizer.fit_on_texts(sentences_train)
	tokens_train = tokenizer.texts_to_sequences(sentences_train)
	tokens_test = tokenizer.texts_to_sequences(sentences_test)

	# Pad sequences with zeros
	tokens_train = pad_sequences(tokens_train, padding='post', maxlen=maxlen)
	tokens_test = pad_sequences(tokens_test, padding='post', maxlen=maxlen)

	# Create the model
	if model == None:
		model = create_model()

	hist = model.fit(tokens_train, label_train, epochs=epochs, batch_size=batch_size)

	train_acc = (hist.history['accuracy'][epochs - 1]) * 100
	test_acc = test_model(model, tokens_test, labels_test)

	return model, train_acc, test_acc

def print_accuracy(train_acc, test_acc):
	print('Test Accuracy: %.2f' % (test_acc))
	print("Training accuracy: " + str(train_acc)[0:4])


def test_model(model, tokens_test, labels_test):
	_, test_accuracy = model.evaluate(tokens_test, labels_test)
	accuracy = test_accuracy * 100
	return accuracy

# Save Model with Timestamp
def save_model(model, train_acc, test_acc):
	now = datetime.now()
	date_str = now.strftime("%d-%m-%Y %H-%M-%S")
	model_name = str(train_acc)[0:4] + " " + str(test_acc)[0:4] + " " + str(date_str);
	model.save("../models/" + str(model_name) + ".hdf5")