import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

from datetime import datetime

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

def generate_model(data_training):
	model, train_acc, test_acc = train_model(data_training);
	print_accuracy(train_acc, test_acc)
	save_model(model, train_acc, test_acc)

def train_model(data_training, model = None):
	sentences = data_training['sentence'].values
	labels = data_training['label'].values

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

	train_acc = (hist.history['accuracy'][epochs - 1])
	test_acc = test_model(model, tokens_test, labels_test)

	return model, train_acc, test_acc

def create_model():
	model = Sequential()
	model.add(Dense(150, input_dim=maxlen, activation='relu'))
	model.add(Dense(300, activation='relu'))
	# model.add(Dense(250, activation='relu'))
	model.add(Dense(15, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	plot_model(model, show_shapes=True, to_file='../model.png')

	return model

def test_model(model, tokens_test, labels_test):
	_, test_acc = model.evaluate(tokens_test, labels_test)
	return test_acc

def save_model(model, train_acc, test_acc):
	now = datetime.now()
	date_str = now.strftime("%d-%m-%Y %H-%M-%S")
	model_name = get_acc(train_acc) + " " + get_acc(test_acc) + " " + str(date_str);

	model_dir="../models/"

	test_train_avg = avg_accuracy(train_acc, test_acc)
	if float(test_train_avg) >= 90:
		model_dir += "90s/"

	model.save(str(model_dir) + "" + str(test_train_avg) + " " + str(model_name) + ".hdf5")

def retrain_model(file_name, data_training, times = 1):
	model = load_model(file_name)
	model.summary()
	train_acc = 0
	test_acc = 0

	while times > 0:
		times = times - 1
		data_training = shuffle(data_training)
		model, train_acc, test_acc = train_model(data_training)
		print_accuracy(train_acc, test_acc)

	save_model(model, train_acc, test_acc)

def model_summary(file_name):
	model = load_model(file_name)
	model.summary()

def avg_accuracy(train_acc, test_acc):
 	return str(((train_acc + test_acc) * 100) / 2.0)[0:4]

def print_accuracy(train_acc, test_acc):
	print("Avg Acc: " + avg_accuracy(train_acc, test_acc))
	print('Test Acc:	' + get_acc(test_acc))
	print("Train Acc: " + get_acc(train_acc))

def get_acc(acc):
	return str(acc * 100)[0:4]