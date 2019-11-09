
from data_preprocessor import * 
from model import *

current_model = "../models/90s/91.9 93.8 90.1 09-11-2019 20-28-22.hdf5"

def do_action(inp):
	if inp == 0:
		print("Exiting...")
		exit()
	elif inp == 1:
		generate_model(data_training)
	elif inp == 2:
		runs = int(get_input("How many models do you want to create?: ", 1, 100))
		while runs > 0:
			runs = runs - 1
			generate_model(data_training)
	elif inp == 3:
		retrain_model(current_model, data_training)
	elif inp == 4:
		runs = int(get_input("How many retrains do you want to do?: ", 1, 100))
		while runs > 0:
			runs = runs - 1
			retrain_model(current_model, data_training)
	elif inp == 5:
		runs = int(get_input("How many loops do you want to do?: ", 1, 100))
		retrain_model(current_model, data_training, runs)
	elif inp == 6:
		model_summary(current_model);
	else:
		print("Invalid input.")

def print_actions():
	print("\n")
	print("0. Exit")
	print("1. Generate a new model")
	print("2. Generate multiple models")
	print("3. Retrain a model")
	print("4. Retrain multiple times")
	print("5. Retrain multiple loops")
	print("6. Show current model summary")

def get_input(msg, min = None, max = None):
	inp = -1;
	while inp == -1:
		try:
			inp = int(input(msg))
			if min != None and max != None:
				if inp < min or inp > max:
					inp = -1
					print("Min:Max values are " + str(min) + ":" + str(max))
		except:
			inp = -1
			print("Invalid character, please try again.")
	return inp


while True:
	print_actions()
	do_action(int(get_input("Please choose what to do: ")))

