
from data_preprocessor import *
from input import *
from model import *

def do_action(inp):
    if inp == 0:
        print("Exiting...")
        exit()
    elif inp == 1:
        generate_model(data_training)
    elif inp == 2:
        runs = int(get_input("How many models do you want to create?: ", 1, 100))
        acc = float(get_input("What is the target accuracy?: ", 1, 100, mode = float))
        while runs > 0:
            runs = runs - 1
            avg_acc = float(generate_model(data_training, acc))
            if avg_acc >= acc:
            	print("Target model with accuracy " + str(acc) + " generated.")
            	break
    elif inp == 3:
        retrain_model(data_training)
    elif inp == 4:
        runs = int(get_input("How many retrains do you want to do?: ", 1, 100))
        while runs > 0:
            runs = runs - 1
            retrain_model(data_training)
    elif inp == 5:
        runs = int(get_input("How many loops do you want to do?: ", 1, 100))
        retrain_model(data_training, runs)
    elif inp == 6:
        model_summary();
    elif inp == 7:
        # predict_file = get_input("Please enter the file name: ", mode = str)
        # model = load_handler("")
        model_predict(predict_file = "../data/jobs.csv", model_file = "../models/threshold_target/98.2 100. 96.5 12-11-2019 16-36-00.hdf5");
    else:
        print("Invalid input.")

def print_actions():
    print("\n")
    print("0. Exit")
    print("1. Generate a new model")
    print("2. Generate multiple models")
    print("3. Retrain model")
    print("4. Multiple retrain and save per training")
    print("5. Multiple retrain and save final result")
    print("6. Show current model summary")
    print("7. Predict classes")
    print("\n")

while True:
    print_actions()
    do_action(int(get_input("Please choose what to do: ")))

