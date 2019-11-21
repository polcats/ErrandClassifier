def get_input(msg, min = None, max = None, mode = int):
    inp = -1
    while inp == -1:
        try:
            inp = mode(input(msg))
            if min != None and max != None:
                if inp < min or inp > max:
                    inp = -1
                    print("Min:Max values are " + str(min) + ":" + str(max))
        except:
            inp = -1
            print("Invalid character, please try again.")
    return inp
