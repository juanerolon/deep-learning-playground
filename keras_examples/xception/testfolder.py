import os

current_dir = os.path.dirname(os.path.realpath(__file__))

if not os.path.isdir(current_dir + "/saved_models"):
    os.system('mkdir saved_models')
    print("Success in creating folder 'saved_models'\n")
else:
    print("'The saved_models folder is already present in script directory\n'")
    pass
