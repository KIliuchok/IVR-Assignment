# Computer Vision and Robot Control

The main code to be executed for the assignment is located in files image1.py, image2.py and target_move.py.

image2.py has to be run first with the rosrun command because it creates topics which are later subscribed in image1.py. 

Both image1.py and image2.py have to be run from src directory because they have to load templates from path which is acquired with os.getcwd() function.
