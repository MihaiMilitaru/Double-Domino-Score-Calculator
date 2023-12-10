# Double-Domino-Score-Calculator

1. Libraries used for this project.

- CV2
- Numpy

2. Folders structure

The archive contains the following:

- solution.py (a .py file that solves Task1, Task2, Task3)
The solution.py file reads from folder 'antrenare' the images for the 5 training sets and mutari.txt files for each set. 
For example for the first set it reads the images 1_01.jpg, 1_02.jpg and so on to 1_20.jpg and 1_mutari.txt for the players order.
Other example for the second set it reads the images 2_01.jpg, 2_02.jpg and so on to 2_20.jpg and 2_mutari.txt for the players order.
All the 100 training images and the 5 index_mutari.txt files for the input are in the folder 'antrenare'. This folder 'antrenare' is the same that was downloaded from dropbox archive.
This folder will not be included in this archive because it is part of the homework archive from dropbox.
You can replace the path of the 'antrenare' folder at lines 152 and 237 at the following part of the code with your local path for training data:

line 152: image_name = 'antrenare/' + str(set) + '_' + str(image_index) + '.jpg'
line 237: fisier_mutari = 'antrenare/' + str(set) + '_mutari.txt'

If you want to run the solution.py file for only one set change line 7:

line 7: sets_games = [1, 2, 3, 4, 5]

Here I ran the python script on all the 5 sets from the training set. 
If you want to run the python file for only one set change the array to set_games = [1].

- imagini_auxiliare (a folder with the empty board game image and two other images with all pieces placed on the gameboard)
This folder contains the following:
	- 01.jpg (an image with the empty board)
	- 02.jpg (an image with the board with all the pieces placed horizontally)
	- 03.jpg (an image with the board with all the pieces placed vertically)

- my_tests (a folder that contains all the output files in .txt format for submission)

This folder contains all the output files for all the sets. For the first set it contains the files 1_01.txt, 1_02.txt and so on to 1_20.txt. The same rule applies for all the sets, from 1 to 5.
The folder is empty at first in the achive and after you run the script, the text files for output are saved in this folder.

Note: The script may run slower but there are prints in the console. It prints the pieces of dominos found and for every new set started there is an announcement. There is also a print when the current set is finished.

- the documentation of the project with details how i handled and solved Task1, Task2 and Task3 in PDF format.





