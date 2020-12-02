import os
import shutil  # For moving files

# DIRS = r'put the path of the folder you want to organize here'
# MAIN = r'put the path of where you want to move them here'

# Will go through each folder and sub-folder in a specified directory
for root, subdirs, files in os.walk(DIRS):
    print("Root: ", root)
    print("Subdirs: ", subdirs)
    print("Files: ", files)

    # Will join the file path and the name of the file to form the full path. After that it moves the file
    for file in files:
        path = os.path.join(root, file)
        shutil.move(path, MAIN)

