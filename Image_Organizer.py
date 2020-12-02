import os
import datetime
from PIL import Image, ExifTags
import shutil  # For moving files

# OPDATE!!
# Create and move the files in the same function, will really reduce the code

# PROBLEM IF CANT CREATE OBJECT IT WILL TROW AN ERROR, HAPPENS WITH VIDEO FILES


class get_img_data:

    def __init__(self, path, date_year, date_month, date_week, time):
        self.path = path
        self.date_year = date_year
        self.date_month = date_month
        self.date_week = date_week
        self.time = time

    @classmethod
    def from_img(cls, initial_folder,  img):
        try:
            # Generates the initial image path
            full_path = os.path.join(initial_folder, img)

            # Will open up the image and extract the date and time it was created using the TAG: 36867
            pil_img = Image.open(full_path).getexif()[36867]

            # Splits the date and time into two elements in a list
            # The date and time is separated by a dash
            date, time = pil_img.split(" ")

            # Gets the year, month and week the image was taken
            year, month, week = get_img_data.split_date(date)

            # Returns the class object
            return cls(full_path, year, month, week, time)

        except IOError:  # IOError only works in Windows !!!PROBLEM!!!
            print("This might be a video: ", img)

    @staticmethod
    # Splits up the date and uses it to find the specific week day for that date. Returns year, month and week
    def split_date(date):
        year, month, day = date.split(":")
        week = datetime.date(int(year), int(month), int(day)).isocalendar()[1]
        return year, month, str(week)


# Checks if subfolders exit in the initial folder
def subfolder_check(initial_folder):
    for dirpath, subdir, files in os.walk(initial_folder):
        if len(subdir) > 0:
            return True
        else:
            return False


# Creates a list of image objects
def get_object_list():
    object_list = []
    for image in img_contents:
        object_list.append(get_img_data.from_img(initial_folder, image))
    return object_list


def create_folders(object_list, selection):

    def create_years():
        for objects in object_list:
            os.chdir(initial_folder)
            if not os.path.isdir(objects.date_year):  # If a folder/directory does not exit
                os.mkdir(objects.date_year)  # Make the folder
            else:
                pass

    def create_months():
        for objects in object_list:
            if objects.date_year in os.listdir(initial_folder):
                target_folder = os.path.join(initial_folder, objects.date_year)
                os.chdir(target_folder)
                if objects.date_month not in os.listdir(target_folder):
                    os.mkdir(objects.date_month)
                else:
                    pass
            else:
                pass

    def create_weeks():
        for objects in object_list:
            target_folder = os.path.join(initial_folder, objects.date_year, objects.date_month)
            if objects.date_week not in os.listdir(target_folder):
                os.chdir(target_folder)
                os.mkdir(objects.date_week)
            else:
                pass

    if selection == 0:
        create_years()
    elif selection == 1:
        create_years()
        create_months()
    else:
        create_years()
        create_months()
        create_weeks()


def move_files(object_list, selection):
    if selection == 0:
        for objects in object_list:
            target_folder = os.path.join(initial_folder, objects.date_year)
            shutil.move(objects.path, target_folder)

    elif selection == 1:
        for objects in object_list:
            target_folder = os.path.join(initial_folder, objects.date_year, objects.date_month)
            shutil.move(objects.path, target_folder)

    else:
        for objects in object_list:
            target_folder = os.path.join(initial_folder, objects.date_year, objects.date_month, objects.date_week)
            shutil.move(objects.path, target_folder)




initial_folder = r"U:\organizer_test"
img_contents = os.listdir(initial_folder)

# Checks for existing sub-folders
if not subfolder_check(initial_folder):

    # Creates a list of image objects
    object_list = get_object_list()

    selection = 2
    # Creates folders
    create_folders(object_list, selection)

    # Sort images
    #move_files(object_list, selection)
else:
    print("Sub-folders already exit, please remove them and try again")


# For debugging, moves all the files back into the initial folder
"""
for dirpath, subdir, files in os.walk(initial_folder):
    for file in files:
        image_path = os.path.join(dirpath, file)
        shutil.move(image_path, initial_folder)

"""









"""
list_dates_times = []  # For temporary storage of date and time

# The following code gets the image data and stores it in a dictionary
for image in img_contents:
    try:  # If the image name starts with 'I' it's a image
        full_path = os.path.join(initial_folder, image)

        # Will open up the image and extract the date and time it was created using the TAG: 36867
        pil_img = Image.open(full_path)._getexif()[36867]

        # Splits the date and time into two elements in a list
        # The date and time is separated by a dash
        list_dates_times = pil_img.split(" ")

        # Splits up the date into year, month and day. [0] is the full date
        year, month, day = list_dates_times[0].split(":")

        # Takes each image and checks if a folder for the year the image was taken exits
        # If not it creates a folder for the image and moves it into it
        os.chdir(initial_folder)
        if not os.path.isdir(year):  # If a folder/directory does not exit
            os.mkdir(year)  # Make the folder
            target_folder = os.path.join(initial_folder, year)
            shutil.move(full_path, target_folder)  # Move the first image with that particular date
        else:
            shutil.move(full_path, target_folder)  # Move any subsequent image with the same date into folder

    except IOError:  # IOError only works in Windows !!!PROBLEM!!!
        print("This might be a video: ", image)
"""