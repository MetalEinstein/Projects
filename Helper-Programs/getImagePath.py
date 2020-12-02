import os  # Allows us to access things in our operating system

# Type in the full path to the folder between ""
# r gives the raw path without escape characters
img_folder = r""
img_contents = os.listdir(img_folder)

# For all images in the folder
for image in img_contents:
    print(image)  # Will print the image name
    full_path = os.path.join(img_folder, image)
    print(full_path)
