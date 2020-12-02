import os
from PIL import Image, ExifTags

img_folder = r"/home/rasmus/Desktop/Fra_Mobil"
img_contents = os.listdir(img_folder)

# Print out all Exif-data/metadata stored in each image
"""
for image in img_contents:
    full_path = os.path.join(img_folder, image)

    # Will open up the image and extract the Exif-data/metadata
    pil_img = Image.open(full_path)
    
    # Extracts the value stored at each key in the dictionary (value = Exif-data/metadata)
    exif = {ExifTags.TAGS[k]: v for k, v in pil_img._getexif().items() if k in ExifTags.TAGS}

    # The code right below can be used to get the tags needed to access specific data like GPS or Date info
    # 1)  exif = {ExifTags.TAGS[k]: k for k, v in pil_img._getexif().items() if k in ExifTags.TAGS}

    print(image)  # Image name
    print(exif)   # Associated Exif-data
"""


# Get specific Exif-data

for image in img_contents:
    full_path = os.path.join(img_folder, image)

    # Will open up the image and extract the date it was created using the TAG: 36867
    # Tag was found using "1)" from the code above
    pil_img = Image.open(full_path)._getexif()[36867]

    print(image)  # Image name
    print(pil_img)  # Associated Exif-data