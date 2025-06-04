class Animal:
    def __init__(self , name , category , desc , type):
        self.name = name
        self.category = category
        self.description = desc
        self.type_of_category = type
    



import os

def count_images(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            image_files.append(filename)

    image_files.sort()  # Sort filenames alphabetically
    return len(image_files), image_files

def cat_description():
    desc = []
    for i in range(1, 11):
        if i == 1:
            desc.append("This is a black cat on wood in hous.")
        elif i == 2:
            desc.append("This is a black cat with orange eyes.")
        elif i == 3:
            desc.append("This is a black cat with scary mood.")
        elif i == 4:
            desc.append("This is a black cat on grass in angary mood.")
        elif i == 5:
            desc.append("This is a gray cat with orange or yellow eyes.")
        elif i == 6:
            desc.append("This is a white cat in white place.")
        elif i == 7:
            desc.append("This is a white cat with black eyes.")
        elif i == 8:
            desc.append("This is a white cat.")
        elif i == 9:
            desc.append("This is a white cat on pillow")
        elif i == 10:
            desc.append("This is a white cat on grass")
        elif i == 11:
            desc.append("This is a white cat on pillow with blue eyes.")
    return desc
        


import datetime

# ---- Logger Class ----
class Logger:
    def __init__(self, log_file="logfile.txt"):
        self.log_file = log_file

    def log(self, level, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level.upper()}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(entry)

    def info(self, message):
        self.log("INFO", message)

    def warning(self, message):
        self.log("WARNING", message)

    def error(self, message):
        self.log("ERROR", message)



# import cv2

# img = cv2.imread('images/cats/whiteCat6.jpg')

# cv2.imshow('x' , img)
# cv2.waitKey()
# cv2.destroyAllWindows()