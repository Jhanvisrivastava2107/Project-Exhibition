import os

DATA_DIR = "data"

for class_name in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, class_name)

    if os.path.isdir(class_path):

        images = os.listdir(class_path)

        print(class_name, ":", len(images), "images")