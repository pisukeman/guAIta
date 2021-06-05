import os

def load_imgs(path):
    image_files = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            image_files.append(path+'/'+file)
    return sorted(image_files)


