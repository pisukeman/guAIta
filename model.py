from fastai.vision.all import *
import torch
import cv2 as cv
import os


print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))


def loadPositiveMeteors(folder,filename):
    file1 = open(folder+"/"+filename,"r")    
    content = file1.readlines()
    content = [x.strip() for x in content]
    file1.close()
    return content



meteors =loadPositiveMeteors('C:/Development/meteor_detector/dataset/','positives.txt')

def is_meteor(x): return x in meteors
    
#path="C:/Development/meteor_detector/dataset/v3_Processed"
path="C:/Development/meteor_detector/dataset/Fake_training"
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_meteor, num_workers=0) #item_tfms=Resize(224)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(1)
learn.export("model.pkl")
