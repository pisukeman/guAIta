from fastai.vision.all import *
import os
from datetime import datetime



def load_imgs(path):
    image_files = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            image_files.append(path+'/'+file)
    return image_files

def predictDay(model_file,images_path,threshold=0.85):
    learn = load_learner(model_file)
    path = images_path
    detections = open(path+"detections.txt","w")

    now = datetime.now()
    detections.write("guAIta inference\n") 
    detections.write("model:"+model_file+"\n")
    detections.write("execution:"+now.strftime("%d/%m/%Y %H:%M:%S")+"\n")

    test_path_lily = path
    test_img_lily = load_imgs(test_path_lily)
    uploader = SimpleNamespace(data = test_img_lily)
    for i in range(len(test_img_lily)):
        img = PILImage.create(uploader.data[i])
        predict = learn.predict(img)
        if (predict[0]=="meteor" and predict[2][0]>threshold):
            detections.write(f"loop {i}, {uploader.data[i]} , {predict[0]} , {predict[2][0]} , {predict[2][1]}"+"\n")
    detections.close() 

predictDay("C:/Development/meteor_detector/dataset/v7_wrong/guAIta_latest_version.pkl","C:/Development/meteor_detector/dataset/vDef_dia0504/test/")