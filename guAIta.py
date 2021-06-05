from fastai.vision.all import *
from fastai import *
import cv2 as cv
import json
import base64
from datetime import datetime
import requests
from utilities import load_imgs
from guAItaConfig import *
import PySimpleGUI as sg
import time

class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)   
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

def printHeader():
    print("")
    print("  ██████╗ ██╗   ██╗ █████╗ ██╗████████╗ █████╗")
    print(" ██╔════╝ ██║   ██║██╔══██╗██║╚══██╔══╝██╔══██╗")
    print(" ██║  ███╗██║   ██║███████║██║   ██║   ███████║")
    print(" ██║   ██║██║   ██║██╔══██║██║   ██║   ██╔══██║")
    print(" ╚██████╔╝╚██████╔╝██║  ██║██║   ██║   ██║  ██║")
    print("  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝")
    print("Version 1.0.0 (Beta) - Created by David Regordosa @pisukeman")
    print("")
    now = datetime.now() 
    print("Starting guAIta at:"+now.strftime("%d/%m/%Y %H:%M:%S"))


def getLastFolder(folder=guAIta_main_folder):
    subfolders = sorted([ f.path for f in os.scandir(folder) if f.is_dir() ])
    if (len(subfolders)>0):
        return subfolders[-1]
    else:
        return ""

def guAItaDayAnalisis():
    printHeader()
    
    #Check if model is available. If not, download from S3 (latest_version always)
    try:
        learn = load_learner("guAIta_latest_version.pkl")
        print("Model available")
    except:
        print("Downloading Model")
        model = requests.get("https://guaita.s3-eu-west-1.amazonaws.com/models/guAIta_latest_version.pkl")
        open("guAIta_latest_version.pkl",'wb').write(model.content)
        model.close()
        print("Model downloaded")
        learn = load_learner("guAIta_latest_version.pkl")
    
    if guAIta_enable_telegram==True:
        guAIta_enable_telegram_txt = 'True'
    
    folder = getLastFolder(guAIta_main_folder)
    if folder=="":
        print("Error reading Folder")
        return False
    print ("Scanning folder: "+folder)        
    #Reading images in folder
    try:
        imgs = load_imgs(folder)
    except:
        print("Error reading images")
        return False
    tuple_list = [[imgs[i], imgs[i + 1]]
        for i in range(len(imgs) - 1)]

    now = datetime.now() 
    imgs_processed=0
    if guAIta_enable_telegram==True:
        json_log = json.dumps({'log':'Start','obs_id':guAIta_obs_id,})
        requests.post(guAIta_URL, json_log)

    for i in tuple_list:
        imgs_processed=imgs_processed+1
        imgName1 = i[0]
        imgName2 = i[1]
        img1=cv.imread(imgName1,cv.IMREAD_GRAYSCALE)
        img2=cv.imread(imgName2,cv.IMREAD_GRAYSCALE)
        img_subs = cv.subtract(img2,img1)
        (thresh, img_subs) = cv.threshold(img_subs, 3.5, 255, cv.THRESH_BINARY)
        img_subs = cv.resize(img_subs,(224,224),interpolation=cv.INTER_AREA)

        img = PILImage.create(img_subs)
        predict = learn.predict(img)

        if guAIta_enable_extra_log==True:
            print(f"{imgName2} , {predict[0]} , {predict[2][0]} , {predict[2][1]}")
        if (predict[0]=="meteor" and predict[2][0]>guAIta_threshold):
            print("Detection:")
            print(f"{imgName2} , {predict[0]} , {predict[2][0]} , {predict[2][1]}")   
            img_resized = cv.resize(img2,(224,224),interpolation=cv.INTER_AREA)
            id = imgName2[-18:-4]

            #Join the 2 images
            numpy_horizontal = cv.hconcat([img_resized,img_subs])
            retval, buffer = cv.imencode('.jpg',numpy_horizontal)
            imgb64_string = base64.b64encode(buffer).decode('utf-8')
                      
            json_meteor = json.dumps({'id': id,'date': id[6:8]+"/"+id[4:6]+"/"+id[0:4],
                'time': id[8:10]+":"+id[10:12]+":"+id[12:14],'score':f"{predict[2][0]}",'obs_id':guAIta_obs_id,
                'processed':now.strftime("%d/%m/%Y %H:%M"),
                'img':imgb64_string,'telegram':guAIta_enable_telegram_txt,
                'folder':folder[-8:]})
            requests.post(guAIta_URL, json_meteor)
        
    print("guAIta finished at: "+now.strftime("%d/%m/%Y %H:%M:%S")+"\n")
    print("Processed : "+str(imgs_processed)+" files")
    if guAIta_enable_telegram==True:
        json_log = json.dumps({'log':'End','obs_id':guAIta_obs_id,'total':str(imgs_processed)})
        requests.post(guAIta_URL, json_log)
    return True    


def guAItaRunner():
    printHeader()
    
    #Check if model is available. If not, download from S3 (latest_version always)
    try:
        learn = load_learner("guAIta_latest_version.pkl")
        print("Model available")
    except:
        print("Downloading Model")
        model = requests.get("https://guaita.s3-eu-west-1.amazonaws.com/models/guAIta_latest_version.pkl")
        open("guAIta_latest_version.pkl",'wb').write(model.content)
        model.close()
        print("Model downloaded")
        learn = load_learner("guAIta_latest_version.pkl")
 
    if guAIta_enable_telegram==True:
        guAIta_enable_telegram_txt = 'True'
    active = True
    imgName_done = ""
    actual_folder=""
    folder_update=0
    while active==True:
        later = time.time()
        if (folder_update==0 or int(later-folder_update)>300):
            print("Checking Folder")
            #Check folders and get the last one
            folder = getLastFolder(guAIta_main_folder)
            if folder=="":
                continue   
            if folder!=actual_folder:
                print("Destination folder: "+folder)
                actual_folder=folder
            folder_update = time.time()

        now = datetime.now() 

        scan = False
        if (int(now.strftime("%H%M"))>=int(guAIta_start_time)):
            scan=True
        if (int(now.strftime("%H%M"))<int(guAIta_end_time)):           
            scan=True
        if scan == False:
            time.sleep (10)
            continue

        #Reading images in folder
        try:
            imgs = load_imgs(folder)
        except:
            print("Error reading images")
            continue

        if (len(imgs)>1):
            try:
                imgName2 = imgs[-1]
                imgName1 = imgs[-2]
                if (imgName2!=imgName_done):
                    print("Image pre-processing")
                    img1=cv.imread(imgName1,cv.IMREAD_GRAYSCALE)
                    img2=cv.imread(imgName2,cv.IMREAD_GRAYSCALE)
                    img_subs = cv.subtract(img2,img1)
                    (thresh, img_subs) = cv.threshold(img_subs, 3.5, 255, cv.THRESH_BINARY)
                    img_subs = cv.resize(img_subs,(224,224),interpolation=cv.INTER_AREA)

                    img = PILImage.create(img_subs)
                    print("Image Inference")
                    predict = learn.predict(img)
                    imgName_done = imgName2

                    if guAIta_enable_extra_log==True:
                        print(f"{imgName2} , {predict[0]} , {predict[2][0]} , {predict[2][1]}")
                    if (predict[0]=="meteor" and predict[2][0]>guAIta_threshold):
                        print("Detection:")
                        print(f"{imgName2} , {predict[0]} , {predict[2][0]} , {predict[2][1]}")   

                        img_resized = cv.resize(img2,(224,224),interpolation=cv.INTER_AREA)
                        id = imgName2[-18:-4]

                        #Join the 2 images
                        numpy_horizontal = cv.hconcat([img_resized,img_subs])
                        retval, buffer = cv.imencode('.jpg',numpy_horizontal)
                        imgb64_string = base64.b64encode(buffer).decode('utf-8')
                      
                        
                        json_meteor = json.dumps({'id': id,'date': id[6:8]+"/"+id[4:6]+"/"+id[0:4],
                            'time': id[8:10]+":"+id[10:12]+":"+id[12:14],'score':f"{predict[2][0]}",'obs_id':guAIta_obs_id,
                            'processed':now.strftime("%d/%m/%Y %H:%M"),
                            'img':imgb64_string,'telegram':guAIta_enable_telegram_txt,
                            'folder':folder[-8:]})
                        requests.post(guAIta_URL, json_meteor)
            except:
                continue
        if guAIta_enable_extra_log==True:
            print("Sleep "+str(guAIta_sleep_between_imgs)+ " sec.")
        time.sleep(guAIta_sleep_between_imgs)
    now = datetime.now() 
    print("guAIta finished at: "+now.strftime("%d/%m/%Y %H:%M:%S")+"\n")
    return True

#guAItaRunner()
guAItaDayAnalisis()

