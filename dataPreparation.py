#from fastai.vision.all import *
import torch
import cv2 as cv
import os

#Utility to create a txt file with all the files containing the "x" in the filename
def getFilesMarked(folder):
    files_list = getFileListFromFolder(folder)
    positives = open(folder+"/../positives.txt","w")
    for filename in files_list:
        if filename[20]=='x':
            positives.write(filename[:20]+filename[21:]+'\n')
    positives.close()   

#Utility function to return a list of images based on txt file
def loadPositiveMeteors(filename):
    file1 = open(filename,"r")    
    content = file1.readlines()
    content = [x.strip() for x in content]
    file1.close()
    return content

#Function to generate a custom subtract function
def custom_subtract(img1,img2,ratio):
    height, width = img1.shape
    img_ret=np.zeros((height,width), np.uint8)
    for i in range(height):
        for j in range(width):
            a = img1.item(i,j) 
            b = img2.item(i,j) 
            if b > a:
                if 1-(a/b)>=ratio:
                    c = 255
                else:
                    c = 0
            else:
                c = 0
            img_ret.itemset((i,j), c)
    return img_ret

#Function to increase the brightness in a image
def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

#Function to add a black frame to image to convert the image to square proportions
def addFrameTopAndBottom(img,bordersize=164):
    row, col = img.shape[:2]
    img = cv.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=0,
        right=0,
        borderType=cv.BORDER_CONSTANT,
        value=[0, 0, 0]
        )
    return img

#Utility function to get all the files from a folder, skiping directories
def getFileListFromFolder(folder):
    files_list = []
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isdir(path):
            # skip directories
            continue
        files_list.append(entry)
    return files_list

#Utility function to test image transformations in all the images in a folder
# - Convert images to grayscale
# - Subtract images
# - Resize image 
# - Count the ratio of black pixels
# - Store image
def substractImagesFolderAndResize(folder,from_time,to_time,threeshold):
    print ("Scanning folder: "+folder)
    files_list = getFileListFromFolder(folder)

    tuple_list = [[files_list[i], files_list[i + 1]]
        for i in range(len(files_list) - 1)]
    
    out_folder=folder+"/subst/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for i in tuple_list:
        if (from_time and to_time and not(int(i[0][6:20])>=from_time and int(i[0][6:20])<=to_time)):
            continue
        img1=cv.imread(folder+"/"+i[0],cv.IMREAD_GRAYSCALE)
        img2=cv.imread(folder+"/"+i[1],cv.IMREAD_GRAYSCALE)
           
        img_subs = cv.subtract(img2,img1)
            
        img_subs = cv.resize(img_subs,(224,224),interpolation=cv.INTER_AREA)

        pixel_not_black = cv.countNonZero(img_subs)     
        img_pixels = img1.shape[0] * img1.shape[1]         
        pixel_threeshold = img_pixels * threeshold / 100
        if pixel_not_black>=pixel_threeshold:
            cv.imwrite(out_folder+i[1],img_subs,[cv.IMWRITE_JPEG_QUALITY, 100])
            print (out_folder+i[1])

#Test function to test image transformation and find ROIs(Regions of Interes) in the resulting image
# - Convert images to grayscale
# - Subtract images
# - Apply Threeshold
# - Delete "lost" pixels
# - Detect contours to find ROIs
# - Count the number of ROIs per Image
# - Store image with the ROIs applied
def v1_substractImagesFolderAndGenerateROIs(folder,from_time="",to_time=""):
    print ("Scanning folder: "+folder)
    files_list = getFileListFromFolder(folder)
    
    tuple_list = [[files_list[i], files_list[i + 1]]
        for i in range(len(files_list) - 1)]
    
    out_folder=folder+"/ROIs/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    total_ROIs=0
    imgs_wROIs=0
    imgs_processed=0
    for i in tuple_list:
        if (from_time and to_time and not(int(i[0][6:20])>=from_time and int(i[0][6:20])<=to_time)):
            continue
        imgs_processed=imgs_processed+1
        img1=cv.imread(folder+"/"+i[0],cv.IMREAD_GRAYSCALE)
        img2=cv.imread(folder+"/"+i[1],cv.IMREAD_GRAYSCALE)
        
        img_subs = cv.subtract(img1,img2)
        (thresh, img_subs) = cv.threshold(img_subs, 3.5, 255, cv.THRESH_BINARY)

        input_image_comp = cv.bitwise_not(img_subs)  
        kernel1 = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]], np.uint8)
        kernel2 = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 0, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]], np.uint8)
        hitormiss1 = cv.morphologyEx(img_subs, cv.MORPH_ERODE, kernel1)
        hitormiss2 = cv.morphologyEx(input_image_comp, cv.MORPH_ERODE, kernel2)
        hitormiss  = cv.bitwise_and(hitormiss1, hitormiss2)
        hitormiss_comp = cv.bitwise_not(hitormiss)  
        img_subs = cv.bitwise_and(img_subs, img_subs, mask=hitormiss_comp)
        
        contours, hierarchy = cv.findContours(img_subs, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        applied_contours=0
        for c in range (len(contours)):
            if hierarchy[0][c][3]==-1:
                x,y,w,h = cv.boundingRect(contours[c])
                if (w<=10 and h<=10):
                     continue
                applied_contours=applied_contours+1
                cv.rectangle(img_subs,(x,y),(x+w,y+h),(255,0,0),1)
        print("Img: "+i[1]+" (Number of Contours found: " + str(len(contours))+", Countours applied: "+str(applied_contours)+")")
        total_ROIs=total_ROIs+applied_contours
        
        if (applied_contours>0):
            imgs_wROIs=imgs_wROIs+1
            cv.imwrite(out_folder+i[1],img_subs,[cv.IMWRITE_JPEG_QUALITY, 100])  

    print("---General Statistics---")        
    print("Total Images processed :"+str(imgs_processed))
    print("Images with ROIs :   "+str(imgs_wROIs))    
    print("Images without ROIs :   "+str(imgs_processed-imgs_wROIs))
    print("Total ROIs generated:   "+str(total_ROIs))
    print("-------------------------")   

#Functions to apply transformations to a pair of images
def version6_transformations(img1,img2):
    img_subs = custom_subtract(img1,img2,0.1)
    img_subs = cv.GaussianBlur(img_subs, (11, 11), 0)
    (thresh, img_subs) = cv.threshold(img_subs, 3.5, 255, cv.THRESH_BINARY)
    img_subs = cv.resize(img2,(400,400),interpolation=cv.INTER_AREA)
    return img_subs
def version1_transformations(img1,img2):
    img_subs = cv.subtract(img2,img1)
    img_subs = cv.resize(img_subs,(224,224))
    return img_subs
def version7_transformations(img1,img2):
    img_subs = cv.subtract(img2,img1)
    (thresh, img_subs) = cv.threshold(img_subs, 3.5, 255, cv.THRESH_BINARY)
    img_subs = cv.resize(img_subs,(400,400),interpolation=cv.INTER_AREA)
    return img_subs
def version8_transformations(img1,img2):
    img_subs = cv.resize(img2,(400,400),interpolation=cv.INTER_AREA)
    value=100
    #Brightness
    img_subs = np.where((255 - img_subs) < value,255,img_subs+value)
    return img_subs
def vDef_transformations(img1,img2):
    img_subs = cv.subtract(img2,img1)  # res=img1-img2
    (thresh, img_subs) = cv.threshold(img_subs, 3.5, 255, cv.THRESH_BINARY)
    img_subs = addFrameTopAndBottom(img_subs)
    img_subs = cv.resize(img_subs,(224,244),interpolation=cv.INTER_AREA)
    return img_subs

#Function to get the meteor images from the CSV of the dataset
def getPositivesFromDataSet(dataset_file='C:/Development/meteor_detector/dataset/index.csv'):
    positives = loadPositiveMeteors(dataset_file)
    positives_test = [p for p in positives if (p[-9:]!="no-meteor" and p[1:9]=='test_set')]
    positives_train = [p for p in positives if p[-9:]!="no-meteor" and p[1:13]=='training_set']
    positives_test = [p[17:41] for p in positives_test]
    positives_train = [p[21:45] for p in positives_train]
    return positives_train+positives_test

#Function to create a new version of the dataset applying a specific transformation    
def createDataset(folder,positiveFolder,negativeFolder,version="v6",from_time="",to_time=""):
    print ("Scanning folder: "+folder)
    positives = getPositivesFromDataSet()
    files_list = getFileListFromFolder(folder)
    tuple_list = [[files_list[i], files_list[i + 1]]
        for i in range(len(files_list) - 1)]
   
    out_folder=folder+"/../"+version+"/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(folder+"/../"+version+"/"+positiveFolder+"/"):
        os.makedirs(folder+"/../"+version+"/"+positiveFolder+"/")
    if not os.path.exists(folder+"/../"+version+"/"+negativeFolder+"/"):
        os.makedirs(folder+"/../"+version+"/"+negativeFolder+"/")        

    for i in tuple_list:
        if (from_time and to_time and not(int(i[0][6:20])>=from_time and int(i[0][6:20])<=to_time)):
            continue
        img1=cv.imread(folder+"/"+i[0],cv.IMREAD_GRAYSCALE)
        img2=cv.imread(folder+"/"+i[1],cv.IMREAD_GRAYSCALE)
        
        if (version=="v1"):
           img_subs = version1_transformations(img1,img2)
        if (version=="v6"):
            img_subs = version6_transformations(img1,img2)
        if (version=="v7"):
            img_subs = version7_transformations(img1,img2)
        if (version=="v8"):
            img_subs = version8_transformations(img1,img2)            
        # ...
        if (i[1] in positives):
            cv.imwrite(out_folder+positiveFolder+"/"+i[1],img_subs,[cv.IMWRITE_JPEG_QUALITY, 100]) 
        else:
            cv.imwrite(out_folder+negativeFolder+"/"+i[1],img_subs,[cv.IMWRITE_JPEG_QUALITY, 100]) 

# Helper function to create transformations over 2 images
def createExamples(source1,source2,name):
    img1=cv.imread(source1,cv.IMREAD_GRAYSCALE)
    img2=cv.imread(source2,cv.IMREAD_GRAYSCALE)
    imgsub = version7_transformations(img1,img2)
    cv.imwrite("C:/Development/meteor_detector/dataset/Positius/tractades/"+name,imgsub,[cv.IMWRITE_JPEG_QUALITY, 100]) 

# Helper function to convert image to grayscale, and resize to 400x400
def createGray400(sourcename,name):
    img1=cv.imread(sourcename,cv.IMREAD_GRAYSCALE)
    img_subs = cv.resize(img1,(400,400),interpolation=cv.INTER_AREA)
    cv.imwrite("C:/Development/meteor_detector/dataset/Positius/tractades/"+name,img_subs,[cv.IMWRITE_JPEG_QUALITY, 100]) 


#Examples of execution
#Scan all images in folder 20210219 and generate a v7 folder with the image transformation applied
# Generate a meteor and no-meteor folders and put the images on them based on the CSV of the dataset 
createDataset('C:/Development/meteor_detector/dataset/20210219',"meteor","no-meteor",version="v7")

#To create a full version of the dataset:
#createDataset('C:/Development/meteor_detector/dataset/20210219',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210306',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210307',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210310',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210311',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210312',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210313',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210314',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210315',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210316',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210317',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210318',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210319',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210320',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210321',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210322',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210323',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210324',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210325',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210326',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210327',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210328',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210329',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210330',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210331',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210401',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210402',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210403',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210404',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210405',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210406',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210407',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210408',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210409',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210410',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210411',"meteor","no-meteor")
#createDataset('C:/Development/meteor_detector/dataset/20210412',"meteor","no-meteor")








