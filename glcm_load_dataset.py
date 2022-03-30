import numpy as np 
import cv2 
import os
import re

# -------------------- Utility function ------------------------
def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("","", "()"))
    str_ = str_.split("_")
    return re.sub(r'\d+$', '',''.join(str_[:2]))

def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder 
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text

def print_progress(val, val_len, folder, filename, bar_size=10):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] folder : %s/ ----> file : %s" % (progr, folder, filename), end="\r")
        

# -------------------- Load Dataset ------------------------
 
dataset_dir = "dataset_cropping" 

imgs = [] #list image matrix 
labels = []
descs = []

for folder in os.listdir(dataset_dir):
        sub_folder_files = os.listdir(os.path.join(dataset_dir,folder))
        len_sub_folder = len(sub_folder_files) - 1
        for i,name in enumerate(sub_folder_files):
            # print(os.path.join(dataset_dir,folder,name))

            # Save image in set directory 
            # Read RGB image 
            img = cv2.imread(os.path.join(dataset_dir,folder, name))  

            # Convert RGB image to grayscale  use cv2.cvtColor
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Get Height and Weight from gray shape
            h, w = gray.shape
            # Set ymin, ymax, xmin, xmax from each gray shape
            ymin, ymax, xmin, xmax = h//150, h*149//150, w//150, w*149//150           

            # crop region of interest (ROI) to get important part from citra leaf
            crop = gray[ymin:ymax, xmin:xmax]

            # resize 20% use cv2.resize()
            resize = cv2.resize(crop, (0,0), fx=0.2, fy=0.2)

            # keseluruhan dataset citra daun akan tersimpan pada 
            # list imgs dan nama daun disimpan pada list labels
            imgs.append(resize)
            labels.append(normalize_label(os.path.splitext(name)[0]))
            descs.append(normalize_desc(dataset_dir,folder))

            # print(labels)

            print_progress(i, len_sub_folder, dataset_dir, name)         
            
            # # Output img with window name as 'image' 
            cv2.imshow(name, resize)  
            
            # # Maintain output window utill 
            # # user presses a key 
            cv2.waitKey(0)           
            
            # # Destroying present windows on screen 
            cv2.destroyAllWindows()  

          