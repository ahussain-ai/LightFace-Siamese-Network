import cv2
import numpy  as np
from itertools import combinations, product
import random
import glob
import os
from PIL import Image
import matplotlib.pyplot  as plt

import warnings 
warnings.filterwarnings('ignore')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def read_image(image_path, margin=10):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) != 1:
        return None
    (x, y, w, h) = faces[0]

    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, img.shape[1])
    y_end = min(y + h + margin, img.shape[0])
    
    cropped_face = img[y_start:y_end, x_start:x_end]
    cropped_face = cv2.resize(cropped_face, (105, 105))  # Resize to 105x105
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    return cropped_face

def read_pair(pair) : 
    path1, path2 = pair 
    return (read_image(path1), read_image(path2))

def arrange(dataset_dir) : 
    
    peoples = {} 
    
    for dire in os.listdir(dataset_dir) :
        dir_path = os.path.join(dataset_dir,dire)
        image_files = glob.glob(f"{dir_path}/*.*")
        if not len(image_files) : 
            print(f"{dir_path} no image found")
        
        peoples[dire] = image_files 
    return peoples

def generate_pairs(d, num_pairs = 5000) : 
    
    unique_positives = []
    positive = []
    anchor = [] 
    negative = [] 
    
    #for positive pairs
    for person,images in d.items() : 
        
        if len(images) > 1 :
            positive.extend( list(combinations(images, 2)))
            unique_positives.append(person)
    
    #negative_pairs
    all_keys = list (d.keys() )
    for _ in range(num_pairs) : 
        person1, person2 = random.sample(all_keys, 2) 
        neg = random.sample( list(product(d[person1], d[person2])), 1) 
        negative.extend(neg) 
    
    
    ## if the length of positive is more than num_pairs, select only num_pairs 
    
    if len(positive) > num_pairs : 
        positive = random.sample(positive, num_pairs)
        
    positive_label = np.ones(num_pairs)
    negative_label = np.zeros(num_pairs) 
    
    return (positive, positive_label), (negative, negative_label) , unique_positives