import numpy  as np
from itertools import combinations, product
import random
import glob
import os
import tensorflow as tf 
import warnings 
warnings.filterwarnings('ignore')



def read_tf_path(tf_img_path, target_size = (105,105), grayscale=False) :

    n_c =  1 if grayscale else 3
    # Read the image
    image = tf.io.read_file(tf_img_path)
    image = tf.image.decode_jpeg(image, channels=n_c)

    image = tf.image.resize(image,size =target_size)

    return tf.cast(image, tf.uint16)


def preprocess_image(file_path, label):

       image_path_1 , image_path_2 = file_path

       img_1 = read_tf_path(image_path_1)
       img_2 = read_tf_path(image_path_2)

       return (img_1, img_2), label

def arrange_dataset(dataset_dir) : 
    
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
        elif len(images) == 1: 
            positive.append((images[0], images[0]))
    
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


if __name__ == '__main__' : 
    pass