import os 
import cv2 
import numpy as np 
from PIL import Image


def read_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def detect_and_crop_face(image_array):
    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Assume the first face is the desired one
    x, y, width, height = faces[0]
    cropped_image = image_array[y:y+height, x:x+width]

    return cropped_image

def preprocess_image(image_array, target_size=(224, 224)):
    cropped_image = detect_and_crop_face(image_array)
    if cropped_image is None:
        return None

    resized_image = cv2.resize(cropped_image, target_size)
    normalized_image = resized_image / 255.0  # Normalize to [0, 1]

    return normalized_image

def process_image(imagefile, source_dir, dest_path, target_size):
    source_image = os.path.join(source_dir, imagefile)
    dest_image = os.path.join(dest_path, imagefile)

    image_array = read_image(source_image)
    if image_array is not None:
        preprocessed_image = preprocess_image(image_array, target_size)
        if preprocessed_image is not None:
            # Convert normalized image back to 8-bit for saving
            preprocessed_image_8bit = (preprocessed_image * 255).astype(np.uint8)
            cv2.imwrite(dest_image, cv2.cvtColor(preprocessed_image_8bit, cv2.COLOR_RGB2BGR))

def preprocess_lfw_dataset(dataset_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name in sorted( os.listdir(dataset_dir) ):
        source_dir = os.path.join(dataset_dir, name)
        dest_path = os.path.join(output_dir, name)

        print(f"Processing {name}, contains {len(os.listdir(source_dir))} images ")
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for imagefile in os.listdir(source_dir):
            process_image(imagefile, source_dir, dest_path, target_size)



if __name__ == '__main__' : 
    pass