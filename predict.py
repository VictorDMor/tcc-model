import tensorflow as tf
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Some parameters for the image prediction script.')
parser.add_argument('--images_folder', type=str, help='[OPTIONAL] Folder where the images are stored. Default is "prediction_images"')
parser.add_argument('--model_path', type=str, help='[OPTIONAL] Path for the model you want to use. Default is the latest model saved')
args = parser.parse_args()

if args.images_folder:
    folder = args.images_folder
else:
    folder = 'prediction_images'

image_paths = []

for i in os.listdir(folder):
    if '.png' in i:
        image_paths.append(folder + '/' + i)

# Trying to get the latest available model
if args.model_path:
    model = tf.keras.models.load_model(args.model_path)
else:
    model = tf.keras.models.load_model('trained_models/' + sorted(os.listdir('trained_models'), reverse=True)[0])

for path in image_paths:
    image = tf.keras.preprocessing.image.load_img(
        path, target_size=(200, 200)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    predictions = model.predict(image_array)
    score = predictions[0]

