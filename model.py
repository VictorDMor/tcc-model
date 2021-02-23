from datetime import datetime
import argparse
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import kerastuner as kt
import os
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

logging_filename = datetime.now().strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename='logging/{}.log'.format(logging_filename), level=logging.DEBUG, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser(description='Some parameters for the newly created network.')
parser.add_argument('--image_width',  type=int, help='Optional width for the image inputs')
parser.add_argument('--image_height', type=int, help='Optional height for the image inputs')
parser.add_argument('--batch_size', type=int, help='Custom batch size')
parser.add_argument('--color_mode', type=str, help='Color mode to be used during network training', choices=['rgb', 'rgba', 'grayscale'])
args = parser.parse_args()

if args.image_width or args.image_height:
    if not args.image_height:
        print('Image height is missing!')
    elif not args.image_width:
        print('Image width is missing!')
    else:
        image_size = (args.image_width, args.image_height)
else:
    logging.debug('Image width and size were not defined. Setting the default value: (200, 200, 3)')
    image_size = (200, 200)

if args.batch_size:
    batch_size = args.batch_size
else:
    logging.debug('Custom batch size was not set. Setting the default value: 32')
    batch_size = 32

if args.color_mode:
    color_mode = args.color_mode
else:
    logging.debug('Color mode was not specified. Setting the default value: rgb')
    color_mode='rgb'


events = ['penalty', 'freekick', 'none']
folders = [directory for directory in os.listdir('images/') if directory in events]
num_of_classes = len(events)
logging.debug('Image folders: {}'.format(folders))
logging.debug('Number of classes: {}'.format(num_of_classes))

# Entender a função "dataset_from_directory"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'images',
    validation_split=0.2,
    subset='training',
    color_mode=color_mode,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'images',
    validation_split=0.2,
    subset='validation',
    color_mode=color_mode,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)