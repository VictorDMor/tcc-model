from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import argparse
import kerastuner as kt
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

logging_filename = datetime.now().strftime('%Y%m%d-%H%M%S')
model_filename = datetime.now().strftime('%Y%m%d-%H%M')
logging.basicConfig(filename='logging/{}.log'.format(logging_filename), level=logging.DEBUG, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser(description='Some parameters for the newly created network.')
parser.add_argument('--image_width', type=int, help='Optional width for the image inputs')
parser.add_argument('--image_height', type=int, help='Optional height for the image inputs')
parser.add_argument('--batch_size', type=int, help='Custom batch size')
parser.add_argument('--color_mode', type=str, help='Color mode to be used during network training', choices=['rgb', 'rgba', 'grayscale'])
parser.add_argument('--data_augmentation', help='Choose whether there will be a data augmentation on the images', action='store_false')
parser.add_argument('--plot_model', help='If set to True, plots the model architecture and saves it on script\'s directory', action='store_false')
parser.add_argument('--epochs', type=int, help='Set number of epochs for training')
parser.add_argument('--optimizer', type=str, help='Set optimizer for model compilation')
parser.add_argument('--learning_rate', type=int, help='Set a learning rate for the optimizer')
parser.add_argument('--saved_model_path', type=str, help='Set a path to save the trained model')
args = parser.parse_args()

def build_model(shape, num_of_classes, activation_function, network_size=128, convolution_blocks=3):
    inputs = tf.keras.Input(shape=shape)
    x = inputs
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(x)

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_function)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_function)(x)

    previous_block_activation = x
    conv_sizes = [network_size * i for i in range(1, convolution_blocks)]

    for convs in conv_sizes:
        x = tf.keras.layers.Activation(activation_function)(x)
        x = tf.keras.layers.SeparableConv2D(convs, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation(activation_function)(x)
        x = tf.keras.layers.SeparableConv2D(convs, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = tf.keras.layers.Conv2D(convs, 1, strides=2, padding='same')(previous_block_activation)
        x = tf.keras.layers.add([x, residual])
        previous_block_activation = x

    x = tf.keras.layers.SeparableConv2D(network_size * len(conv_sizes), 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_function)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    class_activation = 'sigmoid' if num_of_classes == 2 else 'softmax'
    units = num_of_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=class_activation)(x)

    return tf.keras.Model(inputs, outputs)


if args.image_width or args.image_height:
    if not args.image_height:
        print('Image height is missing!')
    elif not args.image_width:
        print('Image width is missing!')
    else:
        image_size = (args.image_width, args.image_height)
else:
    logging.debug('Image width and size were not defined. Setting the default value: (200, 200)')
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

if args.data_augmentation is True:
    data_augmentation_flag = True
else:
    logging.debug('Data augmentation option was not specified. Setting the default value: False')
    data_augmentation_flag = False

if args.epochs:
    epochs = args.epochs
else:
    logging.debug('Number of epochs was not set. Setting the default value: 20')
    epochs = 20

if args.learning_rate:
    learning_rate = args.learning_rate
else:
    logging.debug('Learning rate was not set. Setting the default value: 0.001 (1e-3)')
    learning_rate = 1e-3

if args.optimizer:
    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
else:
    logging.debug('Optimizer was not specified. Setting the default optimizer: Adam')
    optimizer = tf.keras.optimizers.Adam(learning_rate)

if args.saved_model_path:
    saved_model_path = args.saved_model_path
else:
    logging.debug('Path for trained model not set. Setting the default path: trained_models')
    saved_model_path = 'trained_models'

events = ['penalty', 'freekick', 'none']
folders = [directory for directory in os.listdir('images/') if directory in events]
num_of_classes = len(events)
logging.debug('Image folders: {}'.format(folders))
logging.debug('Number of classes: {}'.format(num_of_classes))

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'images',
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    color_mode=color_mode,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
).prefetch(buffer_size=32)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'images',
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    color_mode=color_mode,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
).prefetch(buffer_size=32)

logging.debug('Size of train dataset: {} images'.format(len(train_dataset)))
logging.debug('Size of validation dataset: {} images'.format(len(validation_dataset)))

if data_augmentation_flag:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')
        ]
    )

model = build_model(shape=image_size + (3,), num_of_classes=num_of_classes, activation_function='relu')

if args.plot_model:
    tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)
model.save(saved_model_path + '/' + model_filename)