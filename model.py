from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as incres_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import SGD
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
parser.add_argument('--learning_rate', type=float, help='Set a learning rate for the optimizer')
parser.add_argument('--saved_model_path', type=str, help='Set a path to save the trained model')
parser.add_argument('--network', type=str, help='Network to be used for training', choices=['inception', 'inception_resnet', 'regular', 'resnet50'])
args = parser.parse_args()

def plot_metrics(metrics, epochs, option='categorical_accuracy'):
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, metrics[0]['history'].history[option], 'red', label='{} {}'.format(metrics[0]['network'], option))
    plt.plot(epoch_range, metrics[1]['history'].history[option], 'green', label='{} {}'.format(metrics[1]['network'], option))
    plt.plot(epoch_range, metrics[2]['history'].history[option], 'blue', label='{} {}'.format(metrics[2]['network'], option))
    plt.plot(epoch_range, metrics[3]['history'].history[option], 'yellow', label='{} {}'.format(metrics[3]['network'], option))
    plt.xlabel('epochs')
    plt.ylabel(option)
    plt.legend()
    plt.savefig('metrics/comparison_{}_{}_metrics_{}.png'.format(option, epochs, datetime.now().strftime('%Y%m%d-%H%M%S')))
    plt.clf()

def build_model(shape, num_of_classes, alpha=0.5):
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='linear', padding='same', input_shape=shape),
                                        tf.keras.layers.LeakyReLU(alpha=alpha),
                                        tf.keras.layers.MaxPooling2D((2, 2),padding='same'),
                                        tf.keras.layers.Conv2D(64, (3, 3), activation='linear',padding='same'),
                                        tf.keras.layers.LeakyReLU(alpha=alpha),
                                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation='linear'),
                                        tf.keras.layers.LeakyReLU(alpha=alpha),
                                        tf.keras.layers.Dense(num_of_classes, activation='softmax')])
    return model

def transfer_learning(shape, num_of_classes, train, epochs, valid, network, learning_rate=1e-4):
    if network == 'resnet50':
        base_model = ResNet50(include_top=False, weights='imagenet')
        trainable_limit = 143
    elif network == 'inception_resnet':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    else:
        base_model = InceptionV3(weights='imagenet', include_top=False)
        trainable_limit = 249
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_of_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)        
    
    for layer in base_model.layers:
        layer.trainable = False
      
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(train, epochs=5, validation_data=valid)
    
    if network != 'inception_resnet':
        for layer in model.layers[:trainable_limit]:
            layer.trainable = False
        for layer in model.layers[trainable_limit:]:
            layer.trainable = True
    else:
        for layer in model.layers:
            layer.trainable = True
    
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=categorical_accuracy)
    history = model.fit(train, epochs=epochs, validation_data=valid)
    return model, history

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
    if color_mode == 'grayscale':
        color_channels = 1
    else:
        color_channels = 3
else:
    logging.debug('Color mode was not specified. Setting the default value: rgb')
    color_mode='rgb'
    color_channels = 3

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
    logging.debug('Learning rate: {args.learning_rate}')
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

if args.network:
    network = args.network
else:
    logging.debug('Network not set. Setting the default network: Regular Convolutional')
    network = 'regular'

events = ['penalty', 'corner', 'freekick', 'none']
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

if data_augmentation_flag:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')
        ]
    )

metrics = []
for network in ['regular', 'resnet50', 'inception', 'inception_resnet']:
    if network == 'regular':
        model = build_model(shape=image_size + (color_channels,), num_of_classes=num_of_classes)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=categorical_accuracy
        )

        history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)
    else:
        model, history = transfer_learning(image_size + (color_channels,), num_of_classes, train_dataset, epochs, validation_dataset, network)
    metrics.append({
        'network': network,
        'model': model,
        'history': history
    })

for option in ['categorical_accuracy', 'val_categorical_accuracy', 'loss', 'val_loss']:
    plot_metrics(metrics, epochs, option)

model.save(saved_model_path + '/{}_'.format(network) + model_filename)