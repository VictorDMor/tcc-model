from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
# import kerastuner as kt
import os
# import tensorflow as tf
import numpy as np

events = ['penalty', 'freekick', 'none']
train_folders = [directory for directory in os.listdir('images/train/') if directory in events]
classes = len(events)

# Entender a função "flow_from_directory"
