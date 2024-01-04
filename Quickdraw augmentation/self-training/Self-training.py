import tensorflow as tf
import numpy as np
import math
from mypackages.learningSort import massiveDataMethods as massMethods
import tensorflow_datasets as tfds

#allows to use the GPU in the server
strategy = tf.distribute.MirroredStrategy()

##directories to save results
mainDirect = '/home/edgar/Edgar/Semestre 5/model implementation/quickdraw augmentation/self-training/experiment0/'

print('Number of devices: %d' % strategy.num_replicas_in_sync)

#get quickdraw dataset to have the element_spec
OrigTrain = tfds.load(
    'quickdraw_bitmap',
    split='train',
    #batch_size=10000,# -1
    as_supervised=True,
    shuffle_files = True
)

#get our quickdraw dataset
numClasses_ = 10
shape = (28, 28)
pathTrain = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/train/'
pathTest = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/test/'

#loading data for supervised
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#Data Pre-processing (supervised)
train, test = massMethods.supervisedPreProcessing(train,test,shape)

#Separating labaled data
supTrain = train.take(116690)

#Separating unlabaled data
sourceUnlabeled = train.skip(116690)
sourceUnlabeled = sourceUnlabeled.map(lambda image, label: (image),num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
unlabeled = sourceUnlabeled.take(50000)

# tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw

#self-training method
massMethods.selfTrainingAugmented(supTrain,
             test,
             unlabeled,
             sourceUnlabeled,
             setTrueLabels,
             mainDirect,
             strategy,
             numClasses = 10,
             Epochs = 50,
             verb = 1)