import os
import json
import csv
from numpy import genfromtxt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from mypackages.learningSort import massiveDataMethods as massMethods
import tensorflow_datasets as tfds

# Multi-worker strategy
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['132.247.46.16:20200', '132.247.46.91:20201']#workers are n48 and n44. And the same for all machines.
    },
    'task': {'type': 'worker', 'index': 0}#0 because it is the chief and different for each worker.
})

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)##7s/epoch
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

print('Number of devices: %d' % strategy.num_replicas_in_sync)

#directory to save results and model weights
mainDirect = '/home/est1/Edgar/Semestre 5/model implementation/quickdraw augmentation/self-trainingLayer-wise/experiment1/'
pretrainfile = 'pretrainingLayerWiseQuickdrawAugmented' #file to load weights

#Hyperparams
epochs = 50
batch_size = 2048
lr = 0.001
warmupEpochs = 16

#Dataset definition
#get quickdraw dataset to have the element_spec
OrigTrain = tfds.load(
    'quickdraw_bitmap',
    split='train',
    #batch_size=10000,# -1
    as_supervised=True,
    shuffle_files = True
)

#get labeled and unlabeled quickdraw dataset
numClasses_ = 10
shape = (28, 28)
pathTrain = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/train/'
pathTest = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/test/'
pathPreTrain = '/home/est1/tensorflow_datasets/quickdraw_bitmap/pretrainAugmented/'

#loading data for supervised
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#loading data for unsupervised
pretrain = tf.data.experimental.load(pathPreTrain, element_spec=OrigTrain.element_spec)

#Data Pre-processing (unsupervised)
unsupTrain, unsupTest = massMethods.unsupervisedPreProcessing(pretrain,test,shape)#RGB (32,32,3) to (96,32)

#Pre-training
massMethods.preTraining(unsupTrain,
               unsupTest,
               mainDirect = mainDirect,
               namefile = pretrainfile,
               strategy = strategy,
               Epochs = epochs,
               batchSize = batch_size,
               lr = lr,
               warmupEpoch = warmupEpochs)#the logic will compute the min warmup epochs based on 2048 bs and 16 warmup epochs.

#Data Pre-processing (supervised)
train, test = massMethods.supervisedPreProcessing(train,test,shape)

#Separating labaled data
supTrain = train.take(116690)

#Separating unlabaled data
sourceUnlabeled = train.skip(116690)
sourceUnlabeled = sourceUnlabeled.map(lambda image, label: (image),num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
unlabeled = sourceUnlabeled.take(50000)

#tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw

#fine tunning on each of the few labeled data and repeated ten times.
massMethods.selfTrainingLayerWiseAugmented(supTrain,
             test,
             unlabeled,
             sourceUnlabeled,
             setTrueLabels,
             mainDirect,
             pretrainfile,
             strategy,
             numClasses = 10,
             Epochs = 50,
             verb = 1)