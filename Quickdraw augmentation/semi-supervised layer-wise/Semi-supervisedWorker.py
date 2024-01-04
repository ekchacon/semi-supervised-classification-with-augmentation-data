import os
import json
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from mypackages.learningSort import massiveDataMethods as massMethods

# Multi-worker strategy
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['132.247.46.16:20100', '132.247.46.91:20101']#workers are n48 and n44. And the same for all machines.
    },
    'task': {'type': 'worker', 'index': 1}#1 because it is the worker.
})

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy_ = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

print('Number of devices: %d' % strategy_.num_replicas_in_sync)  # 输出设备数量

#directory to save results and model weights
mainDirect_ = '/home/edgar/Edgar/Semestre 5/model implementation/quickdraw augmentation/semi-supervised/experiment4/'
namefile_ = 'semi-supervisedQuickdrawAugmented' #file to write data
pretrainfile_ = 'pretrainingLayerWiseQuickdrawAugmented' #file to load weights

#Hyperparams
epochs_ = 50
batch_size_ = 2048
lr_ = 0.001
warmupEpochs_ = 16

#Dataset definition
#get quickdraw dataset to have the element_spec
OrigTrain = tfds.load(
    'quickdraw_bitmap',
    split='train',
    #batch_size=10000,# -1
    as_supervised=True,
    shuffle_files = True
)

#get our quickdraw dataset: labeled and unlabeled
numClasses_ = 10
shape = (28, 28)
pathTrain = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/train/'
pathTest = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/test/'
pathPreTrain = '/home/edgar/tensorflow_datasets/quickdraw_bitmap/pretrainAugmented/'

#loading data for supervised
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#loading data for unsupervised
pretrain = tf.data.experimental.load(pathPreTrain, element_spec=OrigTrain.element_spec)

#Data Pre-processing (unsupervised)
unsupTrain, unsupTest = massMethods.unsupervisedPreProcessing(pretrain,test,shape)

#Pre-training
massMethods.preTraining(unsupTrain,
               unsupTest,
               mainDirect = mainDirect_,
               namefile = pretrainfile_,
               strategy = strategy_,
               Epochs = epochs_,
               batchSize = batch_size_,
               lr = lr_,
               warmupEpoch = warmupEpochs_)#the logic will compute the min warmup epochs based on 2048 bs and 16 warmup epochs.

#FINE TUNNING

#loading data for supervised
pretrainDataAmount = 583310
supTrain = train.skip(pretrainDataAmount)#because I took the first pretrainDataAmount for pre-training.

#Data Pre-processing (supervised)
supTrain, supTest = massMethods.supervisedPreProcessing(supTrain,test,shape)

#tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels_ = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw

#fine tunning on each of the few labeled data and repeated ten times.
massMethods.semiSupervisedAugmented(supTrain,
                            supTest,
                            mainDirect = mainDirect_,
                            namefile = namefile_,
                            pretrainfile = pretrainfile_,
                            strategy = strategy_,
                            Epochs = 50,#50,epochs_,
                            numClasses = numClasses_,
                            batchSize = 32,
                            lr = 0.001,
                            warmupEpoch = 8,
                            verb = 1,
                            setTrueLabels = setTrueLabels_)
