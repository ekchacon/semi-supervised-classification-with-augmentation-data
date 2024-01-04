import os
import json
import tensorflow as tf
import numpy as np
from mypackages.learningSort import methods as met
import tensorflow_datasets as tfds
from mypackages.learningSort import massiveDataMethods as massMethods

#Multi-worker strategy
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['132.247.46.16:20200', '132.247.46.91:20201']#workers are n48 and n44. And the same for all machines.
    },
    'task': {'type': 'worker', 'index': 0}#0 because it is the chief and different for each worker.
})

communication_options = tf.distribute.experimental.CommunicationOptions(
   implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy_ = tf.distribute.MultiWorkerMirroredStrategy(
   communication_options=communication_options)

print('Number of devices: %d' % strategy_.num_replicas_in_sync)

#directory to save files
mainDirect_ = '/home/est1/Edgar/Semestre 5/model implementation/quickdraw augmentation/supervised/experiment3/first3FewExamWith90stepsIIMAS/'
namefile_ = 'supervisedQuickdraw' #file to write data

#amount to pre-train with, quickdraw dataset.
toAndFro_ = 583310

#Dataset definition
#get quickdraw dataset to have the element_spec
OrigTrain = tfds.load(
    'quickdraw_bitmap',
    split='train',
    #batch_size=10000,# -1
    as_supervised=True,
    shuffle_files = True
)

numClasses_ = 10#number of clases
shape = (28, 28)#image shape

#directory of our quickdraw dataset
pathTrain = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/train/'
pathTest = '/home/est1/tensorflow_datasets/quickdraw_bitmap/edgarShuffled/test/'

#loading data for supervised
train = tf.data.experimental.load(pathTrain, element_spec=OrigTrain.element_spec)
test = tf.data.experimental.load(pathTest, element_spec=OrigTrain.element_spec)

#Dividing dataset for training.
supTrain = train.skip(toAndFro_)
supTest = test

#Labeled data Pre-processing
supTrain, supTest = massMethods.supervisedPreProcessing(supTrain,supTest,shape)

# tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels_ = np.array([2310, 3500, 4690, 5810, 7000, 8190, 9310, 10500, 11690, 23310, 35000, 46690, 58310, 70000, 81690, 93310, 105000, 116690]) #Quickdraw

#The supervised method
massMethods.supervisedAugmented(supTrain,
                            supTest,
                            mainDirect = mainDirect_,
                            namefile = namefile_,
                            #pretrainfile = pretrainfile_,
                            strategy = strategy_,
                            Epochs = 50,#50,epochs_,
                            numClasses = numClasses_,
                            batchSize = 32,#not really used
                            lr = 0.001,#not really used
                            warmupEpoch = 8,#not really used
                            verb = 1,
                            setTrueLabels = setTrueLabels_)
