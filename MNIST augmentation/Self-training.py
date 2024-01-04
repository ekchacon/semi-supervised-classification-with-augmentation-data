import tensorflow as tf
import numpy as np
import csv
from mypackages.learningSort import methods as met
import tensorflow_datasets as tfds

#get mnist dataset
(X_train, y_train), (X_test, y_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))

#Data pre-processing 
  #Normalization
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  image = ((image / 255.) - .5) * 2
  return image, label

X_train, y_train = normalize_img(X_train, y_train)
X_test, y_test = normalize_img(X_test, y_test)

  #Reshaping
X_train = X_train.transpose(0,3,1,2).reshape(X_train.shape[0],-1,X_train.shape[2])
X_test = X_test.transpose(0,3,1,2).reshape(X_test.shape[0],-1,X_test.shape[2])

#Directory to save results
mainDirect = '/home/est1/Edgar/Semestre 5/model implementation/mnist augmentation/self-training/experiment0/log/'

# tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]) #MNIST

#Separating labeled data from unlabeled data
unlabeledXtrain = X_train[:50000]
labeledX = X_train[50000:]
labeledy = y_train[50000:]

#Hyperparams
trainBATCH_SIZE = 32
testBATCH_SIZE = 512
unlabBATCH_SIZE = 512
Epochs = 1000

#self-training method
met.selfTrainingAugmented(mainDirect,
                setTrueLabels,
                unlabeledXtrain,
                labeledX,
                labeledy,
                X_test,
                y_test, 
                trainBATCH_SIZE,
                testBATCH_SIZE, 
                unlabBATCH_SIZE,
                Epochs)
