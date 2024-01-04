import tensorflow as tf
import numpy as np
from mypackages.learningSort import methods as met
from mypackages.learningSort import massiveDataMethods as massMethods
import tensorflow_datasets as tfds

#directories to save results and get pre-trained weights.
mainDirect = '/home/edgar/Edgar/Semestre 5/model implementation/fashion augmentation/self-training layer-wise/experiment0/log/'
pretrainfile = '/home/edgar/Edgar/Semestre 5/model implementation/fashion augmentation/self-training layer-wise/experiment0/log/pretrainingLayerWiseFashion'

#hyperparams
trainBATCH_SIZE = 32
testBATCH_SIZE = 512
unlabBATCH_SIZE = 512
Epochs = 1000

#UNSUPERVISED
#get mnist dataset
train, test = tfds.load( 
    'fashion_mnist',
    split=['train', 'test'],
    #batch_size=10000,# -1
    as_supervised=True,
    shuffle_files = True
)

#get the unlabeled mnist dataset
shape = (28, 28)
pathPreTrain = '/home/est1/tensorflow_datasets/fashion_mnist/pretrainAugmented/'
pretrain = tf.data.experimental.load(pathPreTrain, element_spec=train.element_spec)

#Unlabeled data Pre-processing
unsupTrain, unsupTest = massMethods.unsupervisedPreProcessing(pretrain,test,shape)

# pre-training stage
met.preTrainingAugmented(unsupTrain,
                unsupTest,
                mainDirect = mainDirect,
                namefile = pretrainfile,
                Epochs = Epochs,
                trainBATCH_SIZE = trainBATCH_SIZE,
                testBATCH_SIZE = testBATCH_SIZE)

#FINE TUNNING
#get fashion dataset
(X_train, y_train), (X_test, y_test) = tfds.as_numpy(tfds.load(
    'fashion_mnist',
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

#separating labeled data from unlabeled data
unlabeledXtrain = X_train[:50000]
labeledX = X_train[50000:]
labeledy = y_train[50000:]

#tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]) #MNIST

#Self-training layer-wise method
met.selfTrainingLayerWiseAugmented(mainDirect,
                pretrainfile,
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

