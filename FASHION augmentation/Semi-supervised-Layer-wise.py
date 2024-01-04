import tensorflow as tf
import numpy as np
from mypackages.learningSort import methods as met
from mypackages.learningSort import massiveDataMethods as massMethods
import tensorflow_datasets as tfds

#directory to save files and hyperparams
mainDirect_ = '/home/est1/Edgar/Semestre 5/model implementation/fashion augmentation/semi-supervised layer-wise/experiment0/log/'
namefile_ = 'semi-supervisedFashion' #file to write data
pretrainfile_ = 'pretrainingLayerWiseFashion' #file to load weights
epochs_ = 1000
trainBATCH_SIZE_ = 32
testBATCH_SIZE_ = 512

#UNSUPERVISED
#get fashion dataset
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
                mainDirect = mainDirect_,
                namefile = pretrainfile_,
                Epochs = epochs_,
                trainBATCH_SIZE = trainBATCH_SIZE_,
                testBATCH_SIZE = testBATCH_SIZE_)

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

#Taking the 16.67% of few labeled examples for training
X_train = X_train[50000:] 
y_train = y_train[50000:] 

#tailor the amount of labeled training example to be used from 0.33% to 16.67%.
setTrueLabels_ = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]) 

#fine tunning on each of the few labeled data and repeated ten times.
met.semiSupervisedAugmented(X_train, 
                   X_test, 
                   y_train, 
                   y_test,
                   mainDirect = mainDirect_,
                   namefile = namefile_,
                   pretrainfile = pretrainfile_,
                   Epochs = epochs_,
                   trainBATCH_SIZE = trainBATCH_SIZE_,
                   testBATCH_SIZE = testBATCH_SIZE_,
                   #numberTrueLabeles = 2,
                   setTrueLabels = setTrueLabels_)
