import tensorflow as tf
import numpy as np
import math
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

#Taking the 16.67% of few labeled examples for training
X_train = X_train[50000:] 
y_train = y_train[50000:] 

#Assuring sequence length.
rows = 28
features = 28
X_train = X_train.reshape((len(y_train),rows,features))
X_test = X_test.reshape((len(y_test),rows,features))

for j in range(2,20):# tailor the amount of labeled training example to be used from 0.33% to 16.67%.
  if j <= 10:
    AmountTrueLab = j*100
  else:
    AmountTrueLab = (j-9)*1000
  
  print(AmountTrueLab)
  datatrain = AmountTrueLab
  #Hyperparams
  trainBATCH_SIZE = 32
  testBATCH_SIZE = 512
  BUFFER_SIZE = 10000
  LAYER = 3
  UNITS = 512
  
  #getting the precise number of data to train
  dataXtrain = X_train[:datatrain]
  dataytrain = y_train[:datatrain]

  #from numpy to tf.data of training data
  train = tf.data.Dataset.from_tensor_slices((dataXtrain, dataytrain)) # [:datatrain]
  #train.cardinality().numpy()
  #train.element_spec

  #Data augmentation
  train = met.dataAugmentation(train, datatrain)
  numTrainExam = train.cardinality().numpy()
  train = train.batch(trainBATCH_SIZE).repeat()
  
  #from numpy to tf.data of test data
  test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test = test.batch(testBATCH_SIZE).repeat()
  
  #Each experiment is repeated ten times to ensure the robustness of the results
  tenRepet = np.empty((1,0), float)
  #loop for ten repetitions 
  for i in range(0,10,1):
    def get_lr_metric(optimizer):
          def lr(y_true, y_pred):
              return optimizer.learning_rate
          return lr
    
    #model architecture creation
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
        tf.keras.layers.LSTM(units=UNITS),
        tf.keras.layers.Dense(UNITS, activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')])
    
    opt = tf.keras.optimizers.Adam(1e-3)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['acc', lr_metric])
    
    cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), #using a decay learning rate 
              #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
              tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1), #ES for stopping the training process
              #tf.keras.callbacks.CSVLogger('/content/drive/My Drive/PhD/Semestre 3/model implementation/LSTM semi-supervised/unsupervised logs/mnist/log_L'+str(LAYER)+'_U'+str(UNITS)+'.csv', append=True, separator=',')
              ]
    
    #to have a correct relation of batchSize and step_per_epoch (or steps on the whole training examples)
    stepsPerEpoch = math.floor(numTrainExam/32)

    model.fit(train,epochs=1000, 
                          verbose=1,steps_per_epoch=stepsPerEpoch,  
                          validation_data=test,
                          validation_steps=19, 
                          callbacks=cbks
              )
    
    #model evaluation and getting the results
    results = model.evaluate(x=test, batch_size=testBATCH_SIZE, verbose=0, steps=19)
    if i == 0:
      tenRepet = np.concatenate((tenRepet,dataXtrain.shape[0]), axis=None)
      tenRepet = np.concatenate((tenRepet,numTrainExam), axis=None)
    
    tenRepet = np.concatenate((tenRepet,round(results[1]*100,2)), axis=None)
  
  #Saving the results
  import csv
  csv.register_dialect("hashes", delimiter=",")
  f = open('/home/edgar/Edgar/Semestre 5/model implementation/mnist augmentation/supervised/experiment3/log/supervised.csv','a')
  
  with f:
      #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
      writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
      writer.writerow(tenRepet)
