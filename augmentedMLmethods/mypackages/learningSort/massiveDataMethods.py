import tensorflow as tf
import numpy as np
import math
from mypackages.learningSort import massiveDataMethods as massMethods
import os
import csv
from numpy import genfromtxt
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import backend as K
import os, shutil
from tensorflow.keras import layers

#Augmentation
def dataAugmentation(train, AmountTrueLab):
    train = train.map(lambda x,y: (tf.expand_dims(x, -1),y))#Expand dimension (28,28,1)
    train = train.batch(32)
    #one by one functions
    RandomFlip = tf.keras.Sequential([
      layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    ])
    
    RandomRotation = tf.keras.Sequential([ 
      layers.experimental.preprocessing.RandomRotation(0.2),  
    ])
    
    RandomTranslation = tf.keras.Sequential([ 
      layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2)
    ])
    
    RandomZoomOut = tf.keras.Sequential([ 
      layers.experimental.preprocessing.RandomZoom(height_factor=(0.5, 0.2), width_factor=(0.5, 0.2))
    ])
    
    RandomZoomIn = tf.keras.Sequential([ 
      layers.experimental.preprocessing.RandomZoom(height_factor=(-0.5, -0.2), width_factor=(-0.5, -0.2))
    ])
    
    #Apllying functions
    RandomFlipData = train.map(
        lambda image, label: (RandomFlip(image), label)
    )
    
    #RandomFlipData.unbatch
    #RandomFlipData.element_spec
    #RandomFlipData.cardinality()
    #len(list(RandomFlipData.unbatch().as_numpy_iterator()))
    RandomFlipData = RandomFlipData.unbatch()
    RandomFlipData = RandomFlipData.apply(tf.data.experimental.assert_cardinality(AmountTrueLab))
    #RandomFlipData.cardinality()
    
    RandomRotationData = train.map(
        lambda image, label: (RandomRotation(image), label)
    )
    RandomRotationData = RandomRotationData.unbatch()
    RandomRotationData = RandomRotationData.apply(tf.data.experimental.assert_cardinality(AmountTrueLab))
    
    RandomTranslationData = train.map(
        lambda image, label: (RandomTranslation(image), label)
    )
    RandomTranslationData = RandomTranslationData.unbatch()
    RandomTranslationData = RandomTranslationData.apply(tf.data.experimental.assert_cardinality(AmountTrueLab))
    
    
    RandomZoomOutData = train.map(
        lambda image, label: (RandomZoomOut(image), label)
    )
    RandomZoomOutData = RandomZoomOutData.unbatch()
    RandomZoomOutData = RandomZoomOutData.apply(tf.data.experimental.assert_cardinality(AmountTrueLab))
    
    
    RandomZoomInData = train.map(
        lambda image, label: (RandomZoomIn(image), label)
    )
    RandomZoomInData = RandomZoomInData.unbatch()
    RandomZoomInData = RandomZoomInData.apply(tf.data.experimental.assert_cardinality(AmountTrueLab))
    
    train = train.unbatch()
    train = train.apply(tf.data.experimental.assert_cardinality(AmountTrueLab))
    
    mnistAugmented = tf.data.experimental.sample_from_datasets(
              [RandomFlipData,
               RandomRotationData,
               RandomTranslationData,
               RandomZoomOutData,
               RandomZoomInData,
               train])
    
    mnistAugmentedCardi = (RandomFlipData.cardinality() + 
                           RandomRotationData.cardinality() + 
                           RandomTranslationData.cardinality() + 
                           RandomZoomOutData.cardinality() + 
                           RandomZoomInData.cardinality()+
                           train.cardinality())
    
    mnistAugmented = mnistAugmented.apply(tf.data.experimental.assert_cardinality(mnistAugmentedCardi.numpy()))
    #mnistAugmented.cardinality()
    
    train = mnistAugmented
    train = train.map(lambda x,y: (tf.squeeze(x,axis=2),y))#Reduce dimension (28,28)
    return train

def semiSupervisedLEGWautomaticScaled(sample_count, legwSteps):
  warmup_epochs = 16#minWarmupEpochs(sample_count, 32, 16)#batch size always will be 32.
  if sample_count > 5000:
    possibleBatchSizeFactor = sample_count / legwSteps#90 #30 for I see 50k / 2048 batch size = 24 steps perform very well. 120 for self-training otherwise out of memory (OOM). 
    batchSizefactor = math.floor(np.log2(possibleBatchSizeFactor))
    ScaledBatchSize = 2**batchSizefactor
    factor = ScaledBatchSize/32

    if ScaledBatchSize > 8192:#to stop increasing the batch size in function to the training examples.
      ScaledBatchSize = 8192
    
    batchSize = ScaledBatchSize
    #warmup_epochs = warmup_epochs*factor
    lr = 0.001#round(0.001*(math.sqrt(factor)),4)
  else:
    batchSize = 32
    lr = 0.001
  return batchSize, warmup_epochs, lr

def supervisedLEGWautomaticScaled(sample_count, legwSteps):
  warmup_epochs = minWarmupEpochs(sample_count, 32, 16)#batch size always will be 32.
  if sample_count > 5000:
    possibleBatchSizeFactor = sample_count / legwSteps#350 #30 for I see 50k / 2048 batch size = 24 steps perform very well. And 120 for quickdraw
    batchSizefactor = math.floor(np.log2(possibleBatchSizeFactor))
    ScaledBatchSize = 2**batchSizefactor
    factor = ScaledBatchSize/32

    if ScaledBatchSize > 8192:#to stop increasing the batch size in function to the training examples.
      ScaledBatchSize = 8192
    
    batchSize = ScaledBatchSize
    warmup_epochs = warmup_epochs*factor
    lr = round(0.001*(math.sqrt(factor)),4)
  else:
    batchSize = 32
    lr = 0.001
  return batchSize, warmup_epochs, lr

def minWarmupEpochs(sample_count=41665, batch_size=2048, warmup_epoch=4):
  if warmup_epoch > 1:
        warmup_epochAux = warmup_epoch-1
  warmup_batches = math.floor((warmup_epochAux) * sample_count / batch_size)
  
  while (warmup_batches != 1 ):
    warmup_epoch = (warmup_epoch/2.0)
    if warmup_epoch > 1:
        warmup_epochAux = warmup_epoch-1
    else:
        warmup_epochAux = warmup_epoch
    warmup_batches = math.floor((warmup_epochAux) * sample_count / batch_size)
  
    #print(warmup_epoch)
  return warmup_epoch

def LEGWFewExamples(sample_count):
  kindFewExamples = np.array([256, 512, 1024, 2048, 4096, 8192, 16384]) #Possible few examples amount and after 16,384 the bs is 2048
  for element in kindFewExamples:
    batch_size = 2048
    factor = 64
    if element > sample_count:
      batch_size = int(element/8)
      factor = int(batch_size/32)
      break
  #print('batch_size:', batch_size)
  #print('factor:', factor)
  return factor

class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

def supervisedPreProcessing(train,test,shape):#RGB (32,32,3) to (96,32)
  timeSteps = shape[0]
  features = shape[1]
  train = train.map(lambda x,y: [((float(x) / 255.) - .5) * 2,y]) #x and x are normalized
  test = test.map(lambda x,y: [((float(x) / 255.) - .5) * 2,y])

  train = train.map(lambda x,y: [tf.reshape(tf.transpose(x, perm=[2, 0, 1]), [timeSteps, features]), y]) 
  test = test.map(lambda x,y: [tf.reshape(tf.transpose(x, perm=[2, 0, 1]), [timeSteps, features]), y]) 
  return train, test

def unsupervisedPreProcessing(train,test,shape):#RGB (32,32,3) to (96,32)
  timeSteps = shape[0]
  features = shape[1]
  train = train.map(lambda image, label: (image, image)) 
  test = test.map(lambda image, label: (image, image))

  train = train.map(lambda x,y: [((float(x) / 255.) - .5) * 2,((float(y) / 255.) - .5) * 2]) #x and x are normalized
  test = test.map(lambda x,y: [((float(x) / 255.) - .5) * 2,((float(y) / 255.) - .5) * 2])

  train = train.map(lambda x,y: [tf.reshape(tf.transpose(x, perm=[2, 0, 1]), [timeSteps, features]), tf.reshape(tf.transpose(y, perm=[2, 0, 1]), [timeSteps, features])]) 
  test = test.map(lambda x,y: [tf.reshape(tf.transpose(x, perm=[2, 0, 1]), [timeSteps, features]), tf.reshape(tf.transpose(y, perm=[2, 0, 1]), [timeSteps, features])]) 
  return train, test

def create_supervised_dataset(dataset, shape, batch_size):
      def gen():
        for X_train, y_train in tfds.as_numpy(dataset):
    
          #print(X_train.shape, y_train.shape)
          #print(X_train[0])
          #def normalize_img(image, label):
          #  """Normalizes images: `uint8` -> `float32`."""
          #  image = ((image / 255.) - .5) * 2
          #  return image, label
    
          #X_train, y_train = normalize_img(X_train, y_train)
          #print(X_train[0,:,:,])
          #X_train = X_train.transpose(0,3,1,2).reshape(X_train.shape[0],-1,X_train.shape[2]) #for (10k,96,32), for example
          #X_train = X_train.transpose(2,0,1).reshape(-1,X_train.shape[1]) #for (96,32)
          yield X_train, y_train
      ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), (shape, ())) # x and y
    
      return ds.batch(batch_size,drop_remainder=True).repeat()#

def create_unsupervised_dataset(dataset, shape, batch_size): 
  def gen():
    for X_train, y_train in tfds.as_numpy(dataset):

      #print(X_train.shape, y_train.shape)
      #print(X_train[0])
      #def normalize_img(image, label):
      #  """Normalizes images: `uint8` -> `float32`."""
      #  image = ((image / 255.) - .5) * 2
      #  return image, label

      #X_train, y_train = normalize_img(X_train, y_train)
      #print(X_train[0,:,:,])
      #X_train = X_train.transpose(0,3,1,2).reshape(X_train.shape[0],-1,X_train.shape[2]) #for (10k,96,32), for example
      #X_train = X_train.transpose(2,0,1).reshape(-1,X_train.shape[1]) #for (96,32)
      yield X_train, y_train
  ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32), (shape, shape)) # x and y

  return ds.batch(batch_size, drop_remainder=True).repeat()#

def create_unsupervised_dataset_topredict(dataset, shape, batch_size): 
  def gen():
    for X_train in tfds.as_numpy(dataset):

      #print(X_train.shape, y_train.shape)
      #print(X_train[0])
      #def normalize_img(image, label):
      #  """Normalizes images: `uint8` -> `float32`."""
      #  image = ((image / 255.) - .5) * 2
      #  return image, label

      #X_train, y_train = normalize_img(X_train, y_train)
      #print(X_train[0,:,:,])
      #X_train = X_train.transpose(0,3,1,2).reshape(X_train.shape[0],-1,X_train.shape[2]) #for (10k,96,32), for example
      #X_train = X_train.transpose(2,0,1).reshape(-1,X_train.shape[1]) #for (96,32)
      yield X_train
  ds = tf.data.Dataset.from_generator(gen, (tf.float32), (shape)) # x and y

  return ds.batch(batch_size, drop_remainder=True).repeat()#

def supervisedAugmented(train, 
                   test, 
                   mainDirect,
                   namefile,
                   #pretrainfile,
                   strategy,
                   Epochs = 1000,
                   numClasses = 10,
                   batchSize = 32,
                   lr = 0.001,
                   warmupEpoch = 0.25,
                   verb = 0,
                   setTrueLabels = np.array([],dtype=int)):

  #Sequence Length 
  timeSteps = train.element_spec[0].shape[0]
  features = train.element_spec[0].shape[1]
  inputShape = (timeSteps,features)
  numExamTest = test.cardinality().numpy()
  
  for AmountTrueLab in setTrueLabels:# Iteration on few labeled datasets
  
    print(AmountTrueLab)
    tf.keras.backend.clear_session()

    BUFFER_SIZE = 10000
    UNITS = 512

    datatrain = train.take(AmountTrueLab)# taking few labeled data
    datatrain = massMethods.dataAugmentation(datatrain, AmountTrueLab)# augmentating the small dataset
    datatest = test

    sample_count = datatrain.cardinality().numpy()

    batch_size, warmup_epoch, lrm = massMethods.supervisedLEGWautomaticScaled(sample_count)# applying LEGW method
    # Compute the number of warmup batches.
    if warmup_epoch > 1:
      warmup_epoch = warmup_epoch-1

    # Compute the number of warmup batches.
    warmup_batches = math.floor(warmup_epoch * sample_count / batch_size)
     
    # Create the Learning rate scheduler.
    warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=lrm)

    #required for Multi-worker strategy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset0 = massMethods.create_supervised_dataset(datatrain, inputShape, batch_size)
    test_dataset0 = massMethods.create_supervised_dataset(datatest, inputShape, batch_size)

    train_dataset = train_dataset0.with_options(options)
    test_dataset = test_dataset0.with_options(options)
    
    tenRepet = np.empty((1,0), float)
    #loop for ten repetitions 
    i = 0
    while i < 10:
      with strategy.scope():
        def get_lr_metric(optimizer):
              def lr(y_true, y_pred):
                  return optimizer.learning_rate
              return lr
        
        #RNN-LSTM setup
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS),
            tf.keras.layers.Dense(features),
            tf.keras.layers.Dense(numClasses,activation='softmax')])
        
        opt = tf.keras.optimizers.Adam(lrm)
        lr_metric = get_lr_metric(opt)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=['accuracy', lr_metric])

      cbks = [warm_up_lr,
                tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lrm)*(np.exp(-epoch/25))), #(1e-3)/(epoch+1) #0.001
                #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                tf.keras.callbacks.CSVLogger(mainDirect+'log_'+namefile+'.csv', append=True, separator=',')
                ]#(write_graph=True)] 
      #model.summary()
      
      #steps computation for training and test
      stepsPerEpoch = math.floor(sample_count/batch_size)
      stepsTest = math.floor(numExamTest/batch_size)

      try:
        model.fit(train_dataset,epochs=Epochs, 
                            verbose=verb,
                            steps_per_epoch=stepsPerEpoch, 
                            validation_data=test_dataset,
                            validation_steps=stepsTest, 
                            callbacks=cbks
                )
        #i = i + 1
      except tf.errors.OutOfRangeError:
        print("End of dataset")# indicates the iteration is over
        #pass # or print("End of dataset") indicates the iteration is over
        continue
      
      #getting the results
      results = model.evaluate(x=test_dataset,  verbose=0, steps=stepsTest)
      
      if i == 0:
        tenRepet = np.concatenate((tenRepet,AmountTrueLab), axis=None)
        tenRepet = np.concatenate((tenRepet,sample_count), axis=None)
          
      tenRepet = np.concatenate((tenRepet,round(results[1]*100,2)), axis=None)
      i = i + 1
    
    #saving the results
    import csv
    csv.register_dialect("hashes", delimiter=",")
    f = open(mainDirect+namefile+'AccuTenTimes.csv','a')
        
    with f:
        #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
        writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
        writer.writerow(tenRepet)

def semiSupervisedAugmented(train, 
                   test, 
                   mainDirect,
                   namefile,
                   pretrainfile,
                   strategy,
                   Epochs = 1000,
                   numClasses = 10,
                   batchSize = 32,
                   lr = 0.001,
                   warmupEpoch = 0.25,
                   verb = 0,
                   setTrueLabels = np.array([],dtype=int)):

  #Sequence Length
  timeSteps = train.element_spec[0].shape[0]
  features = train.element_spec[0].shape[1]
  inputShape = (timeSteps,features)
  numExamTest = test.cardinality().numpy()
  
  
  for AmountTrueLab in setTrueLabels:# Iteration on few labeled datasets
  
    print(AmountTrueLab)
    tf.keras.backend.clear_session()

    BUFFER_SIZE = 10000
    UNITS = 512

    datatrain = train.take(AmountTrueLab)# taking few labeled data
    datatrain = massMethods.dataAugmentation(datatrain, AmountTrueLab)# augmenting the small dataset
    datatest = test

    sample_count = datatrain.cardinality().numpy()
    
    batch_size, warmup_epoch, lrm = massMethods.semiSupervisedLEGWautomaticScaled(sample_count) # applying LEGW method
    # Compute the number of warmup batches.
    if warmup_epoch > 1:
      warmup_epoch = warmup_epoch-1

    # Compute the number of warmup batches.
    warmup_batches = math.floor(warmup_epoch * sample_count / batch_size)
     
    # Create the Learning rate scheduler.
    warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=lrm)

    #required for Multi-worker strategy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset0 = massMethods.create_supervised_dataset(datatrain, inputShape, batch_size)
    test_dataset0 = massMethods.create_supervised_dataset(datatest, inputShape, batch_size)

    train_dataset = train_dataset0.with_options(options)
    test_dataset = test_dataset0.with_options(options)
    
    tenRepet = np.empty((1,0), float)
    #loop for ten repetitions 
    for i in range(0,10,1):
      with strategy.scope():
        def get_lr_metric(optimizer):
              def lr(y_true, y_pred):
                  return optimizer.learning_rate
              return lr
        
        #RNN-LSTM setup
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS),
            tf.keras.layers.Dense(features),
            tf.keras.layers.Dense(numClasses,activation='softmax')])
        
        opt = tf.keras.optimizers.Adam(lrm)
        lr_metric = get_lr_metric(opt)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=['accuracy', lr_metric])
        
        #RNN-LSTM setup to load initialised weights
        pretrainedModel = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
            tf.keras.layers.Dense(features) #tested with tanh, sigmoid, linear and the better accu (41.10%) was reached without any func actv.
        ])
        
        #pretrainedModel.summary()
        
        pretrainedModel.load_weights(mainDirect+pretrainfile+'.h5')
        
        pretrainedModel.layers[0].get_weights()
        
        #weights assigning
        model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
        model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
        model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
        model.layers[3].set_weights(pretrainedModel.layers[3].get_weights())

      cbks = [warm_up_lr,
                tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lrm)*(np.exp(-epoch/25))), #(1e-3)/(epoch+1) #0.001
                #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                tf.keras.callbacks.CSVLogger(mainDirect+'log_'+namefile+'.csv', append=True, separator=',')
                ]#(write_graph=True)]
      
      #steps computation for training and test
      stepsPerEpoch = math.floor(sample_count/batch_size)
      stepsTest = math.floor(numExamTest/batch_size)

      model.fit(train_dataset,epochs=Epochs, 
                            verbose=verb,
                            steps_per_epoch=stepsPerEpoch,  
                            validation_data=test_dataset,
                            validation_steps=stepsTest,
                            callbacks=cbks
                )
      
      #getting the results
      results = model.evaluate(x=test_dataset,  verbose=0, steps=stepsTest)#batch_size=testBATCH_SIZE,
      
      if i == 0:
        tenRepet = np.concatenate((tenRepet,AmountTrueLab), axis=None)
          
      tenRepet = np.concatenate((tenRepet,round(results[1]*100,2)), axis=None)
    
    #saving the results
    import csv
    csv.register_dialect("hashes", delimiter=",")
    f = open(mainDirect+namefile+'AccuTenTimes.csv','a')
        
    with f:
        #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
        writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
        writer.writerow(tenRepet)

def preTraining(train,
                test,
                mainDirect,
                namefile,
                strategy,
                Epochs = 1000,
                batchSize = 32,
                lr = 0.001,
                warmupEpoch = 0.25):

  #timeSteps and features, which define sequence length
  timeSteps = train.element_spec[0].shape[0]
  features = train.element_spec[0].shape[1]
  inputShape = (timeSteps,features)

  # Number of training samples.
  sample_count = train.cardinality().numpy()
  numExamTest = test.cardinality().numpy()
  
  # Recompute the warmup epochs to find the minimum.
  warmup_epoch = massMethods.minWarmupEpochs(sample_count, batchSize, warmupEpoch)
  
  # Training batch size, set small value here for demonstration purpose.
  batch_size = batchSize
  
  # Compute the number of warmup batches.
  warmup_batches = math.floor(warmup_epoch * sample_count / batch_size)
   
  # Create the Learning rate scheduler.
  
  warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=lr)

  #required for Multi-worker strategy
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

  train_dataset0 = massMethods.create_unsupervised_dataset(train, inputShape, batch_size)
  test_dataset0 = massMethods.create_unsupervised_dataset(test, inputShape, batch_size)

  train_dataset = train_dataset0.with_options(options)
  test_dataset = test_dataset0.with_options(options)
  
  UNITS = 512
  
  #Layer-wise pre-training-------
  for L in range(1,4,1): #4
    print(L)
    with strategy.scope():
      def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.learning_rate
            return lr
  
    
      if L == 1:
        
        ## 1L, 2L o 3L
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
           #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           tf.keras.layers.Dense(features) 
        ])
      
      if L == 2:
        
        ## 1L, 2L o 3L
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           tf.keras.layers.Dense(features) 
        ])
      
        pretrainedModel = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           #tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
           tf.keras.layers.Dense(features) 
        ])
      
      if L == 3:
        
        ## 1L, 2L o 3L
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           tf.keras.layers.Dense(features) 
        ])
      
        pretrainedModel = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
           #tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
           tf.keras.layers.Dense(features) #tested with tanh, sigmoid, linear and the better accu was reached without any func actv.
        ])
      #Compile
      opt = tf.keras.optimizers.Adam(lr)
      lr_metric = get_lr_metric(opt)
      model.compile(optimizer=opt, 
                    loss=tf.keras.losses.MeanSquaredError(), 
                    metrics=['accuracy',lr_metric]) 
      #model.summary()
    
    cbks = [warm_up_lr,
              tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lr)*(np.exp(-epoch/25))), #(1e-3)/(epoch+1) #0.001
              #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
              tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
              tf.keras.callbacks.CSVLogger(mainDirect+'log_'+namefile+'.csv', append=True, separator=',')
              ]#(write_graph=True)]
    
    ##if 1L 
      ##No load
    
    ##if 2L then Load
    if L == 2:
      pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy',lr_metric])
      pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
      #L2ToPretrain[1]( L1Pretrained [0] and [1]Dense)
      model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
      model.layers[2].set_weights(pretrainedModel.layers[1].get_weights())

      #print("***")
      model.layers[0].trainable = False
      #model.summary()
    
    ##if 3L then Load
    if L == 3:
      pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy',lr_metric])
      pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
      #L3ToPretrain[2]( L1L2Pretrained [0], [1] and [2]Dense)
      model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
      model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
      model.layers[3].set_weights(pretrainedModel.layers[2].get_weights())

      #print("***")
      model.layers[0].trainable = False
      model.layers[1].trainable = False
      #model.summary()
    
    #steps computation for training and test
    stepsPerEpoch = math.floor(sample_count/batch_size)
    stepsTest = math.floor(numExamTest/batch_size)

    model.fit(train_dataset,epochs=Epochs, 
                          verbose=1,
                          steps_per_epoch=stepsPerEpoch,  
                          validation_data=test_dataset,
                          validation_steps=stepsTest, 
                          callbacks=cbks
              )
    
    
    model.save_weights(mainDirect+str(namefile)+'.h5')
    #Layer-wise pre-training-------
    

def transferUnlabToAug(unlabeled,
                        datatrain,
                        accuPerExample,
                        labelPerExample,
                        mainDirect):

  sample_count = datatrain.cardinality().numpy()
  #getting classes with the best accu
  numberExamplesPerClass = int(10 ** ((math.log10( sample_count ) // 1) - 1))
  if numberExamplesPerClass == 10000:# when sample_count =< 100000 then will increase by 20000
    numberExamplesPerClass = int((numberExamplesPerClass/10)*2)
  
  exampleToMove = np.empty((1,0), int)
  for cat in range(0,10,1):
    #cat = 0
  
    label = np.array( np.where(labelPerExample == cat) ).flatten() #give a cat indices
  
    #try:
    ind = np.argpartition(accuPerExample[label],-numberExamplesPerClass)[-numberExamplesPerClass:]# N largest numbers
    #except ValueError:  #raised if `y` is empty.
    #    continue
       
    #ind.shape
    #accuPerExample[label].shape
    #try:
    minAccu = np.amin(accuPerExample[label][ind]) # min accu
    #except ValueError:  #raised if `y` is empty.
    #    continue
    accu = np.array( np.where(accuPerExample >= minAccu) ).flatten() #given a min accu of the betters accu, I take all cats (categories) above this min accu.
    accuLabelOK = np.intersect1d(accu,label)[:numberExamplesPerClass] #10 examples or 100, it depends on. And label means the indices of a specific cat.
    #accuLabelOK.shape # indices of a numberExamplesPerClass and they have the best accu.
    #accu.shape
    #label.shape
    exampleToMove = np.concatenate((exampleToMove,accuLabelOK), axis=None)
  #exampleToMove.shape
  
  #How to move examples from unlabeled into labeled dataset
  #1. I need to separate seudo-labeled from unlabaled through indices.
  
  #unlabeled = unlabeled.enumerate()
  #unlabeled.element_spec
  
  #//getting the examplesToremain idxs
  amountUnlabeled = unlabeled.cardinality().numpy()
  idxUnlabeled = np.arange(start=0, stop=amountUnlabeled, step=1)
  #idxUnlabeled.shape
  exampleToremain = np.setdiff1d(idxUnlabeled, exampleToMove)#taking not intersection but a setdifference
  #exampleToremain.shape
  
  #//making all unlabeled and labels as a tuple. This is all (41665) the seudoLabeled.
  labels = tf.data.Dataset.from_tensor_slices(labelPerExample)
  labels = (labels
  .prefetch(tf.data.AUTOTUNE)
    )  
  #labels.element_spec
  #labels.cardinality().numpy()
  
  seudoLabeled = tf.data.Dataset.zip((unlabeled,labels))
  seudoLabeled = (seudoLabeled
  .prefetch(tf.data.AUTOTUNE)
    )  
  #seudoLabeled.element_spec
  
  #//enumerating in order to get not examplesToremain but exampleToMove idxs
  seudoLabeled = seudoLabeled.enumerate()
  seudoLabeledSmall = seudoLabeled.filter(lambda y,x: tf.reduce_all(tf.not_equal(y, exampleToremain)))#.batch(200)
  seudoLabeledSmall = (seudoLabeledSmall
  .prefetch(tf.data.AUTOTUNE)
    )
  #seudoLabeledSmall.element_spec
  seudoLabeledSmall = seudoLabeledSmall.map(lambda x, y: (y),num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
  
  seudoPath = mainDirect+'seudoLabeledTemp/'
  try:
    os.makedirs(seudoPath)
  except FileExistsError:
      # directory already exists
    pass
  
  dir = seudoPath
  for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
  
  tf.data.experimental.save(seudoLabeledSmall, seudoPath)
  seudoLabeledSmall = tf.data.experimental.load(seudoPath, element_spec=datatrain.element_spec)

  #moving to augmentation set
  #datatrain = datatrain.concatenate(seudoLabeledSmall)#this before sample_from_datasets(
  #datatrain = datatrain.interleave(lambda x,y: seudoLabeledSmall,cycle_length=1,num_parallel_calls=tf.data.experimental.AUTOTUNE)
  datatrain = tf.data.experimental.sample_from_datasets(
    [datatrain, seudoLabeledSmall], weights=[0.5, 0.5])
  datatrain = (datatrain
  .prefetch(tf.data.AUTOTUNE)
    )
  #datatrain = seudoLabeled
  
  #giving cardinality
  #cardinality = tf.data.experimental.cardinality(datatrain)
  #print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())
  sample_count = sample_count + exampleToMove.shape[0]
  
  datatrain = datatrain.apply(tf.data.experimental.assert_cardinality(sample_count))
  #datatrain = datatrain.shuffle(1000, reshuffle_each_iteration=False)#1000 avoid Filling up shuffle buffer (this may take a while): 60 of 1000
  #datatrain.cardinality().numpy() #datatrain ready for the next training

  #This code is to avoid error: "10 contain but 11 at least"
  #if datatrain.cardinality().numpy() < 1000000:#55000
  dataTrainPlusSeudoPath = mainDirect+'dataTrainPlusSeudoLabeledTemp/'
  try:
    os.makedirs(dataTrainPlusSeudoPath)
  except FileExistsError:
      # directory already exists
    pass
  
  dir = dataTrainPlusSeudoPath
  for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
  
  tf.data.experimental.save(datatrain, dataTrainPlusSeudoPath)
  datatrain = tf.data.experimental.load(dataTrainPlusSeudoPath, element_spec=datatrain.element_spec)

  dataTrainPlusSeudoPath2 = mainDirect+'dataTrainPlusSeudoLabeledTemp2/'
  try:
    os.makedirs(dataTrainPlusSeudoPath2)
  except FileExistsError:
      # directory already exists
    pass
  
  dir = dataTrainPlusSeudoPath2
  for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
  
  tf.data.experimental.save(datatrain, dataTrainPlusSeudoPath2)
  datatrain = tf.data.experimental.load(dataTrainPlusSeudoPath2, element_spec=datatrain.element_spec)   
  #This code is to avoid error: "10 contain but 11 at least" 
  #/////
  
  #///// delete seudoLabeled from unlabeleds
  #exampleToremain
  unlabeled = unlabeled.enumerate()
  #unlabeled.element_spec
  
  auxUnlabeled = unlabeled.filter(lambda y,x: tf.reduce_all(tf.not_equal(y, exampleToMove)))#.batch(200)
  
  #auxUnlabeled.element_spec
  #auxUnlabeled.cardinality().numpy()
  auxUnlabeled = auxUnlabeled.apply(tf.data.experimental.assert_cardinality(exampleToremain.shape[0]))
  
  unlabeled = auxUnlabeled.map(lambda x, y: (y)) 

  #This code is to avoid slow prediction, for unlabeled is concatenated.
  unlabeledPath = mainDirect+'unlabeledTemp/'
  try:
    os.makedirs(unlabeledPath)
  except FileExistsError:
      # directory already exists
    pass
  
  dir = unlabeledPath
  for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
       
  
  tf.data.experimental.save(unlabeled, unlabeledPath)
  unlabeled = tf.data.experimental.load(unlabeledPath, element_spec=unlabeled.element_spec)
  
  unlabeledPath2 = mainDirect+'unlabeledTemp2/'
  try:
    os.makedirs(unlabeledPath2)
  except FileExistsError:
      # directory already exists
    pass
   
  dir = unlabeledPath2
  for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
  
  tf.data.experimental.save(unlabeled, unlabeledPath2)
  unlabeled = tf.data.experimental.load(unlabeledPath2, element_spec=unlabeled.element_spec)
  ##This code is to avoid slow prediction, for unlabeled is concatenated.

  return unlabeled, datatrain

def selfTrainingAugmented(train,
                test,
                noLabeled,
                sourceUnlabeled,
                setTrueLabels,
                mainDirect,
                strategy,
                numClasses = 10,
                Epochs = 50,
                verb = 1):
  
  #timeSteps and features, which define sequence length
  timeSteps = train.element_spec[0].shape[0]
  features = train.element_spec[0].shape[1]
  inputShape = (timeSteps,features)
  datatest = test
  numExamTest = datatest.cardinality().numpy()
  #AmountTrueLab = datatrain.cardinality().numpy()
  
  for AmountTrueLab in setTrueLabels: # iteration on few labeled datasets
    print(AmountTrueLab)
    #AmountTrueLab = 200
  
    #directory creation to save results
    try:
        os.makedirs(mainDirect+str(AmountTrueLab))
    except FileExistsError:
          # directory already exists
        pass
    
    for tenTimes in range(0,10,1):
      #tenTimes = 0
      tf.keras.backend.clear_session()
      try:
          os.makedirs(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes))
      except FileExistsError:
            # directory already exists
          pass
      
      #building the model architecture
      with strategy.scope():
        def get_lr_metric(optimizer):
              def lr(y_true, y_pred):
                  return optimizer.learning_rate
              return lr
        UNITS = 512

        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS),
            tf.keras.layers.Dense(features),
            #tf.keras.layers.Dense(28, activation='relu'),
            tf.keras.layers.Dense(numClasses,activation='softmax')])#25 for slmnist    
        
        opt = tf.keras.optimizers.Adam(0.001)
        lr_metric = get_lr_metric(opt)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=['accuracy', lr_metric])
      
      unlabeled = noLabeled  
      
      datatrain = train.take(AmountTrueLab)# taking a small dataset
      
      #self-training top edge
      dataToTakeAcum = 0
      while (unlabeled.cardinality().numpy() > 5000): #LOOP FOR ADDING PSEUDO-LABELS INTO TRUE LABELS unlabX_train.shape[0]
        
        #we always hold 50k examples to predict pseudo-labels, the best predictions are used as pseudo-labeled examples.
        if unlabeled.cardinality().numpy() != 50000:
          dataToTake = 50000 - unlabeled.cardinality().numpy()
          dataToTakeAcum = dataToTakeAcum + dataToTake
          datatoSkip = 50000 + dataToTakeAcum
          print("dataToTake", dataToTake, "datatoSkip",datatoSkip, "sourceUnlabeled", sourceUnlabeled.cardinality().numpy())
          if (sourceUnlabeled.cardinality().numpy() - datatoSkip) < dataToTake:
            break

          if (initialAccu-10) > notInitialAccu:#to break when accu is too low than expected.
            break

          unlabeledCardi = unlabeled.cardinality().numpy() + dataToTake
          unlabeled = tf.data.experimental.sample_from_datasets(
          [unlabeled, sourceUnlabeled.skip(datatoSkip).take(dataToTake)], weights=[0.5, 0.5])
          unlabeled = (unlabeled
                       .prefetch(tf.data.AUTOTUNE)
                       )
          unlabeled = unlabeled.apply(tf.data.experimental.assert_cardinality(unlabeledCardi))
          #ds = ds.skip(datatoSkip).take(dataToTake)
          print("dataToTake", dataToTake, "datatoSkip",datatoSkip, "unlabeledCardi", unlabeled.cardinality().numpy())
        #we always hold 50k examples to predict pseudo-labels, the best predictions are used as pseudo-labeled examples.
        
        #data augmentation
        sample_count = datatrain.cardinality().numpy()
        datatrainAug = massMethods.dataAugmentation(datatrain, datatrain.cardinality().numpy())
        sample_countAug = datatrainAug.cardinality().numpy()
        
        batch_size, warmup_epoch, lrm = massMethods.supervisedLEGWautomaticScaled(sample_countAug)#Applying LEGW method
        # Compute the number of warmup batches.
        if warmup_epoch > 1:
          warmup_epoch = warmup_epoch-1
        
        # Compute the number of warmup batches.
        warmup_batches = math.floor(warmup_epoch * sample_countAug / batch_size)
         
        # Create the Learning rate scheduler.
        warm_up_lr = massMethods.WarmUpLearningRateScheduler(warmup_batches, init_lr=lrm)
        
        #required for multi-worker strategy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        #training dataset
        train_dataset0 = massMethods.create_supervised_dataset(datatrainAug, inputShape, batch_size) #add train_dataset0
        test_dataset0 = massMethods.create_supervised_dataset(datatest, inputShape, batch_size)
        
        train_dataset = train_dataset0.with_options(options)
        test_dataset = test_dataset0.with_options(options)
        
        #dataset for prediction
        topredict_dataset0 = massMethods.create_unsupervised_dataset_topredict(unlabeled, inputShape, batch_size) #add train_dataset0
        topredict_dataset = topredict_dataset0.with_options(options)
        
        tf.keras.backend.set_value(model.optimizer.lr, lrm)
        
        cbks = [warm_up_lr,
                  tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lrm)*(np.exp(-epoch/25))), #(1e-3)/(epoch+1) #0.001
                  #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                  tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                  tf.keras.callbacks.CSVLogger(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(sample_count)+'.csv', append=True, separator=',')
                  ]#(write_graph=True)] 
        #model.summary()
        
        print("numTrainBeforeAug: ", sample_count, "numTrainExam: ",sample_countAug)

        #steps computation for training and test
        stepsPerEpoch = math.floor(sample_countAug/batch_size)
        stepsTest = math.floor(numExamTest/batch_size)


        try:
          model.fit(train_dataset,epochs=Epochs, 
                              verbose=verb,
                              steps_per_epoch=stepsPerEpoch,  
                              validation_data=test_dataset,
                              validation_steps=stepsTest, 
                              callbacks=cbks
                  )
        except tf.errors.OutOfRangeError:
          print("End of dataset")# indicates that the iteration is over
          pass # or print("End of dataset") indicates that the iteration is over
        
        model.save_weights(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(sample_count)+'.h5')
        
        #getting the results
        amodel = [sample_count, round(model.history.history['val_accuracy'][-1]*100,2)]

        #To halt self-training when accuracy is lower ten units than the initial accuracy
        if (AmountTrueLab*6) == datatrainAug.cardinality().numpy():
          initialAccu = round(model.history.history['val_accuracy'][-1]*100,2)
        
        notInitialAccu = round(model.history.history['val_accuracy'][-1]*100,2)
        
        ## saving results
        csv.register_dialect("hashes", delimiter=",")
        f = open(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrain.csv','a')
        
        with f:
            #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
            writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
            writer.writerow(amodel)
        
        #label prediction on unlabeled data
        stepsPred = math.floor(unlabeled.cardinality().numpy()/batch_size)
        predImages = model.predict(topredict_dataset, steps=stepsPred)
        #predImages[0]
        
        #In two vectors I have accu and label of all unlabeled (already seudo-labeled).
        accuPerExample = predImages[np.arange(predImages.shape[0]), predImages.argmax(axis=1)]
        labelPerExample = predImages.argmax(axis=1)
        
        
        try:# moving pseudo-labeled examples to actual (or actual and pseudo-labeled) examples.
          unlabeled, datatrain = massMethods.transferUnlabToAug(unlabeled,
                                datatrain,
                                accuPerExample,
                                labelPerExample,
                                mainDirect)
        except ValueError:  #raised if `y` is empty.
          print("Broken by inbalance classes")
          pass
          break 
      
      #getting existing results
      fn = mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrain.csv'
      
      my_data = genfromtxt(fn, delimiter=',')
      try: 
        my_data.shape[1] #sometimes it is one row then I need a matrix as 1x2
      except IndexError:
        pass
        my_data = np.array([my_data])
          
      data = np.array([my_data[np.argmax(my_data[:,1]),1], AmountTrueLab, int(my_data[np.argmax(my_data[:,1]),0]), np.argmax(my_data[:,1])])
      ## saving results of best iteration
      csv.register_dialect("hashes", delimiter=",")
      f = open(mainDirect+str(AmountTrueLab)+'/'+'selftrainAccuPerTrueLabAnd.csv','a')
      
      with f:
          #fieldnames = ['Accu', 'AmountTrueLab', 'AugDataTrueAndSeudoLabels', Iteration]
          writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
          writer.writerow(data)
      ## CSV writing

def selfTrainingLayerWiseAugmented(train,
                test,
                noLabeled,
                sourceUnlabeled,
                setTrueLabels,
                mainDirect,
                pretrainfile,
                strategy,
                numClasses = 10,
                Epochs = 50,
                verb = 1):
  
  #timeSteps and features, which define sequence length
  timeSteps = train.element_spec[0].shape[0]
  features = train.element_spec[0].shape[1]
  inputShape = (timeSteps,features)
  datatest = test
  numExamTest = datatest.cardinality().numpy()
  #AmountTrueLab = datatrain.cardinality().numpy()
  
  for AmountTrueLab in setTrueLabels: # iterate on few labeled examples
    print(AmountTrueLab)
    #AmountTrueLab = 200
  
    # directory creation to save results
    try:
        os.makedirs(mainDirect+str(AmountTrueLab))
    except FileExistsError:
          # directory already exists
        pass
    
    for tenTimes in range(0,10,1):
      #tenTimes = 0
      tf.keras.backend.clear_session()
      try:
          os.makedirs(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes))
      except FileExistsError:
            # directory already exists
          pass
      
      #building the model
      with strategy.scope():
        def get_lr_metric(optimizer):
              def lr(y_true, y_pred):
                  return optimizer.learning_rate
              return lr
        UNITS = 512
        #model creation
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS),
            tf.keras.layers.Dense(features),
            tf.keras.layers.Dense(numClasses,activation='softmax')])
        
        opt = tf.keras.optimizers.Adam(0.001)
        lr_metric = get_lr_metric(opt)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=['accuracy', lr_metric])
        
        #model.layers[0].get_weights()
        
        #auxiliar model creation
        pretrainedModel = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=0.0),
            tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
            tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
            tf.keras.layers.Dense(features) #tested with tanh, sigmoid, linear and the better accu (41.10%) was reached without any func actv.
        ])
        
        #pretrainedModel.summary()
        
        pretrainedModel.load_weights(mainDirect+pretrainfile+'.h5')
        
        
        #assigning weights from aux model to model
        model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
        model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
        model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
        model.layers[3].set_weights(pretrainedModel.layers[3].get_weights())
      
      unlabeled = noLabeled  
      
      datatrain = train.take(AmountTrueLab)# getting a small dataset
      
      #self-training top edge
      dataToTakeAcum = 0
      while (unlabeled.cardinality().numpy() > 5000): #LOOP FOR ADDING PSEUDO-LABELS INTO TRUE LABELS unlabX_train.shape[0]
        
        #we always hold 50k examples to predict pseudo-labels, the best predictions are used as pseudo-labeled examples.
        if unlabeled.cardinality().numpy() != 50000:
          dataToTake = 50000 - unlabeled.cardinality().numpy()
          dataToTakeAcum = dataToTakeAcum + dataToTake
          datatoSkip = 50000 + dataToTakeAcum
          print("dataToTake", dataToTake, "datatoSkip",datatoSkip, "sourceUnlabeled", sourceUnlabeled.cardinality().numpy())
          if (sourceUnlabeled.cardinality().numpy() - datatoSkip) < dataToTake:#to break when unlabeled run out.
            break
          if (initialAccu-0.3) > notInitialAccu:#to break when accu is too low than expected.
            break

          unlabeledCardi = unlabeled.cardinality().numpy() + dataToTake
          unlabeled = tf.data.experimental.sample_from_datasets(
          [unlabeled, sourceUnlabeled.skip(datatoSkip).take(dataToTake)], weights=[0.5, 0.5])
          unlabeled = (unlabeled
                       .prefetch(tf.data.AUTOTUNE)
                       )
          unlabeled = unlabeled.apply(tf.data.experimental.assert_cardinality(unlabeledCardi))
          #ds = ds.skip(datatoSkip).take(dataToTake)
          print("dataToTake", dataToTake, "datatoSkip",datatoSkip, "unlabeledCardi", unlabeled.cardinality().numpy())
        #we always hold 50k examples to predict pseudo-labels, the best predictions are used as pseudo-labeled examples.

        #data augmentation
        sample_count = datatrain.cardinality().numpy()
        datatrainAug = massMethods.dataAugmentation(datatrain, datatrain.cardinality().numpy())
        sample_countAug = datatrainAug.cardinality().numpy()        
        
        batch_size, warmup_epoch, lrm = massMethods.semiSupervisedLEGWautomaticScaled(sample_countAug)# applying LEGW method
        # Compute the number of warmup batches.
        if warmup_epoch > 1:
          warmup_epoch = warmup_epoch-1
        
        # Compute the number of warmup batches.
        warmup_batches = math.floor(warmup_epoch * sample_countAug / batch_size)
         
        # Create the Learning rate scheduler.
        warm_up_lr = massMethods.WarmUpLearningRateScheduler(warmup_batches, init_lr=lrm)
        
        #required for multi-worker strategy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        #training dataset
        train_dataset0 = massMethods.create_supervised_dataset(datatrainAug, inputShape, batch_size) #add train_dataset0
        test_dataset0 = massMethods.create_supervised_dataset(datatest, inputShape, batch_size)
        
        train_dataset = train_dataset0.with_options(options)
        test_dataset = test_dataset0.with_options(options)
        
        #dataset for prediction
        topredict_dataset0 = massMethods.create_unsupervised_dataset_topredict(unlabeled, inputShape, batch_size) #add train_dataset0
        topredict_dataset = topredict_dataset0.with_options(options)
        
        tf.keras.backend.set_value(model.optimizer.lr, lrm)
        
        cbks = [warm_up_lr,
                  tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lrm)*(np.exp(-epoch/25))), #(1e-3)/(epoch+1) #0.001
                  #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                  tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                  tf.keras.callbacks.CSVLogger(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(sample_count)+'.csv', append=True, separator=',')
                  ]#(write_graph=True)] 
        #model.summary()
        
        print("numTrainBeforeAug: ", sample_count, "numTrainExam: ",sample_countAug)

        #steps computation for training and test
        stepsPerEpoch = math.floor(sample_countAug/batch_size)
        stepsTest = math.floor(numExamTest/batch_size)

        if sample_count > 296690:
          break


        try:
          model.fit(train_dataset,epochs=Epochs, 
                              verbose=verb,
                              steps_per_epoch=stepsPerEpoch,  
                              validation_data=test_dataset,
                              validation_steps=stepsTest, 
                              callbacks=cbks
                  )
        except tf.errors.OutOfRangeError:
          print("End of dataset")# indicates that the iteration is over
          pass # or print("End of dataset") indicates that the iteration is over
        
        model.save_weights(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(sample_count)+'.h5')
        
        #getting the results
        amodel = [sample_count, round(model.history.history['val_accuracy'][-1]*100,2)]

        #To halt self-training when accuracy is lower ten units than the initial accuracy
        if (AmountTrueLab*6) == datatrainAug.cardinality().numpy():
          initialAccu = round(model.history.history['val_accuracy'][-1]*100,2)
        
        notInitialAccu = round(model.history.history['val_accuracy'][-1]*100,2)
        
        ## saving results
        csv.register_dialect("hashes", delimiter=",")
        f = open(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrain.csv','a')
        
        with f:
            #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
            writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
            writer.writerow(amodel)
        ## CSV writing
        #results
        
        #predicting on unlabeled data
        stepsPred = math.floor(unlabeled.cardinality().numpy()/batch_size)
        predImages = model.predict(topredict_dataset, steps=stepsPred)
        
        
        #In two vectors I have accu and label of all unlabeled (already seudo-labeled).
        accuPerExample = predImages[np.arange(predImages.shape[0]), predImages.argmax(axis=1)]
        labelPerExample = predImages.argmax(axis=1)
        
        
        try: # moving pseudo-labeled examples to actual (or actual and pseudo-labeled) examples.
          unlabeled, datatrain = massMethods.transferUnlabToAug(unlabeled,
                                datatrain,
                                accuPerExample,
                                labelPerExample,
                                mainDirect)
        except ValueError:  #raised if `y` is empty.
          print("Broken by inbalance classes")
          pass
          break 
      
      # getting existing results
      fn = mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrain.csv'
      
      my_data = genfromtxt(fn, delimiter=',')
      try: 
        my_data.shape[1] #sometimes it is one row then I need a matrix as 1x2
      except IndexError:
        pass
        my_data = np.array([my_data])
          
      data = np.array([my_data[np.argmax(my_data[:,1]),1], AmountTrueLab, int(my_data[np.argmax(my_data[:,1]),0]), np.argmax(my_data[:,1])])
      ## saving results of best iteration
      csv.register_dialect("hashes", delimiter=",")
      f = open(mainDirect+str(AmountTrueLab)+'/'+'selftrainAccuPerTrueLabAnd.csv','a')
      
      with f:
          #fieldnames = ['Accu', 'AmountTrueLab', 'AugDataTrueAndSeudoLabels', Iteration]
          writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
          writer.writerow(data)
      ## CSV writing

