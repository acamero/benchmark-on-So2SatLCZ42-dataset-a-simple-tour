# @Date:   2020-02-04T17:07:00+01:00
# @Last modified time: 2020-04-24T10:31:52+02:00

import resnet
import model
import lr

from dataLoader import generator

from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45#0.41
session = tf.compat.v1.Session(config=config)

###################################################
'path to save models from check points:'
file0='./results/'

'path to data, needs to be set accordingly'
train_file='/data/local_home/came_an/git/lcz42/training.h5'
validation_file='/data/local_home/came_an/git/lcz42/validation.h5'

numClasses=17
batchSize=32
###################################################

'number of all samples in training and validation sets'
trainNumber=352366
validationNumber=24119
lr_sched = lr.step_decay_schedule(initial_lr=0.002, decay_factor=0.5, step_size=5)

###################################################
# patch_shape=(32,32,10)
# model = resnet.resnet_v2(input_shape=patch_shape, depth=11, num_classes=numClasses)
model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)
model_name = 'sen2LCZ'

model.compile(optimizer = Nadam(), loss = 'categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 40)
modelbest = file0 + model_name + "_" + str(batchSize) +"_weights.best.hdf5"
checkpoint = ModelCheckpoint(modelbest, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.fit_generator(generator(train_file, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(validation_file, num=validationNumber, batchSize=batchSize),
                validation_steps = validationNumber//batchSize,
                epochs=1,
                max_queue_size=100,
                callbacks=[early_stopping, checkpoint, lr_sched])
