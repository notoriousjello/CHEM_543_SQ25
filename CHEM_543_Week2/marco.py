import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

tf.config.threading.set_inter_op_parallelism_threads(4)

#Collect the path of all training images in a list
paths = glob.glob('./marcodata/Clear/*.jpeg')+glob.glob('./marcodata/Crystals/*.jpeg')+glob.glob('./marcodata/Precipitate/*.jpeg')+glob.glob('./marcodata/Other/*.jpeg')

#count the number of samples in each folder
num_clear = len(glob.glob('./marcodata/Clear/*.jpeg'))
num_crystal = len(glob.glob('./marcodata/Crystals/*.jpeg'))
num_preci = len(glob.glob('./marcodata/Precipitate/*.jpeg'))
num_other = len(glob.glob('./marcodata/Other/*.jpeg'))

# Convert the labels into integers
#0: clear
#1: crystal
#2: precipitate
#3: Other
labels = [str(0)]*num_clear+[str(1)]*num_crystal+[str(2)]*num_preci+[str(3)]*num_other

# split images into train data and test data
img_train, img_test, label_train, label_test = train_test_split(paths,labels,test_size = 0.3)

# split images into train data and validation data
img_train, img_val, label_train, label_val = train_test_split(img_train,label_train,test_size = 0.2)

# pack the image path and labels in dataframe
df_train = pd.DataFrame(data=list(zip(img_train,label_train)),columns=['path','label'])
df_val = pd.DataFrame(data=list(zip(img_val,label_val)),columns=['path','label'])
df_test = pd.DataFrame(data=list(zip(img_test,label_test)),columns=['path','label'])
# save the df_test into a csv file, you can load this when you evalute the model
df_test.to_csv('./testdata.csv')


# parameters 
nclasses = 4
nchannels = 3 #RGB
imagesize=(300,300)
batchsize=20
epochs=1  #how many epochs you want your CNN to be trained
model_dir = './models/'
model_path = model_dir+'MARCO.{epoch:02d}-{val_accuracy:.4f}.hdf5' # Path at which to save the model file

#create the directory for saving the models
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#Load the densenet as a basemodel
basemodel = tf.keras.applications.DenseNet121(weights='imagenet',include_top=False,input_shape=(imagesize[0],imagesize[1],nchannels))
basemodel.trainable = True

#Build CNN
model = tf.keras.Sequential()
model.add(basemodel)
model.add(tf.keras.layers.GlobalAveragePooling2D())  #Join the global average pooling layer
model.add(tf.keras.layers.Dense(512,activation='relu'))  #Add fully connected layer
model.add(tf.keras.layers.Dropout(rate=0.25))  #Add Dropout layer to prevent overfitting
model.add(tf.keras.layers.Dense(128,activation='relu'))  #Add fully connected layer
model.add(tf.keras.layers.Dropout(rate=0.25))  #Add Dropout layer to prevent overfitting
model.add(tf.keras.layers.Dense(nclasses,activation='softmax'))  #Add output layer
model.summary()   #Print each layer parameter information 

# ImageDataGenerator for training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=[0.9, 1.2],
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='reflect',
                rescale=1./255
                )

# ImageDataGenerator for validation data
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
                )

train_generator = train_datagen.flow_from_dataframe(
                df_train, #this is the training data
                x_col = 'path',  
                y_col ='label',
                target_size=imagesize,  
                batch_size=batchsize,
                shuffle = True,
                class_mode='sparse')

validation_generator = validation_datagen.flow_from_dataframe(
                df_val,  #This is the validation data
                x_col = 'path',  
                y_col ='label',
                target_size=imagesize,  
                batch_size=batchsize,
                shuffle = False,
                class_mode='sparse')

#Compile model
initial_learning_rate = 0.0002
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), #Cross entropy loss function
              metrics=["accuracy"]) #Evaluation function

#Callback function: Save the optimal model
checkpoint = ModelCheckpoint(
                                filepath=model_path, # Path at which to save the model file.
                                monitor='val_accuracy', #Need to monitor the value
                                save_weights_only=True, #If set to True, only the model weight will be saved, otherwise the entire model (including model structure, configuration information, etc.) will be saved
                                save_best_only=True, #When set to True, the current model will only be saved when the monitoring value is improved
                                mode='auto', #When the monitoring value is val_acc, the mode should be max. When the monitoring value is val_loss, the mode should be min. In auto mode, the evaluation criterion is automatically inferred from the name of the monitored value
                                verbose=1 #If True, output a message for each update, default value: False
                            )

#fit the model
history = model.fit(x=train_generator,   #Enter training set
                    steps_per_epoch=train_generator.n // batchsize, #The number of training steps included in an epoch
                    epochs=epochs, #Training model iterations
                    validation_data=validation_generator,  #Enter validation set
                    validation_steps=validation_generator.n // batchsize, #The number of validation steps included in an epoch
                    callbacks=[checkpoint],
                    use_multiprocessing=True)

model.save(model_dir + 'MARCO.h5')


