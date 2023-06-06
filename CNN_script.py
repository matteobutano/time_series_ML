#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

in_data_1 = pd.read_csv('https://drive.google.com/uc?id=19xQWXaqtF685BP9-awsRgNv77udv0-sx')
in_data_2 = pd.read_csv('https://drive.google.com/uc?id=19vESYitHjVR2nEiMgLOiEVlIr4LOTGX4')
in_data_3 = pd.read_csv('https://drive.google.com/uc?id=1FJACauRxbNuilgXm-eg7s-K4HM1sAoeA')
in_data = pd.concat([in_data_1,in_data_2,in_data_3],axis = 0,ignore_index = True)
out_data = pd.read_csv('https://drive.google.com/uc?id=19i5CWvm9-wNXVwF-wiBODty0mPdupyLQ')['reod']


# The next cell specifies the training, test and prediction datasets chosen randomly. 

train_fraction = 0.9
test_fraction = 1 - train_fraction

#Randomly choose elements for each sub-dataset
train_index = np.random.choice(range(in_data.shape[0]), size = int(train_fraction*in_data.shape[0]),replace = False)
test_index = np.random.choice(np.delete(range(in_data.shape[0]),train_index), size = int(test_fraction*in_data.shape[0]),replace = False)


# Here we reshape the dataset to be fed the DNN
X_train = in_data.iloc[train_index].values
Y_train = out_data[train_index].values
X_test = in_data.iloc[test_index].values
Y_test = out_data[test_index].values

# Rescale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Prepare Data to be fed the DNN
num_classes = 3
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

# Training parameters
batch_size = 128
epochs = 100

def create_CNN():
    
    model = Sequential()
    model.add(Input(shape=(53, 1, 1)))
    
    model.add(Conv2D(256, (9, 1),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256, (9, 1),activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

print('Model architecture created successfully!')

# add early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# use RMSprop optimizer
optimizer = keras.optimizers.SGD(learning_rate=0.005)

def compile_model(optimizer=optimizer):
    # create the mode
    model=create_CNN()
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    return model

print('Model compiled successfully and ready to be trained.')

# create the deep neural net
model_CNN=compile_model()

# train CNN and store training info in history
history=model_CNN.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),callbacks=[early_stop])

# evaluate model
score = model_CNN.evaluate(X_test, Y_test, verbose=1)

# print performance
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# look into training history
fig, ax = plt.subplots(1,2, figsize = (10,5))

# summarize history for accuracy
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_ylabel('model accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'test'], loc='best')

# summarize history for loss
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_ylabel('model loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'test'], loc='best')

plt.savefig('CNN Training vs Test.png')

# Import test set and predict categories

chal_in_data_1 = pd.read_csv('https://drive.google.com/uc?id=1Zlh9JkPFc7vmudIyT0-UcloOUkP5tcY8')
chal_in_data_2 = pd.read_csv('https://drive.google.com/uc?id=1qNDlxFtF4ZEjcEl7NWo-zMft8ifTfabP')
chal_in_data_3 = pd.read_csv('https://drive.google.com/uc?id=1VZdfOQNI64syZd1-VkyKr0sNDeztAFEb')
chal_in_data = pd.concat([chal_in_data_1,chal_in_data_2,chal_in_data_3],axis = 0,ignore_index = True).values
IDs = np.arange(1e6,1e6 + chal_in_data.shape[0],dtype = int)

X_chal = scaler.transform(chal_in_data)
X_chal = X_chal.reshape(X_chal.shape[0], X_chal.shape[1], 1, 1)
Y_chal = model_CNN.predict(X_chal)
chal_classes = np.array([[0,1,-1] for x in range(Y_chal.shape[0])])
chal_best = np.argmax(Y_chal,axis = 1)
chal_results= np.array([chal_classes[i,chal_best[i]] for i in range(len(chal_best))])
chal_preds = pd.DataFrame(np.vstack([IDs,chal_results]).transpose())
chal_preds.columns = ['ID','reod']

chal_preds.to_csv('output_test.csv',index=False)

