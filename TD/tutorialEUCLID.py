#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Simmetria0_Diego_Tuccillo
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Convolution2D, Conv2D
from keras.layers.noise import GaussianNoise
from keras.models import model_from_json
from keras.models import load_model
import pickle


#Where you take the simulated data for train/validation
pathinData = '/data/Diego/sharing/'
#where do you wanna save the results
pathinModel = '/data/Diego/sharing/Models_1c/'


X = np.load(pathinData+'Stamps_Simulated_Galaxies.npy')
Y = np.load(pathinData+'Parameters_Simulated_Galaxies.npy') 

Y= Y[:,1]

print (X.shape)
print (Y.shape)
#Y = Y0[np.where((Y0[:,0] < 23.0) &  (Y0[:,1] < 31.6 ) &  (Y0[:,2] < 6.2 ))] 
#X = X0[np.where((Y0[:,0] < 23.0) &  (Y0[:,1] < 31.6 ) &  (Y0[:,2] < 6.2 ))] 
#X = X[:30000,:,:,:]
#Y = Y[:30000,:]
#np.save('/data/Diego/sharing/Stamps_Simulated_Galaxies_tutorial.npy',X)
#np.save('/data/Diego/sharing/Parameters_Simulated_Galaxies_tutorial.npy',Y)


#=============================================== 
# Right shape 
#===============================================
print ('X.shape= ', X.shape)
print ('Y.shape= ', Y.shape)
X = np.expand_dims(X[:,0,:,:], axis=3)
Y = Y.reshape(-1,1)
print ('new X.shape= ', X.shape)
print ('Y.shape= ', Y.shape)

#=============================================== 
# Scale
#===============================================
scaler = preprocessing.StandardScaler().fit(Y)
Y=scaler.transform(Y)



# Spliting in Training and Test datasets
X_train = X[0:len(X)//5*4,:,:,:]   
X_val = X[len(X)//5*4:,:,:,:]
Y_train = Y[0:len(Y)//5*4,0]
Y_val = Y[len(Y)//5*4:,0]
print ('X_train.shape= ', X_train.shape)
print ('X_val.shape= ', X_val.shape)
print ('Y_train.shape= ', Y_train.shape)          
print ('Y_val.shape= ', Y_val.shape)



def Build_Model():
    ## PARAMETERS OF THE MODEL
    print ('Using Built Model')
    dropoutpar = 0.15
    img_rows=128
    img_cols=128
    img_channels=1
    depth=16
    nb_dense = 64  

    # KERAS SEQUENTIAL-MODEL
    model = Sequential()
    model.add(Conv2D(depth, (4, 4),activation='relu', input_shape=(img_rows, img_cols, img_channels), padding="same"))
    model.add(Conv2D(depth, (4, 4),activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropoutpar))

    model.add(GaussianNoise(0.01,input_shape=( img_rows, img_cols,img_channels)))
    model.add(Conv2D(4*depth, (3, 3),activation='relu', padding="same"))
    model.add(Conv2D(4*depth, (3, 3),activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(4*depth, (2, 2),activation='relu', padding="same"))
    model.add(Conv2D(4*depth, (2, 2),activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2*nb_dense))
    model.add(Dense(nb_dense))
    model.add(Dense(1))

    return model



def Fit_Model(X_train, X_val,  Y_train, Y_val,  model):
    # let's TRAIN the model using SGD + momentum  ===> you can try different loss, optimizer and and settings
    lr=0.01  
    decay=0   
    momentum=0.9 
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='mean_absolute_error', optimizer=sgd)

    #Do u wanna use data augmentation?
    data_augmentation = True
    #hyperparameters of the training. Try different
    batch_size = 64
    nb_epoch = 3
    if data_augmentation == False:
        print('Not using data augmentation.')
        history = model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_val, Y_val))

    if data_augmentation == True:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(X_train, Y_train,
                                batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(X_val, Y_val))
    return model, history 



#BUILT model
model = Build_Model()

#train model
model, history = Fit_Model(X_train, X_val, Y_train, Y_val, model)

# do you wanna save the model?
saveModel = False
if saveModel == True:
    #save scaler
    scalerfile = pathinModel+'scaler.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))
    model.save(pathinModel+'model.h5')
    print("Saved model to disk")




fig = plt.figure(figsize=(12,8))
plt.plot(history.epoch, history.history['loss'], label='loss')
plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
plt.title('Training performance')
plt.savefig(pathinModel+'TrainingPerformance.pdf', bbox_inches='tight')
plt.legend()
plt.show()
plt.close()


print("Best validation loss: %.3f" % (np.min(history.history['val_loss'])))
print("at: %d" % np.argmin(history.history['val_loss']))


# a better metric
val = scaler.inverse_transform(Y_val)
pred = scaler.inverse_transform(model.predict(X_val))
pred = pred[:,0]
mse=np.mean(np.square(pred-val)) 
R2 = 1. - mse/np.square(np.std(val))
print ('R2=', R2)

#and now we plot prediction versus validation value
fig = plt.figure(figsize=(12,12))
plt.title('Val sample. $R^2$ = %s' % (R2), size=11)
plt.xlabel('Par', fontsize=13)
plt.ylabel("Par Predicted", fontsize=13)
plt.scatter(val, pred)
plt.savefig(pathinModel+'PredictionVsModel.pdf', bbox_inches='tight')
plt.show()
plt.close()







#=================================================
# 1) load real dataset

X_real = np.load(pathinData+'RealStamps_1311.npy')
Y_real = np.load(pathinData+'ParametersRealStamps_1311.npy') 

Y_real = Y_real[:,6]

#np.save('/data/Diego/sharing/RealStamps_tutorial.npy',X_real)
#np.save('/data/Diego/sharing/ParametersRealStamps_tutorial.npy',Y_real)


print ('shape of X_real =', X_real.shape)
print ('shape of Y_real =', Y_real.shape)


#visualize the data
i = 0
plt.imshow(X_real[i,0,:,:,],clim=(0,.75))



#Test the model as it is on Real Data
# RIGHT shape
X_real = np.expand_dims(X_real[:,0,:,:], axis=3)

pred_real = scaler.inverse_transform(model.predict(X_real))
val_real = Y_real

mse=np.mean(np.square(pred_real-val_real)) 
R2_real = 1. - mse/np.square(np.std(val_real))
print ('R2 real=', R2_real)

#and now we plot prediction versus validation value
fig = plt.figure(figsize=(12,12))
plt.title('Real data sample. $R^2$ = %s' % (R2), size=11)
plt.xlabel('Par', fontsize=13)
plt.ylabel("Par Predicted", fontsize=13)
plt.scatter(val_real, pred_real)
plt.savefig(pathinModel+'PredictionVsModel.pdf', bbox_inches='tight')
plt.show()
plt.close()



# TRANSFER LEARNING.
#load model and keep training (transfer learning) using part of the real data

LoadModel = False
if LoadModel == True:
        #load scaler
        scalerfile = pathinModel+'scaler.sav'
        scaler = pickle.load(open(scalerfile, 'rb')) 
        #load model
        model.load(pathinModel+'model.h5')
        print("Loaded model from disk")

#Now you just repeat the training with a subsample of the real dataset (you really don't need all 5000 stamps!)
#Note: you may wanna change some hyperparameters of the model


subsampleSize = 500
X_real[:subsampleSize, :,:,:]
Y_real[:subsampleSize,]

# Spliting in Training and Test dataset
X_real_train = X_real[0:len(X_real)//5*4,:,:,:]   
X_real_val = X_real[len(X_real)//5*4:,:,:,:]
Y_real_train = Y_real[0:len(Y_real)//5*4,]
Y_real_val = Y_real[len(Y_real)//5*4:,]



#train model
model, history = Fit_Model(X_real_train, X_real_val, Y_real_train, Y_real_val, model)



#Now test if improves
val_real = scaler.inverse_transform(Y_real_val)
pred_real = scaler.inverse_transform(model.predict(X_real_val))
mse=np.mean(np.square(pred_real-val_real)) 
R2 = 1. - mse/np.square(np.std(val_real))
print ('R2 after transfer learning=', R2)

#and now we plot prediction versus validation value
fig = plt.figure(figsize=(12,12))
plt.title('Val sample. $R^2$ = %s' % (R2), size=11)
plt.xlabel('Par', fontsize=13)
plt.ylabel("Par Predicted", fontsize=13)
plt.scatter(val_real, pred_real)
plt.savefig(pathinModel+'PredictionVsModel_afterTL.pdf', bbox_inches='tight')
plt.show()
plt.close()
