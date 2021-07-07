import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr, spearmanr
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 20,
})

class CNN_classifier():
    #constructor
    def __init__(self):
        self.dataIsPrepared = False
        self.modelIsPrepared = False
        self.modelIsTrained = False
        self.alreadyReduced = False
       
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def __del__(self):
        pass
        
    #load, normalize, and preprocess data
    def prepareData(self,test_size=0.2):
        assert not self.dataIsPrepared, "Data has been prepared before."
        #load raw data
        diagsN7m5 = pd.read_csv('../../../data/tobias/EC3_data/diagsN7m5.csv',names=np.arange(10000)).T
        fidelsN7m5 = pd.read_csv('../../../data/tobias/EC3_data/fidelsN7m5.csv',names=np.arange(10000)).T

        self.N_instance, self.N_feature = diagsN7m5.shape
        self.N_train, self.N_test = int((1-test_size)*self.N_instance), int(test_size*self.N_instance)
        
        #min max normalization
        diagsN7m5 = (diagsN7m5 - diagsN7m5.to_numpy().min())
        diagsN7m5 /= (diagsN7m5.to_numpy().max() - diagsN7m5.to_numpy().min())
        
        #determine categories
        uni_fidels = np.sort(fidelsN7m5[0].unique())
        self.N_categories = uni_fidels.shape[0]
        
        #create one hot representation
        one_hot_fidels = pd.DataFrame()
        for uni_fidel in uni_fidels:
            one_hot_fidels[uni_fidel] = 1*(fidelsN7m5 == uni_fidel).to_numpy().flatten()
        
        #prepare data for CNN with two channels, one energy, one state number (normalized as well)
        diags_CNN = diagsN7m5.to_numpy().reshape((-1,self.N_feature,1))
        state_CNN = np.linspace(0,1,self.N_feature)\
        .repeat(self.N_instance)\
        .reshape(self.N_feature,self.N_instance).T\
        .reshape((-1,self.N_feature,1))
        diags_CNN_twoChannel = np.append(diags_CNN,state_CNN,axis=2)
        
        #split train/test data
        X_train, X_test, Y_train, Y_test = train_test_split(diags_CNN_twoChannel, \
                                                     one_hot_fidels,\
                                                     test_size=test_size)

        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test 
        self.dataIsPrepared = True
        
    #reduce the number of features to reduce*(number of features) via sampling
    def reduceFeatures(self,reduce):
        assert self.dataIsPrepared, "Data not prepared yet. Do CNN_classifier.prepareData()."
        assert 0 <= reduce and 1 > reduce, "Value for 'reduce' is wrong."
        assert not self.alreadyReduced, "Data has been reduced already."
        
        #separately for train and test data through random sampling
        choice = np.arange(self.N_feature)>=self.N_feature*reduce
        self.N_feature = int(self.N_feature*(1-reduce))
        
        X_train_reduced = np.zeros((self.N_train,self.N_feature,2))
        for i in range(self.N_train):
            np.random.shuffle(choice)
            X_train_reduced[i,:] = self.X_train[i,choice]

        X_test_reduced = np.zeros((self.N_test,self.N_feature,2))
        for i in range(self.N_test):
            np.random.shuffle(choice)
            X_test_reduced[i,:] = self.X_test[i,choice]

        self.X_train, self.X_test = X_train_reduced, X_test_reduced
        
    #first construct and compile the neural network    
    def CNN(self):
        assert self.dataIsPrepared, "Data not prepared yet. Do CNN_classifier.prepareData()."
        
        #build up CNN with hard-coded hyperparameters
        tf.keras.backend.clear_session()
        model = keras.Sequential(name='CNN_classifier')
        model.add(layers.Conv1D(8, (3,), activation='relu', padding='same',\
                                input_shape=(self.N_feature, 2)))
        model.add(layers.Conv1D(16, (3,), activation='relu', padding='same'))
        model.add(layers.Conv1D(32, (3,), activation='relu', padding='same'))
        model.add(layers.AveragePooling1D((16,)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(.0))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.N_categories, activation=tf.keras.activations.softmax))
        
        model.compile(loss='CategoricalCrossentropy',\
                      optimizer=tf.keras.optimizers.Adam(.002),metrics=['accuracy'])

        self.model = model
        self.modelIsPrepared = True

    #then train the neural network
    def training(self,epochs,early=True):
        assert self.modelIsPrepared, "Model not prepared yet. Do CNN_classifier.CNN()."
        
        #introduce early stopping
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
        
        callbacks =[]
        if early:
            callbacks += [stop_early]
    
        self.hist = self.model.fit(self.X_train, self.Y_train, \
                     epochs=epochs,verbose=0,\
                     validation_split=.2,batch_size=128,\
                     callbacks=callbacks)
        self.modelIsTrained = True

    def plotLoss(self):
        assert self.modelIsTrained, "Model not trained yet. Do CNN_classifier.training()."
        plt.semilogy(self.hist.history['loss'], label='loss')
        plt.semilogy(self.hist.history['val_loss'], label = 'val loss')
        plt.xlabel('Epoch')
        plt.legend()
        
    def plotAccuracy(self):
        assert self.modelIsTrained, "Model not trained yet. Do CNN_classifier.training()."
        plt.plot(self.hist.history['accuracy'], label='accuracy')
        plt.plot(self.hist.history['val_accuracy'], label = 'val accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
    def plotTestPrediction(self):
        assert self.modelIsTrained, "Model not trained yet. Do CNN_classifier.training()."
        plt.hist2d(self.Y_test.to_numpy().argmax(axis=1),\
                   self.model.predict(self.X_test).argmax(axis=1),\
                   bins=self.N_categories);
        plt.xlabel('true')
        plt.ylabel('predict')
        plt.colorbar()
        
        