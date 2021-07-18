__author__ = 'Xin, Aboozar'

import numpy as np
from numpy.core.umath_tests import inner1d
from copy import deepcopy

##kerase & CNN:
#from keras import models as Models
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder #LabelBinarizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.engine.saving import model_from_json
from keras.layers import Reshape, Dense, Convolution1D, Dropout, Input, Activation, Flatten,MaxPool1D,add, AveragePooling1D, Bidirectional,GRU,LSTM,Multiply,Activation, MaxPooling1D,TimeDistributed,AvgPool1D
from keras.layers.merge import Concatenate,concatenate
from keras.layers.wrappers import Bidirectional
from pandas.io import excel
from six.moves import cPickle as pickle
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,RMSprop, Adamax, Nadam
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.regularizers import l2, l1
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.backend import sigmoid
from keras import metrics
from keras.constraints import max_norm
import logging
import os
import matplotlib.pyplot as plt

from iCircA3 import mk_dir, bn_activation_dropout, ConvolutionBlock, MultiScale, createModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import time
import argparse
import math
import logging
import os
import sys
import numpy as np
import time
import math
import tensorflow as tf
import collections
from itertools import cycle
from scipy import interp
from Deal_Kmer import *
import keras.layers.core as core
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Lambda
from keras.layers import dot
import sys
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import to_categorical
import easy_excel
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import pickle
import pdb
import logging, multiprocessing
from collections import namedtuple
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention,ScaledDotProductAttention
from scipy import interp
#import matplotlib.pyplot as plt
import xlwt
from DProcess import convertRawToXY
from attention import Attention,myFlatten
from sklearn.ensemble import AdaBoostRegressor



class AdaBoostClassifier(object):


    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state', 'epochs']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 50
        learning_rate = 1
        algorithm = 'SAMME.R'
        random_state = None
        #### CNN (5)
        epochs = 6

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
            ### CNN:
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')
            

        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        self.estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_errors_ = np.ones(self.n_estimators_)
        
        self.epochs= epochs


    def _samme_proba(self, model, n_classes, test_sequence,  test_profile):

        #proba = estimator.predict_proba(test_sequence,  test_profile, testonehot,  test_ANF_NCP)
        proba = model.predict({'sequence_input': test_sequence, 'profile_input': test_profile})
        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])


    def fit(self,model,train_X1, train_X2, train_y, batch_size):
        #self.base_estimator_=model;
        train_y1 = train_y[:, 0]
        ## CNN:
        self.batch_size = batch_size
        
#        self.epochs = epochs
        self.n_samples = train_X1.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort

        self.classes_ = np.array(sorted(list(set(train_y1))))
        
        ############for CNN (2):
#        yl = np.argmax(y)
#        self.classes_ = np.array(sorted(list(set(yl))))

        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_estimators_):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error = self.boost(model,train_X1, train_X2,  train_y, sample_weight)

            # early stop
            if estimator_error == None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break

        return self


    def boost(self,model,train_X1, train_X2,  train_y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(model,train_X1, train_X2, train_y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(model,train_X1, train_X2, train_y, sample_weight)

            
    def real_boost(self, model,train_X1, train_X2,  train_y, sample_weight):
        #            estimator = deepcopy(self.base_estimator_)
        ############################################### my code:
        train_y1 = train_y[:, 0]

        if len(self.estimators_) == 0:
            # Copy CNN to estimator:
            estimator = model  # deepcopy of self.base_estimator_
        else:
            # estimator = deepcopy(self.estimators_[-1])
            estimator = self.estimators_[-1]  # deepcopy CNN
        ###################################################
        if self.random_state_:
            estimator.set_params(random_state=1)
 #################################### CNN (3) binery label:       
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)

        lb=OneHotEncoder(sparse=False)
        y_b=train_y1.reshape(len(train_y1),1)
        y_b=lb.fit_transform(y_b)

        estimator.fit({'sequence_input': train_X1, 'profile_input': train_X2}, train_y, sample_weight=sample_weight, epochs=self.epochs, batch_size = self.batch_size)
############################################################
        y_pred = estimator.predict({'sequence_input': train_X1, 'profile_input': train_X2})
        ############################################ (4) CNN :

        y_pred_l = np.argmin(y_pred, axis=1)
        incorrect = y_pred_l != train_y1,
#########################################################        
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None
        #y_predict_proba=model.predict_proba({'sequence_input': train_X1, 'profile_input': train_X2,'main_input': train_X3, 'input_A': train_X4})

        # repalce zero
        #y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == train_y[:, np.newaxis])

        # for sample weight update
        # #intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
        #                                                       inner1d(y_coding, np.log(
        #                                                           y_predict_proba))))  #dot iterate for each row

        # update sample weight
        # sample_weight *= np.exp(intermediate_variable)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, 1, estimator_error
    
    # def deepcopy_CNN(self, base_estimator0):
    #     #Copy CNN (self.base_estimator_) to estimator:
    #     config=base_estimator0.get_config()
    #     estimator = base_estimator0.from_config(config)
    #     estimator = Sequential.from_config(config)
    #
    #
    #     weights = base_estimator0.get_weights()
    #     estimator.set_weights(weights)
    #     estimator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #     return estimator

    def discrete_boost(self, X, y, sample_weight):
#        estimator = deepcopy(self.base_estimator_)
         ############################################### my code:
           
        if len(self.estimators_) == 0:
            #Copy CNN to estimator:
            estimator = self.deepcopy_CNN(self.base_estimator_)#deepcopy of self.base_estimator_
        else: 
            #estimator = deepcopy(self.estimators_[-1])
            estimator = self.deepcopy_CNN(self.estimators_[-1])#deepcopy CNN
    ###################################################
        
        if self.random_state_:
            estimator.set_params(random_state=1)
#        estimator.fit(X, y, sample_weight=sample_weight)
#################################### CNN (3) binery label:       
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)
        
        lb=OneHotEncoder(sparse=False)
        y_b=y.reshape(len(y),1)
        y_b=lb.fit_transform(y_b)
        
        estimator.fit(X, y_b, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
############################################################        
        y_pred = estimator.predict(X)
        
        #incorrect = y_pred != y
 ############################################ (4) CNN :
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l != y
#######################################################   
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        # update estimator_weight
#        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
#            self.n_classes_ - 1)
        estimator_weight = self.learning_rate_ * (np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        if estimator_weight <= 0:
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, test_sequence,  test_profile):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(self._samme_proba(estimator, n_classes, test_sequence,  test_profile) for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
#            pred = sum((estimator.predict(X) == classes).T * w
#                       for estimator, w in zip(self.estimators_,
#                                               self.estimator_weights_))
########################################CNN disc
            pred = sum((estimator.predict(test_sequence,  test_profile).argmax(axis=1) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
###########################################################
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)



