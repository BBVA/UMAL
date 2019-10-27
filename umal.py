## Copyright 2019
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.layers import Flatten
from keras.optimizers import RMSprop, Adam
from keras.layers.core import Dense
from keras.models import Model
import numpy as np
from tqdm import tqdm

from keras.layers.advanced_activations import ELU

def elu_modif(x, a=1.,shift=5.,epsilon=1e-7):
    return ELU(alpha=a)(x+shift)+1.+epsilon

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    trick = K.sum(K.exp(x - x_max), axis=axis, keepdims=True)
    return K.log(trick)+x_max

def umal_log_pdf(y_true, parameters, general_type, n_taus=None):
    assert n_taus is not None
    components = K.reshape(parameters,[-1, n_taus, 3, 1])
    mu = components[:, :, 0]
    b = components[:, :, 1]
    tau = components[:, :, 2]
    error = K.expand_dims(y_true,1) - mu
    log_like = K.log(tau) + K.log(1.-tau) - K.log(b) - K.maximum(tau*(error),(tau-1.)*(error))/b
    sums = log_sum_exp(log_like, axis=1) - K.log(K.cast(n_taus,general_type))
    return - K.mean(K.sum(sums,axis=[1,2]),axis=0)
    

def build_UMAL(input_size, architecture, learning_rate = 0.001, training = True,
                mu_act = 'linear', b_act = lambda x: elu_modif(x,shift=0.,epsilon=1e-15)):
    """
    UMAL building model process.
    
    If training = True, the returned model is the keras trainable model given any DL architecture.
    If training = False, the returned graph is an inference TensorFlow model.
    
    mu_act is the activation function for the condioned position mu parameter of the ALD.
    b_act is the activation function for the conditioned scale b parameter of the ALD.
    
    Requirements: Given any set of hidden layers, noted as NN, the parameter 'architecture' has 
     to be a function with two inputs and returning the last hidden output as follows
    
        def template_architecture(input, name):
            penultimate_output = NN(input,name=name)
            return penultimate_output
        
    return: the built custom Keras model
    
    """

    general_type = K.floatx()
    i = Input(name='input', shape=(input_size,), dtype=general_type)
    
    if training:
        n_taus = K.variable(1,dtype=general_type.replace('float','int'))
        inputs = Lambda(lambda x: K.reshape(K.repeat(x,n_taus),shape=[-1,i.shape[-1]]))(i)
        tau = Lambda(lambda x: K.random_uniform(shape=[K.shape(inputs)[0],1],minval=1e-2,maxval=1.-1e-2,dtype=general_type))(inputs)
    else:
        sel_taus = tf.placeholder(general_type,shape=[None])
        tau = Lambda(lambda i: K.reshape(K.permute_dimensions(K.repeat(K.reshape(sel_taus,[-1,1]),K.shape(i)[0]),pattern=(1,0,2)),(-1,1)))(i)
        inputs = Lambda(lambda i: K.reshape(K.repeat(i,K.shape(sel_taus)[0]),shape=[-1,i.shape[-1]]))(i)
        
    it = concatenate([inputs,tau], axis=1, name='input_concat_1')
    model = architecture(it,name='_hiden_layers')
    mu = Dense(units=1, activation=mu_act, name='l_mu')(model)
    b = Dense(units=1, activation=b_act, name='l_b')(model)
    model_output = concatenate([mu, b, tau], axis=1, name='main_output')
    model = Model(inputs=[i], outputs=[model_output])
    
    if training:
        model.n_taus = n_taus 
    else:
        model.taus = sel_taus
        n_taus = K.shape(sel_taus)[0]
    
    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=(lambda y,p: umal_log_pdf(y, p, general_type, n_taus=n_taus)))
    model._hp = {'input_size':input_size, 'architecture':architecture,
                 'learning_rate':learning_rate, 'training':training, 'mu_act':mu_act, 
                 'b_act':b_act}
    return model

def np_log_sum_exp(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    trick = np.sum(np.exp(x - x_max), axis=axis, keepdims=True)
    return np.log(trick) + x_max

def ald_log_pdf(y, mu, b, tau):
    """
    Logarithm of the Asymmetric Laplace Probability density function
    """
    return np.where(
        y > mu,
        np.log(tau) + np.log(1 - tau) - np.log(b) - tau * (y - mu) / b,
        np.log(tau) + np.log(1 - tau) - np.log(b) - (tau - 1) * (y - mu) / b)

def minmax(pred, desv_from_minmax = 4):
    """
    For visualization part: Normal assumption of min-max values taking the 
     desv_from_minmax*sigmas deviation
    """
    pos_min = np.argmin(pred[0, :, :].flatten() - desv_from_minmax * pred[1, :, :].flatten())
    pos_max = np.argmax(pred[0, :, :].flatten() + desv_from_minmax * pred[1, :, :].flatten())

    return pred[0, :, :].flatten()[pos_min] - desv_from_minmax * pred[1, :, :].flatten()[pos_min], pred[0, :, :].flatten()[pos_max] + desv_from_minmax * pred[1, :, :].flatten()[pos_max]

def calculate_distribution(predictions, n_points_output=100, epsilon=1e-2, desv_from_minmax = 4):
    assert len(predictions.shape) == 3
    min_ = None
    max_ = None
    if type(n_points_output) == int:
        min_, max_ = minmax(predictions.T, desv_from_minmax)
        points = np.linspace(min_, max_, num=n_points_output)
        step = np.abs(points[1] - points[0]) / 2.
    else:
        points = n_points_output
        step = epsilon
        
    dist = []

    for i in tqdm(range(predictions.shape[1])):
        mu, b, taus = predictions[:, i, :].T
        if type(n_points_output) == int:
            dist += [[ald_log_pdf(p, mu, b, taus) for p in points]]
        else:
            dist += [ald_log_pdf(points[i], mu, b, taus)]
    dist = np.asarray(dist)
    return dist, min_, max_, points