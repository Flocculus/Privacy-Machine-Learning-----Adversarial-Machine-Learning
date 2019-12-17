# -*- coding: utf-8 -*-
""" CIS6930PML -- Homework 3 -- nets.py

# This file contains the neural network architectures
"""

import os
import sys

import numpy as np
import keras

import tensorflow as tf

import matplotlib.pyplot as plt

"""
## Simple fully-connected classifier for MNIST
"""
def get_simple_classifier(flat_size = 28 * 28, num_labels = 10, num_hidden = 32, verbose = True, l2_regularization_constant = 0.0):
    # this is a simple feedforward neural network architecture for classification, it takes inputs of shape (flat_size,)
    # it has a hidden layer of 'num_hidden' neurons
    # finally there is softmax layer to output some probabilities over the class labels
    model = keras.models.Sequential()

    kernel_regularizer = None
    if l2_regularization_constant > 0:
        kernel_regularizer = keras.regularizers.l2(l2_regularization_constant)
    model.add(keras.layers.Dense(units=num_hidden, activation='sigmoid', input_shape=(flat_size,), kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dense(units=num_labels, activation='softmax', kernel_regularizer=kernel_regularizer))

    if verbose:
        model.summary()

    return model

"""
## More complex fully-connected classifier for MNIST with several hidden layers of varying size. 
## (Note the use of ReLU as activation function.)
"""
def get_deeper_classifier(flat_size=28*28, num_labels=10, num_hidden=[128, 96, 64, 32], verbose=True):

    # this is a deeper feedforward neural network architecture for classification, it takes inputs of shape (flat_size,)
    # it has several hidden layers with 'num_hidden' neurons
    # finally there is softmax layer to output some probabilities over the class labels
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(num_hidden[0], activation='sigmoid', input_shape=(flat_size,)))
    for s in num_hidden[1:]:
        model.add(keras.layers.Dense(units=s, activation='relu'))
    model.add(keras.layers.Dense(units=num_labels, activation='softmax'))

    if verbose:
        model.summary()

    return model

"""
## Trains the model given 'x_train' and 'y_train'
"""
def train_model(model, x_train, y_train, x_test, y_test, num_epochs, batch_size=64, optimalg='sgd', lossfn='categorical_crossentropy', metrics_list=['acc'], verbose=True):
    if x_test is None or y_test is None:
        verbose = False

    model.compile(optimizer=optimalg, loss=lossfn, metrics=metrics_list)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose, validation_split=.1)

    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=verbose)

    if x_test is None or y_test is None:
        return train_loss, train_accuracy, None, None

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)

    if verbose:
        fig = plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Testing'], loc='best')
        plt.show()

        print('Model was trained for {} epochs. Train accuracy: {:.1f}%, Test accuracy: {:.1f}%'.format(num_epochs, train_accuracy*100.0, test_accuracy*100.0))

    return train_loss, train_accuracy, test_loss, test_accuracy


"""
## Computes the categorical cross entropy loss
"""
def compute_loss(y_true, y_pred):
    loss = keras.backend.categorical_crossentropy(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), from_logits=False)
    return keras.backend.eval(loss)
