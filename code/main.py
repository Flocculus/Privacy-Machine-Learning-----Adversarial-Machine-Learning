# -*- coding: utf-8 -*-
"""
# This file is the main file
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# we'll use keras for neural networks
import keras
from keras.datasets import mnist

import os

# our neural network architectures
import nets

import attacks

## os / paths
def ensure_exists(dir_fp):
    if not os.path.exists(dir_fp):
        os.makedirs(dir_fp)

## parsing / string conversion to int / float
def is_int(s):
    try:
        z = int(s)
        return z
    except ValueError:
        return None


def is_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return None


"""
## Load and preprocess the dataset
"""
def load_preprocess_mnist_data(train_size=20000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # MNIST has overall shape (60000, 28, 28) -- 60k images, each is 28x28 pixels
    print('Loaded mnist data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape,
                                                                                      x_test.shape, y_test.shape))
    # Let's flatten the images for easier processing (labels don't change)
    flat_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
    x_test = x_test.reshape(x_test.shape[0], flat_vector_size)

    # Put the labels in "one-hot" encoding using keras' to_categorical()
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # let's split the training set further
    aux_idx = train_size

    x_aux = x_train[aux_idx:,:]
    y_aux = y_train[aux_idx:,:]

    x_temp = x_train[:aux_idx,:]
    y_temp = y_train[:aux_idx,:]

    x_train = x_temp
    y_train = y_temp

    return (x_train, y_train), (x_test, y_test), (x_aux, y_aux)


"""
## Plots an image or set of images (all 28x28)
## input is either a single image, i.e., np.array with shape (28,28), or a square number of images, i.e., np.array with shape (z*z, 28, 28) for some integer z > 1
"""
def plot_image(im, fname='out.png', show=False):
    fig = plt.figure()
    im = im.reshape((-1,28, 28))

    num = im.shape[0]
    assert num <= 3 or np.sqrt(num)**2 == num, 'Number of images is too large or not a perfect square!'
    if num <= 3:
        for i in range(0, num):
            plt.subplot(1, num, 1 + i)
            plt.axis('off')
            plt.imshow(im[i], cmap='gray_r') # plot raw pixel data
    else:
        sq = int(np.sqrt(num))
        for i in range(0, num):
            plt.subplot(sq, sq, 1 + i)
            plt.axis('off')
            plt.imshow(im[i], cmap='gray_r') # plot raw pixel data

    ensure_exists('./plots')
    out_fp = './plots/{}'.format(fname)
    plt.savefig(out_fp)

    if show is False:
        plt.close()
    else:
        plt.show()


"""
## Plots an adversarial perturbation, i.e., original input x, adversarial example x_adv, and the difference (perturbation)
"""
def plot_adversarial_example(x, x_adv, fname='adv_ex.png', show=False, cmap='gray_r'):
    fig = plt.figure()
    imx = x.reshape((-1,28, 28))
    imadv = x_adv.reshape((-1, 28, 28))

    assert imx.shape[0] == 1, 'Unexpected image dimensions (is this not a MNIST image?)!'
    assert imadv.shape[0] == 1, 'Unexpected image dimensions (is this not a MNIST image?)!'

    perturb = imx - imadv   # this is the perturbation

    ax1 = plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(imx[0], cmap=cmap)

    ax2 = plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(perturb[0], cmap=cmap)

    ax3 = plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(imadv[0], cmap=cmap)

    ax1.set_title('Input')
    ax2.set_title('Perturbation')
    ax3.set_title('Adv. Example')

    ensure_exists('./plots')
    out_fp = './plots/{}'.format(fname)
    plt.savefig(out_fp)

    if show is False:
        plt.close()
    else:
        plt.show()


"""
## Setup tf session. Please customize to your specific environment!
"""
def setup_session(verbose=False):

    # feel free to customize the session to your specific environment
    #config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=verbose))
    #sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    gpus = keras.backend.tensorflow_backend._get_available_gpus()
    if len(gpus) == 0:
        print('Warning: no available GPU(s)!')




## this is the main
def main():

    # figure out the problem number
    assert len(sys.argv) >= 5, 'Incorrect number of arguments!'
    p_split = sys.argv[1].split('Step')
    assert len(p_split) == 2 and p_split[0] == '', 'Invalid argument {}.'.format(sys.argv[1])
    problem_str = p_split[1]

    assert is_number(problem_str) is not None, 'Invalid argument {}.'.format(sys.argv[1])
    problem = float(problem_str)
    probno = int(problem)

    if probno < 0 or probno > 2:
        assert False, 'Step {} is not a valid problem # for this assignment/homework!'.format(problem)

    ## change this line to override the verbosity behavior
    verb = True if probno == 0 else False

    # get arguments
    model_str = sys.argv[2]
    if model_str.startswith('simple'):
        simple_args = model_str.split(',')
        assert simple_args[0] == 'simple' and len(simple_args) == 3, '{} is not a valid network description string!'.format(model_str)
        hidden = is_int(simple_args[1])
        reg_const = is_number(simple_args[2])
        assert hidden is not None and hidden > 0 and reg_const is not None and reg_const >= 0.0, '{} is not a valid network description string!'.format(model_str)
        target_model_train_fn = lambda: nets.get_simple_classifier(num_hidden=hidden, l2_regularization_constant=reg_const,
                                                                   verbose=verb)
    elif model_str == 'deep':
        target_model_train_fn = lambda: nets.get_deeper_classifier(verbose=verb)
    else:
        assert False, '{} is not a valid network description string!'.format(model_str)

    train_size = is_int(sys.argv[3])
    num_epochs = is_int(sys.argv[4])

    assert train_size is not None and 100 <= train_size <= 50000, '{} is not a valid size for the target model training dataset!'.format(sys.argv[3])
    assert num_epochs is not None and 0 < num_epochs <= 10000, '{} is not a valid size for the number of epochs to train the target model!'.format(sys.argv[4])

    setup_session() # setup the tf session

    # load the dataset
    train, test, aux = load_preprocess_mnist_data(train_size=train_size)
    x_train, y_train = train
    target_train_size = x_train.shape[0]
    x_test, y_test = test
    x_aux, y_aux = aux

    dirpath = os.path.join(os.getcwd(), 'models')
    ensure_exists(dirpath)
    base_fp = '{}/{}_trainsz{}_epochs{}'.format(dirpath, model_str, train_size, num_epochs)

    train_model = probno == 0 # train
    if train_model == True:
        assert len(sys.argv) == 5, 'Incorrect number of arguments!'
        model = target_model_train_fn()  # compile the target model

        # train the target model
        train_loss, train_accuracy, test_loss, test_accuracy = nets.train_model(model, x_train, y_train, x_test, y_test, num_epochs, verbose=verb)

        print('Trained target model on {} records. Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size,
                                                                                    100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))

        # save the model: first the weights then the arch
        model.save_weights('{}-weights.h5'.format(base_fp))
        with open('{}-architecture.json'.format(base_fp), 'w') as f:
            f.write(model.to_json())
    else:
        # Model reconstruction from JSON file
        with open('{}-architecture.json'.format(base_fp), 'r') as f:
            model = keras.models.model_from_json(f.read())

        # Load weights into the new model
        model.load_weights('{}-weights.h5'.format(base_fp))

        print('Loaded model from file ({}).'.format(base_fp))


    def distortion(x_in, x_adv):
        ## TODO ##
        ## Insert your code here to calculate the distortion
        return np.linalg.norm(x_in - x_adv)

    def done_fn(model, x_in, x_adv, target):
        if np.max(model.predict(x_adv)) >= 0.9:
            return np.argmax(model.predict(x_adv), axis=-1) == target
        return False

    ## TODO ##
    ## Insert your code here change the termination criterion


    if probno == 1: ## problem 1
        assert len(sys.argv) == 7, 'Incorrect number of arguments!'

        input_idx = is_int(sys.argv[5])
        assert 0 <= input_idx <= x_aux.shape[0], 'Invalid input index (must be between 0 and {})!'.format(x_aux.shape[0])
        target_label = is_int(sys.argv[6])
        assert 0 <= target_label <= 9, 'Invalid target class label!'

        max_iter = 100      # maximum number of iterations for the attack

        x_input = x_aux[input_idx, :].reshape((1, -1))

        y_true_lab = np.argmax(y_aux[input_idx, :], axis=-1)

        y_pred = model.predict(x_input).reshape(-1)
        y_pred_lab = np.argmax(y_pred, axis=-1)

        print('\nSelected input {} -- true label: {}, predicted label: {} (confidence: {:.2f}%) -- target label for perturbation: {}'.format(
                    input_idx, y_true_lab, y_pred_lab, 100.0*float(y_pred[y_pred_lab]), target_label))
        assert target_label != y_pred_lab, 'Target label should NOT be the same as predicted label!'



        ## turn-off pixels attack
        x_in = x_input.copy()

        t = time.process_time()
        print('\nRunning the turn-off pixels iterative attack (<={} iterations).'.format(max_iter))
        x_adv, iters = attacks.turn_off_pixels_iterative_attack(model, x_in, target_label, max_iter, terminate_fn=done_fn)
        elapsed_time = time.process_time() - t

        y_adv = model.predict(x_adv).reshape(-1)
        y_label = np.argmax(y_adv, axis=-1)

        status = '\tAttack failed'
        if done_fn(model, x_in, x_adv, target_label):
            status = '\tAttack successful ({} iterations - {:.1f} seconds)'.format(iters, elapsed_time)

        print('{}, the adversarial example is classified as \'{}\' with {:.2f}% confidence by the model!'.format(
                    status,  y_label, 100.0 * float(y_adv[y_label])))

        print('\tDistortion: {:.2f}'.format(distortion(x_input, x_adv)))
        fname = 'adv_turn-off_I_' + str(target_label) + '_I_distortion_' + str(distortion(x_input, x_adv)) + "_I_confidence" + str((y_adv[y_label])) + '.png'
        plot_adversarial_example(x_input, x_adv, show=True, fname=fname)
        
        ## turn-on pixels attack
        x_in = x_input.copy()

        t = time.process_time()
        print('\nRunning the turn_on pixels iterative attack (<={} iterations).'.format(max_iter))
        x_adv, iters = attacks.turn_on_pixels_iterative_attack(model, x_in, target_label, max_iter, terminate_fn=done_fn)
        elapsed_time = time.process_time() - t

        y_adv = model.predict(x_adv).reshape(-1)
        y_label = np.argmax(y_adv, axis=-1)

        status = '\tAttack failed'
        if done_fn(model, x_in, x_adv, target_label):
            status = '\tAttack successful ({} iterations - {:.1f} seconds)'.format(iters, elapsed_time)

        print('{}, the adversarial example is classified as \'{}\' with {:.2f}% confidence by the model!'.format(
            status, y_label, 100.0 * float(y_adv[y_label])))

        print('\tDistortion: {:.2f}'.format(distortion(x_input, x_adv)))
        plot_adversarial_example(x_input, x_adv, show=True, fname='adv_turn-on')
        
    elif probno == 2: ## problem 2

        assert len(sys.argv) == 6, 'Incorrect number of arguments!'

        target_label = is_int(sys.argv[5])
        assert 0 <= target_label <= 9, 'Invalid target class label!'

        max_iter = 150  # maximum number of iterations for the attack

        def random_image(size=(1,28*28)):
            ## TODO ##
            ## Insert your code here
            #raise NotImplementedError()
            return np.random.uniform(0.0,255.0,size)
        # distribution of predictions for random images of the model

        # turn this flag to False after answering question 2.1
        show_distribution = False
        #show_distribution = False
        if show_distribution:
            num_samples = 1000
            predictions = np.zeros((10,))
            for i in range(0, num_samples):
                x_rand = random_image()
                y_pred = model.predict(x_rand).reshape(-1)
                y_pred_lab = np.argmax(y_pred, axis=-1)
                predictions[y_pred_lab] += 1.0
            fig = plt.figure()
            x = np.arange(0,10)+0.5
            plt.bar(x, (predictions / np.sum(predictions))*100.0)
            plt.ylabel('Percentage of random images')
            plt.xlabel('Class label')
            plt.xticks(ticks=x,labels=range(10))
            plt.show()

            assert False


        x_input = random_image()
        y_pred = model.predict(x_input).reshape(-1)
        y_pred_lab = np.argmax(y_pred, axis=-1)

        print('\nRandom input predicted label: {} (confidence: {:.2f}%) -- target label for perturbation: {}'.format(
                y_pred_lab, 100.0 * float(y_pred[y_pred_lab]), target_label))
        assert target_label != y_pred_lab, 'Target label should NOT be the same as predicted label (try again)!'

        ## turn-off pixels attack
        x_in = x_input.copy()

        t = time.process_time()
        print('\nRunning the pixels iterative attack (<={} iterations).'.format(max_iter))

        x_adv, iters = attacks.turn_off_pixels_iterative_attack(model, x_in, target_label, max_iter,terminate_fn=done_fn)
        #x_adv, iters = attacks.turn_on_pixels_iterative_attack(model, x_in, target_label, max_iter, terminate_fn=done_fn)

        elapsed_time = time.process_time() - t

        y_adv = model.predict(x_adv).reshape(-1)
        y_label = np.argmax(y_adv, axis=-1)

        status = '\tAttack failed'
        if done_fn(model, x_in, x_adv, target_label):
            status = '\tAttack successful ({} iterations - {:.1f} seconds)'.format(iters, elapsed_time)

        print('{}, the adversarial example is classified as \'{}\' with {:.2f}% confidence by the model!'.format(
            status, y_label, 100.0 * float(y_adv[y_label])))

        plot_adversarial_example(x_input, x_adv, show=True, fname='random_turn-off')

if __name__ == '__main__':
    main()
