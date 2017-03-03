#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wormy_fun as game
import random
import numpy as np
from collections import deque

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

import pygame
from pygame.locals import *
from time import gmtime, strftime
import os.path
import datetime
import shutil

from keras.utils.visualize_util import plot
counter=1
GAME = 'snake' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 5 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 320. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

results_file = open("logs_" + GAME + "/resultsfile_" + str(counter) + ".txt", 'a')
average_file = open("logs_" + GAME + "/averagefile_" + str(counter) + ".txt", 'a')
weights_file = open("logs_" + GAME + "/weightsfile_" + str(counter) + ".txt", 'a')
reward_file = open("logs_" + GAME + "/rewardfile_" + str(counter) + ".txt", 'a') #average reward for each step
reward_final_file = open("logs_" + GAME + "/reward_finalfile_" + str(counter) + ".txt", 'a') #average reward for final step
q_file = open("logs_" + GAME + "/q_averagefile_" + str(counter) + ".txt", 'a') #qmax average for each step
qmax_file = open("logs_" + GAME + "/qfile_" + str(counter) + ".txt", 'a') #qmax for eac step
loss_avgfile = open("logs_" + GAME + "/loss_avgfile_" + str(counter) + ".txt", 'a') # loss average per game

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    #model.summary()
    print("We finish building the model")
    plot(model, to_file='model.png', show_shapes=True)
    return model

def trainNetwork(model,args, num_try):
    # open up a game state to communicate with emulator

    game_state = game.GameState()
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal,_ = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    if args['mode'] == 'Run':
        print ("Run mode")
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("trials/" + str(num_try) +"/model1.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
        training_mode = False  # running

    j = 0
    t = 0
    start_time=datetime.datetime.now()
    num_folder=0
    scores_list = []
    reward_list = []
    max_q_list = []
    loss_per_game_list= []
    reward_sum=0

    while (j<10):
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
	current_time=datetime.datetime.now()
        elapsedTime = (current_time - start_time).total_seconds()  
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:		
            q = model.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            if(elapsedTime>=1*60):
            	a_t[0] = 1
            else:
		action_index = max_Q
            	a_t[action_index] = 1

		q_max=np.amax(q)
		max_q_list.append(q_max)
		weights_file.write(str(num_try) + " weights: " + str(q) + "\n")
		weights_file.flush()
		qmax_file.write(str(num_try) + " max q: " + str(q_max) + "\n")
		qmax_file.flush()


        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal,score = game_state.frame_step(a_t)

        game_over=terminal
	if (r_t==1) or (r_t==-1):
	    start_time=datetime.datetime.now()

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if j==9:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t
                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)


                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss = model.test_on_batch(inputs, targets)
            loss_per_game_list.append(loss)

        reward_sum+=r_t
        s_t = s_t1
        t = t + 1


        if (game_over):
            reward_list.append(reward_sum)
            reward_file.write(str(num_try) + " reward : " + str(reward_sum) + "\n")
            reward_file.flush()
            reward_sum=0


            j = j + 1
            
            results_file.write(str(num_try) + " score: " + str(score) + " timestep: " + str(t) + "\n")
            results_file.flush()
            scores_list.append(score)

    average = np.mean(scores_list)
    average_file.write(str(num_try) + " average score: " + str(average) + "\n")
    average_file.flush()

    reward_final_average = np.mean(reward_list)
    reward_final_file.write(str(num_try) + " reward average: " + str(reward_final_average) + "\n")
    reward_final_file.flush()

    max_q_average=np.mean(max_q_list)
    q_file.write(str(num_try) + " max q average : " + str(max_q_average) + "\n")
    q_file.flush()

    loss_average=np.mean(loss_per_game_list)
    loss_avgfile.write(str(num_try) + " loss average : " + str(loss_average) + "\n")
    loss_avgfile.flush()



def playGame(args):
    model = buildmodel()
    
    for x in range(7, 16):
        print (x)
        trainNetwork(model,args,x)

    results_file.close
    weights_file.close
    average_file.close
    reward_file.close
    reward_final_file.close
    q_file.close
    qmax_file.close
    loss_avgfile.close


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
