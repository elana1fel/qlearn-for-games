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
import shutil
import datetime

from keras.utils.visualize_util import plot
counter=1
GAME = 'snake' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 5 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

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

def trainNetwork2(model1,model2,args):
    # open up a game state to communicate with emulator

    a_file =open("logs_" + GAME + "/Corelation_ratio" + str(counter) + ".txt", 'a')
    scores_file =open("logs_" + GAME + "/scores" + str(counter) + ".txt", 'a')
    game_state = game.GameState()
    # store the previous observations in replay memory
    D1 = deque()
    D2 = deque()

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

    learning_mode = 1  # 2 for learng based on human, 3 for reverse reinforcement
    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model2.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
        training_mode = False  # running
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        learning_mode=int(args['learning_mode'])
        if (learning_mode==6):
            active_study=False #true when you study from player

        else:
            active_study=True

        if os.path.isfile("model2.h5"): #check if file exists.
            model2.load_weights("model2.h5")
            adam = Adam(lr=1e-6)
            model2.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")

        else:
            rau_input("ERRRROR")

        os.mkdir("trials" , 0755);

        training_mode = True # training

    j = 0
    t = 0
    action_sum = 0
    same_action = 0
    num_folder=0
    start_time=datetime.datetime.now()
    time_last_reward=datetime.datetime.now()
    corelation_ratio = 0
    high_score = 0

    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])


	
	current_time=datetime.datetime.now()
        elapsedTime = (current_time - time_last_reward).total_seconds()  
        if ((learning_mode==6) and not active_study):
            #active_stady is false
            q1 = model1.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q1 = np.argmax(q1)
            action_index = max_Q1
            a_t[action_index] = 1

        elif (elapsedTime>=1*60):
            a_t[0] = 1

        else:
            q2 = model2.predict(s_t)  # input a stack of 4 images, get the prediction
            max_Q2 = np.argmax(q2)
            action_index = max_Q2
            a_t[action_index] = 1

            if(learning_mode==4) or (learning_mode==5) or(learning_mode==6) :
                q1 = model1.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q1 = np.argmax(q1)
                action_index_temp = max_Q1
                action_sum += 1
                if (action_index_temp==action_index):
                    same_action += 1

                corelation_ratio=same_action/float(action_sum)
                print("corelation_ratio is: ", corelation_ratio)

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal,score = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

        # store the transition in D
        D2.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D2) > REPLAY_MEMORY:
            D2.popleft()

        game_over=terminal
	if (r_t==1) or (r_t==-1):
	    time_last_reward=datetime.datetime.now()

        if (learning_mode==5):
            if (action_index_temp==action_index):
                r_t=1

            else:
                r_t=-1
                action_index=action_index_temp

        D1.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D1) > REPLAY_MEMORY:
            D1.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D1, BATCH)

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

                targets[i] = model1.predict(state_t)  # Hitting each buttom probability
                Q_sa = model1.predict(state_t1)

                if terminal or (learning_mode==5):
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)


            # targets2 = normalize(targets)
            loss += model1.train_on_batch(inputs, targets)
        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 100 == 0:
            #print("Now we save model")
            model1.save_weights("model1.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model1.to_json(), outfile)

	current_time=datetime.datetime.now()
        elapsedTime = (current_time - start_time).total_seconds()  


	if(elapsedTime>=30*60):	
            num_folder+=1		
            print(num_folder, "corelation ratio", corelation_ratio, "time step:", t, file=a_file)
            a_file.flush()

	    os.mkdir("trials/" + str (num_folder), 0755);
	    shutil.copy2('model1.h5', 'trials/' + str (num_folder) + '/model1.h5')
          
            action_sum = 0
            same_action = 0
	    if learning_mode==6:	:
	        active_study=False
            start_time=datetime.datetime.now()
          
	if learning_mode==6 and (corelation_ratio>=85 or num_folder>48)	:
            active_study=False

	elif learning_mode==6 and elapsedTime>=15*60 and not active_study:	:
            active_study=True

        if game_over and active_study:
            if(score>high_score):
                high_score=score
                j = j + 1
		print(j, "score is:", score, "time step:", t, file=scores_file)
		scores_file.flush()



    print("Episode finished!")
    print("************************")

    finish_time = time.clock()
    print("finish time:", strftime("%Y-%m-%d %H:%M:%S", gmtime()), file=a_file)
    a_file.close

def playGame(args):
    model1 = buildmodel()
    model2 = buildmodel()

    trainNetwork2(model1,model2,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-l','--learning_mode', help='1,2,3,4', required=False)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
