#encoding: utf-8

#Written by JRF
#Inspired by :
#   https://github.com/GaetanJUVIN/Deep_QLearning_CartPole
#   https://www.youtube.com/redirect?v=79pmNdyxEGo&event=video_description&q=https%3A%2F%2Fgithub.com%2FllSourcell%2Fdeep_q_learning&redir_token=hVoT7lQUssUy2C-kPgTvax0N-j18MTUxMzE4MTE5NUAxNTEzMDk0Nzk1


import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

def buildGame(gameName):
        env = gym.make(gameName)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        return env,state_size,action_size

def buildModel(nLayers,nHidden,state_size,action_size,learning_rate):
        if nHidden%2!=0:#
            nHidden+=1
        model = Sequential()
        model.add(Dense(nHidden, input_dim=state_size, activation='relu'))
        for i in range(nLayers-1):
            model.add(Dense(nHidden/2, activation='relu'))
            if nHidden%2!=0:
                nHidden+=1
                if nHidden<=1:
                    break
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate) )
#        model = Sequential()
#        model.add(Dense(36, input_dim=state_size, activation='relu'))
#        model.add(Dense(24, activation='relu'))
##        model.add(Dense(action_size, activation='linear'))
#        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

        return model

def trainModel(model,batch,gamma):
    for i in batch:
        observation, action, reward, newObs, done=i#Unpack inputs.
        rewards=model.predict(observation)
        future_rewards=model.predict(newObs)
        rewards[0,action]=reward#In this example, the result of this action was reward
        if done!=False:
            rewards[0,action]+= gamma*np.amax(future_rewards)#Add in future rew
        model.fit(observation,rewards,epochs=1,verbose=0)
    return model

def play(env,nEpisodes,epsilon,gamma,model,batch_size):
    D = deque(maxlen=2000)#Where do we put our observations?
    for i in range(nEpisodes):
        observation=env.reset()
        observation=np.reshape(observation, [1, env.observation_space.shape[0]])
        done = False
        tot_reward = 0.0
        while not done:
            #env.render()
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]
            else:
                Q = model.predict(observation)
                action = np.argmax(Q)
            newObs, reward, done, info = env.step(action)
            newObs = np.reshape(newObs, [1, env.observation_space.shape[0]])
            D.append((observation, action, reward, newObs, done))         # Add to Memory
            observation = newObs        # Update state
            tot_reward += reward
        print("Episode {}# Total Reward: {}".format(i, tot_reward))
        if len(D) >= batch_size:
            batch=random.sample(D, batch_size)
            model=trainModel(model,batch,gamma)
            epsilon=epsilon*0.95

        #observation_new, reward, done, info = env.step(action)
    #return 0

epsilon = 0.99  # Probability of doing a random move
gamma = 0.9    # Discounted future reward
batch_size=32
nEpisodes = 1000 #Worked best after 1000 entries.

gameName='CartPole-v1'
env,nState,nAction=buildGame(gameName)
MASTERMIND=buildModel(1,20,nState,nAction,0.001)
play(env,nEpisodes,epsilon,gamma,MASTERMIND,batch_size)
