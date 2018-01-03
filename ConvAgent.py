#encoding: utf-8

#Written by JRF
#Inspired by :
#   https://github.com/GaetanJUVIN/Deep_QLearning_CartPole
#   https://www.youtube.com/redirect?v=79pmNdyxEGo&event=video_description&q=https%3A%2F%2Fgithub.com%2FllSourcell%2Fdeep_q_learning&redir_token=hVoT7lQUssUy2C-kPgTvax0N-j18MTUxMzE4MTE5NUAxNTEzMDk0Nzk1


import gym
import random
import numpy as np
from collections      import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam,RMSprop

def last_score(last_5,tot_reward):
    if len(last_5)!=5:
        last_5.append(tot_reward)
    else:
        tmp=last_5[1:5]#Drop earliest reference
        tmp.append(tot_reward)
        last_5=tmp
    avg_score=np.mean(last_5)    
    return avg_score,last_5

def buildGame(gameName):
        env = gym.make(gameName)
        observation = env.reset() 
        observation_size=np.shape(observation)
        action_size = env.action_space.n
        return env,observation_size,action_size

def buildConvModel(state_size,action_size,learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=state_size))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    #model.add(Dropout(0.1))
    
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(20, activation='softmax'))
    
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate) )

    return model

def buildDeeperConvModel(state_size,action_size):
    #From https://github.com/gtoubassi/dqn-atari/blob/master/dqn.py
    model = Sequential()
    # Second layer convolves 32 8x8 filters with stride 4 with relu
    model.add(Conv2D(32, (8, 8),strides=(4,4), activation='relu', input_shape=state_size))
 # Third layer convolves 64 4x4 filters with stride 2 with relu
    model.add(Conv2D(64, (4, 4),strides=(2,2), activation='relu', input_shape=state_size))
        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
    model.add(Conv2D(64, (3, 3),strides=(1,1), activation='relu', input_shape=state_size))
 # Fifth layer is fully connected with 512 relu units   
    model.add(Flatten()  )
    model.add(Dense(512, activation='linear'))   
    #Sixth linear layer
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) )
    #Defaults lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0
    #Paper lr=0.00025, rho=0.9, epsilon=0.01, decay=0.95

    return model

def trainModel(model,batch,gamma):
    for i in batch:
        observation, action, reward, newObs, done=i#Unpack inputs.
        observation=np.expand_dims(observation,axis=0)#Add another dimension to data
        newObs=np.expand_dims(newObs,axis=0)

        rewards=model.predict(observation)
        future_rewards=model.predict(newObs)
        rewards[0,action]=reward #In this example, the result of this action was reward
#        print "rewards",rewards  
        if done!=False:
            rewards[0,action]+= gamma*np.amax(future_rewards)#Add in future rew
#        print "rewards 2",rewards    
        model.fit(observation,rewards,epochs=1,verbose=0)
    return model

def learn(env,nEpisodes,epsilon,gamma,model,batch_size):
    D = deque(maxlen=2000)#Where do we put our observations?
    last_5=[]
    for i in range(nEpisodes):
        observation=env.reset()
        #print ("Observation space: ",np.shape(observation))
        observation=observation/255.0
        done = False
        tot_reward = 0.0
        while not done:
            #env.render()
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]
            else:
                #print "Prediction",np.shape(observation)
                obs=np.expand_dims(observation,axis=0)
                Q = model.predict(obs)
                action = np.argmax(Q)
            newObs, reward, done, info = env.step(action)
            D.append((observation, action, reward, newObs, done))         # Add to Memory

            newObs=newObs/255.0#Scale images
            observation = newObs        # Update state
            tot_reward += reward
        print("Episode {}# Total Reward: {}".format(i, tot_reward))
        avg_score,last_5=last_score(last_5,tot_reward)
        if i%10==0:
            print("Episode {}# Average Reward of Past 5 Games: {}".format(i, avg_score))
        if avg_score>350:
            model.save("SpaceInvaders_model_350.h5")
        if len(D) >= batch_size:
            for i in range(1):
                batch=random.sample(D, batch_size)
                model=trainModel(model,batch,gamma)
        if epsilon > 0.1/0.995: #Decay random actions
            epsilon*=0.9995    

def play(model,env):
    observation=env.reset()

    done = False
    tot_reward = 0.0
    while not done:
        env.render()                    # Uncomment to see game running
        observation=np.expand_dims(observation,axis=0)
        Q = model.predict(observation)        
        action = np.argmax(Q)         
        observation, reward, done, info = env.step(action)

        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))

        #observation_new, reward, done, info = env.step(action)
    #return 0

epsilon = 1.0  # Probability of doing a random move
gamma = 0.9    # Discounted future reward
batch_size=32
nEpisodes = 1000 #Worked best after 1000 entries.

#gameName='CartPole-v1'
gameName='SpaceInvaders-v0'  #210,160,3 RGBs
env,nObs,nAction=buildGame(gameName)
print nAction
#MASTERMIND=buildConvModel(nObs,nAction,0.001)
MASTERMIND=buildDeeperConvModel(nObs,nAction)
learn(env,nEpisodes,epsilon,gamma,MASTERMIND,batch_size)
