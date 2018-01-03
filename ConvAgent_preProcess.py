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
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam,RMSprop
from keras.models import load_model

def last_score(last_games,tot_reward):
    if len(last_games)!=100:
        last_games.append(tot_reward)
    else:
        tmp=last_games[1:100]#Drop earliest reference
        tmp.append(tot_reward)
        last_games=tmp
    return np.mean(last_games),np.std(last_games),last_games

def fearDeath(batch,precogFrames=10):
    D = deque(maxlen=2000)
    #Penalize the frames prior to death
    n,nEntries=0,len(batch)
    for i in batch:
        observation, action, reward, newObs, done=i#Unpack inputs.
        D.append((observation, action, reward, newObs, done)) 
        if done:
            nTmp=n#Where did we die?
            nStop=nTmp-precogFrames#Where should we stop fearing death?
            while nTmp>nStop and nTmp>0: 
                observation, action, reward, newObs, done=D[nTmp]
                reward-=100.#Adjust reward.
                D[nTmp]=observation, action, reward, newObs, done
                nTmp-=1#Move back one.
                
        n+=1
    return D

def preprocess(observation):
    #https://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

def buildBatch(D,batch_size):
    batch_type="Full"
    if batch_type=="Full": #Use all the data
        return D,len(D)
    if batch_type=="Every4":
        batch=[]#Every 4th frame gets added.
        for i in range(len(D)):
            if i%4==0:
                batch.append(D[i])  
        return batch,len(batch)
    return random.sample(D, batch_size),batch_size#If all else, just get a random batch to learn

def buildGame(gameName):
        env = gym.make(gameName)
        observation = env.reset() 
        observation=preprocess(observation)
        observation_size=np.shape(observation)
        action_size = env.action_space.n
        return env,observation_size,action_size

def buildConvModel(state_size,action_size,learning_rate):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=state_size))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
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
    model.compile(loss='mse', optimizer=RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0) )
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
        if done!=False:
            rewards[0,action]+= gamma*np.amax(future_rewards)#Add in future rew
        model.fit(observation,rewards,epochs=1,verbose=0)
    return model

def learn(env,nEpisodes,epsilon,gamma,model,batch_size):
#    D = deque(maxlen=2000)#Where do we put our observations?
    last_games=[]
    for i in range(nEpisodes):
        D = deque(maxlen=2000)
        observation=env.reset()
        observation=preprocess(observation)
#        print ("Observation space: ",np.shape(observation))
        done = False
        tot_reward = 0.0
        while not done:
            #env.render()
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]
            else:
                obs=np.expand_dims(observation,axis=0)
                Q = model.predict(obs) # Predict current reward for this observation
                action = np.argmax(Q) # Pick action that has best possible reward
            newObs, reward, done, info = env.step(action) #Step the game forward, record new observation
            newObs=preprocess(newObs)
            D.append((observation, action, reward, newObs, done))         # Add to Memory
            observation = newObs        # Update state
            tot_reward += reward #Add to reward
        fear=False
        if fear:
            D=fearDeath(D,precogFrames=10)
        avg_score,std_score,last_games=last_score(last_games,tot_reward)
        if i%100==0:
            print("Episode {}# Average Reward of Past 100 Games: {} +/- {}".format(i, avg_score,std_score))
        if i%1000==0 and i!=0:
            model.save("SpaceInvaders_model_INTERM.h5") 
        if len(D) >= batch_size:
            for i in range(1):
                batch,batch_size=buildBatch(D,batch_size)
                model=trainModel(model,batch,gamma)
        if epsilon > 0.1/0.995: #Decay random actions
            epsilon*=0.9995 
    model.save("SpaceInvaders_model_noFEAR.h5") #Save Final Model
    return model       
        
def play(model,env):
    #Watch your model play.
    observation=env.reset()
    done = False
    tot_reward = 0.0
    while not done:
        env.render()                    # Uncomment to see game running
        observation=preprocess(observation)
        observation=np.expand_dims(observation,axis=0)
        Q = model.predict(observation)        
        action = np.argmax(Q)         
        observation, reward, done, info = env.step(action)

        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))
    
def play100(model,env):
    #What is the average score of 100 games for this model?
    scores=[]
    for i in range(100):
        observation=env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            observation=preprocess(observation)
            observation=np.expand_dims(observation,axis=0)
            Q = model.predict(observation)        
            action = np.argmax(Q)         
            observation, reward, done, info = env.step(action)
            tot_reward += reward
        scores.append(tot_reward)
    print('Mean score of 100 games: {} +/- {}'.format(np.mean(scores),np.std(scores)))

gameName='SpaceInvaders-v0'  #210,160,3 RGBs
env,nObs,nAction=buildGame(gameName)
epsilon = 1.0  # Probability of doing a random move
gamma = 0.95    # Discounted future reward
batch_size=32
nEpisodes = 10000 #Worked best after 1000 entries.

warmModel=True
if warmModel:
    epsilon=0.01
    MASTERMIND=load_model("SpaceInvaders_model_INTERM.h5")
#    play(MASTERMIND,env)
#    play100(MASTERMIND,env)
else:
    MASTERMIND=buildDeeperConvModel(nObs,nAction)
MASTERMIND=learn(env,nEpisodes,epsilon,gamma,MASTERMIND,batch_size)
play(MASTERMIND,env)


