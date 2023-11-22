# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:53:14 2023

@author: prath
File: D:\Progress\Machine Learning\Scratch
"""
import gym
import DNNScratch
import numpy as np
from box import Box
import math
import random
import time


class CartPoleNormalizedValues(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prevObserv = env.observation_space
        self.observation_space = Box(shape = (4,), low = [0, 0, 0, 0], high = [1, 1, 1, 1])
        
    def observation(self, obs):
        return [obs[0]/self.prevObserv.high[0], 
                math.atan(obs[1])/math.pi*2, 
                obs[2]/self.prevObserv.high[2], 
                math.atan(obs[3])/math.pi*2]

class RLDNN(DNNScratch.NeuralNet):
    def __init__(self, Layers, Activations, learningRate):
        super().__init__(Layers, Activations, "Custome", learningRate, 10)
        
    def Train(self, Observations, Actions, Rewards):
        Comp = self.ForwardProp(Observations)
        
        #Get the T matrix which is get the highest  value from each output and invert it
        #Tprep = np.argmax(Comp[-1], axis=0)
        Tprep = np.array(Actions)
        Tval = np.zeros((Tprep.size, Tprep.max()+1))
        Tval[np.arange(Tprep.size), Tprep] = 1
        Tprep = (1/np.clip(Comp[-1], 0.001, 0.999)).transpose() * Tval
        
        assert Tprep.shape[0] == len(Rewards)
        
        for i in range(Tprep.shape[0]):
            for j in range(Tprep.shape[1]):
                Tprep[i][j] *= Rewards[i]
        
        MChanges = [[] for x in range(len(self.M))]
        V = np.multiply( Tprep, self.ActivationsDer[-1](Comp[-1]).transpose() )
        
        #Make changes to learning methods
        MChanges[-1] = self.learningRate * np.matmul(Comp[-2], V).transpose()
        
        for i in range(2, len(self.M)+1):
            V = np.multiply( np.matmul(V, self.M[1-i]), self.ActivationsDer[-i](Comp[-i]).transpose() )
            #Make changes to learning methods
            MChanges[-i] = self.learningRate * np.matmul(Comp[-1-i], V).transpose()
        
        for i in range(len(self.M)):
            self.M[i] += MChanges[i]
        
class GymEnv:
    def __init__(self, environment, HiddenLayers, HiddenActivations, 
                 learningRate, EpochSize, Gamma, Randomness, RandomnessDecay):
        self.env = environment
        HiddenLayers = [environment.observation_space.shape[0]] + HiddenLayers + [environment.action_space.n]
        self.Policy = RLDNN(HiddenLayers, HiddenActivations + ['Softmax'], learningRate)
        self.EpochSize = EpochSize
        self.Gamma = Gamma
        self.ObservationVariables = environment.observation_space.shape[0]
        self.ActionVariables = environment.action_space.n
        self.Randomness = Randomness
        self.RandomnessDecay = RandomnessDecay
        self.envTest = CartPoleNormalizedValues(gym.make('CartPole-v1', render_mode="human"))
        
        self.MTemp = []
    
    def TrainRun(self):
        ObsTracks = []
        RewTracks = []
        ActTracks = []
        CummulativeRewards = 0
        for i in range(self.EpochSize):
            observation, info = self.env.reset()
            for j in range(1000):
                ObsTracks.append(observation)
                observation = np.reshape(observation, newshape=(self.ObservationVariables, 1))
                
                action = self.Policy.ForwardProp(observation)[-1]
                diceRoll = random.random()
                if(diceRoll >= self.Randomness):
                    action = np.argmax(action, axis = 0)[0]
                else:
                    action = random.randint(0, self.ActionVariables-1)
                ActTracks.append(action)
                
                observation, reward, terminated, truncated, info = self.env.step(action)
                RewTracks.append(reward)
                CummulativeRewards += reward
                
                if terminated or truncated:
                    Temp0 = self.Gamma ** j
                    for k in range(2, j+2):
                        RewTracks[-k] += self.Gamma * RewTracks[-k+1]
                        RewTracks[-k+1] *=  Temp0
                        Temp0 /= self.Gamma
                    #print("RewTracks: {0}".format(RewTracks))
                    break
        
        self.Randomness *= self.RandomnessDecay
        ObsTracks = np.reshape(ObsTracks, newshape=(len(ObsTracks), len(ObsTracks[0]))).transpose()
        
        
        self.Policy.Train(ObsTracks, ActTracks, RewTracks)
        return CummulativeRewards/self.EpochSize
    
    def Train(self, NumberOfPlays):
        j = 0
        for i in range(NumberOfPlays):
            AvgRew = self.TrainRun()
            j += 1
            if(j >= NumberOfPlays/10):
                self.MTemp.append(self.Policy.M)
                self.TestRun()
                print("Index{0}\tAverage Reward: {1}\tRandomness: {2}".format(i+1, AvgRew, self.Randomness))
                j = 0
    
    
    
    def TestRun(self):

        RewTracks = []
        ObsTracks = []
        obs, info = self.envTest.reset()
        for j in range(1000):
            ObsTracks.append(obs)
            obs = np.reshape(obs, newshape=(self.ObservationVariables, 1))
            action = self.Policy.ForwardProp(obs)[-1]
            action = np.argmax(action, axis = 0)[0]
            obs, reward, terminated, truncated, info = self.envTest.step(action)
            RewTracks.append(reward)
            
            if terminated or truncated:
                print(len(RewTracks))
                break
    

if __name__ == "__main__":
    LEARNING_RATE = 1e-5
    EPOCH_SIZE = 100
    ENV_NAME = 'CartPole-v1'
    GAMMA = 1
    RANDOMNESS = 1.0
    RANDOMNESSDECAY = 0.995
    
    env = CartPoleNormalizedValues(gym.make(ENV_NAME))
    #env = gym.make(ENV_NAME)
    Trainer = GymEnv(env, [64], ['ReLu'], LEARNING_RATE, EPOCH_SIZE, GAMMA, RANDOMNESS, RANDOMNESSDECAY)
    Trainer.Train(100)
    

