import gym
import tensorflow as tf
import tensorflow_probability as tfp
from box import Box
import math
import numpy as np

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

#env = CartPoleNormalizedValues(gym.make("CartPole-v1"))
env = gym.make("CartPole-v1")
GAMMA = 0.99

Model = tf.keras.Sequential([
        tf.keras.Input(shape = (4,)),
        tf.keras.layers.Dense(64, activation = "relu", name = "layer1"),
        tf.keras.layers.Dense(2, activation = "softmax", name = "layer2")
    ])

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

def actioner(state):
    prob = Model(np.array([state]))
    dist = tfp.distributions.Categorical(probs=prob, dtype = tf.float32)
    action = dist.sample()
    return int(action.numpy()[0])

def trainLoop(Epochs):
    RewTracks = []
    ObsTracks = []
    ActTracks = []
    for _ in range(Epochs):
        obs, info = env.reset()
        for j in range(1000):
            ObsTracks.append(obs)
            act = actioner(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            RewTracks.append(reward)
            ActTracks.append(act)
            
            if terminated or truncated:
                break
        
        #Calculate proper rewards
        Temp0 = GAMMA ** j
        for k in range(2, j+2):
            RewTracks[-k] += GAMMA * RewTracks[-k+1]
            RewTracks[-k+1] *=  Temp0
            Temp0 /= GAMMA
    
    
    ActTracks = np.array(ActTracks)
    RewTracks = np.array(RewTracks)
    ObsTracks = np.array(ObsTracks)
    
    with tf.GradientTape() as tape:
        p = Model(ObsTracks, training = True)
        dist = tfp.distributions.Categorical(probs = p, dtype = tf.float32)
        log_prob = dist.log_prob(ActTracks)
        loss = -log_prob*RewTracks
    grads = tape.gradient(loss, Model.trainable_variables)
    optimizer.apply_gradients(zip(grads, Model.trainable_variables))
    
    return len(RewTracks)/Epochs

def testRun():
    #testenv = CartPoleNormalizedValues(gym.make("CartPole-v1", render_mode = "human"))
    testenv = gym.make("CartPole-v1", render_mode = "human")
    obs, info = testenv.reset()
    for j in range(1000):
        act = actioner(obs)
        obs, reward, terminated, truncated, info = testenv.step(act)
        
        if terminated or truncated:
            break
    
    testenv.close()
    return j


XProg = []
for i in range(65):
    XProg.append(trainLoop(10))
    print("{0}: {1}".format(i, XProg[-1]))

import matplotlib.pyplot as plt
plt.plot([x for x in range(len(XProg))], XProg)

