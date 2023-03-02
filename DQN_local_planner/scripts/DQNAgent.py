#!/usr/bin/env python3

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import gym
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as k
import env2
import rospy
import rospkg

import pickle
import matplotlib.pyplot as plt

import gc

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class Feedback:
    def __init__(self):
        # log_varaiables = defineLogVariables()
        self.log_values = {
            "cumulated_reward": [],
            "last_epsilon": 0.99
        }
        
        self.filename = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/models/model4.pkl"

        if not os.path.exists(self.filename):
            with open(self.filename, 'wb') as f:
                pickle.dump(self.log_values, f)     

        self.load()

    def save(self, cumulated_reward, epsilon):
        # save qlearn values to file
        with open(self.filename, 'wb') as f:
            self.log_values['cumulated_reward'].append(cumulated_reward)
            self.log_values['last_epsilon'] = epsilon
            pickle.dump(self.log_values, f)

    def load(self):
            # load qlearn valeus from file
        with open(self.filename, 'rb') as f:
            self.log_values = pickle.load(f)



class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()    

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)
        # q learning params
        self.gamma = 0.95

        self.epsilon = 0.999
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05

        # neural network params
        self.learning_rate = 0.0001


        self.model = self._build_model()
        self.model.summary()

    def get_epsilon(self):
        return self.epsilon

    def _build_model(self):
        model = Sequential()

        model.add(Dense(64, input_dim = self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(learning_rate = self.learning_rate))
        # model.compile(loss="mse", optimizer=Adam(learning_rate = self.learning_rate), run_eagerly=True)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])        

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=ClearMemory())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == '__main__':
    rospy.init_node('DQN_Agent', anonymous=True, log_level=rospy.WARN)

    env = gym.make('LocalPlannerWorld-v1')

    # Set parameters
    output_dir="../model_output4/"

    feedback = Feedback()
    log_values = feedback.log_values['cumulated_reward']
    goal_reached_episode = [i for i in log_values if i>=0]
    print("\n\n************************************\n\ntoplam episode:{} Hedefe ulaşmiş episode:{} \n\n************************************\n\n".format(len(log_values), len(goal_reached_episode)))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # states: [x,y, theta]
    state_size = env.observation_space.shape[0]
    
    # actions:[move forward, turn left, turn right]
    action_size = env.action_space.n

    print(f"state size: {state_size}")
    print(f"action size: {action_size}")

    agent = DQNAgent(state_size, action_size)

    agent.epsilon = feedback.log_values['last_epsilon']

    path = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/model_output4/"
    dir_list = os.listdir(path)

    max_str = '00000'
    if dir_list:
        weights = []
        for fn in dir_list:
            weights.append(int(fn[8:13]))

        max_num = max(weights)
        max_str = f'{max_num:05d}'
    
        filename = "weights_"+max_str+".hdf5"
        print("[Agent] This file will be loaded. filename: {}".format(filename))
        agent.load(path + filename)

    batch_size = 32

    done = False
    n_episodes = 10_000

    cumulated_rewards = feedback.log_values['cumulated_reward']

    plt.plot(moving_average(cumulated_rewards, 200))
    plt.show()

    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        cumulated_reward = 0.0
        
        for time in range(300):

            # env.render()
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            
            cumulated_reward+= reward
            
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e:{:2}".format(e, n_episodes, time, agent.epsilon))
                break
        
        cumulated_rewards.append(cumulated_reward)
        print("cumulated_rewards: {}".format(cumulated_rewards[-100:]))
        
        # save to file
        feedback.save(cumulated_reward, agent.get_epsilon()) 

        if len(agent.memory) >batch_size:
            agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(output_dir + "weights_"+"{:05d}".format(int(max_str)+e) + ".hdf5")    
    env.close()