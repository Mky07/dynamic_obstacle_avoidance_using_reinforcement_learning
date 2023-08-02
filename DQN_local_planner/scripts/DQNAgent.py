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
import env3
import rospy
import rospkg
from utils import moving_average

import pickle
import matplotlib.pyplot as plt

import gc


class Feedback():
    def __init__(self):
        
        self.parent_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/models/"
        self.filename = "model5.pkl"
        self.file_path = self.parent_dir + self.filename
                
        # assign_params if not created
        self.params={
            "cumulated_rewards": [],
            "epsilon": 0.99
        }
            
        self.load()

    def save(self, params):
        """
        save parameters
        """
        
        with open(self.file_path, 'wb') as f:
            if not os.path.exists(self.file_path):
                pass
            else:
                self.params["cumulated_rewards"]+= params["cumulated_rewards"]
                self.params["epsilon"] = params["epsilon"]
            pickle.dump(self.params, f)

    def load(self):
        """
        load parameters
        """
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.params, f)

        with open(self.file_path, 'rb') as f:
            self.params = pickle.load(f)

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()    

class Model():
    def __init__(self, state_size, action_size):
        """
        create sequential model
        """
        # Parameters
        self.learning_rate = 0.0001

        self.state_size = state_size
        self.action_size = action_size

        self.model = Sequential()

        # input layer
        self.model.add(Dense(64, input_dim = self.state_size, activation='relu'))
        
        #hidden layer 1
        self.model.add(Dense(128, activation='relu'))
        
        #hidden layer 2
        self.model.add(Dense(32, activation='relu'))
        
        #output layer
        self.model.add(Dense(self.action_size, activation='linear'))

        self.model.compile(loss="mse", optimizer=Adam(learning_rate = self.learning_rate))

    def summary(self):
        return self.model.summary()

    def save_weights(self, name):
        self.model.save_weights(name)

class DQNAgent():
    def __init__(self, state_size, action_size, epsilon):
        self.state_size = state_size
        self.action_size = action_size
                
        # Q-Learning parameters
        self.gamma = 0.95
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon

        # other parameters
        self.batch_size = 32
        self.max_str = '00000'

        self.output_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/model_weights/" 

        if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # create neural network model
        self.model = Model(self.state_size, self.action_size)
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])        

    def replay(self, batch_size):
        if len(self.memory) <batch_size:
            return
        
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

    def load_latest_model(self):
        dir_list = os.listdir(self.output_dir)
        
        if dir_list:
            weights = []
            for fn in dir_list:
                weights.append(int(fn[8:13]))

            max_num = max(weights)
            self.max_str = f'{max_num:05d}'
            filename = "weights_"+self.max_str+".hdf5"
            self.load(self.output_dir + filename)
            
            print("[Agent] {} file has loaded.".format(filename))


class RL():
    def __init__(self):
        self.n_episodes = 10_000
        self.n_steps = 300

        self.env = gym.make('LocalPlannerWorld-v4')
        self.print_env()

        self.feedback = Feedback()
        
        epsilon = self.latest_feedback()["epsilon"]
        self.agent = DQNAgent(self.state_size(), self.action_size(), epsilon)
        self.agent.load_latest_model()

    def action_size(self):
        """
        action space:[move forward, turn left, turn right]
        """
        return self.env.action_space.n

    def state_size(self):
        """
        state space: [x,y, theta]
        """
        return self.env.observation_space.shape[0]

    def latest_feedback(self):
        """
        get latest feedback
        """
        return self.feedback.params

    def print_env(self):
        print(f"state size:{self.state_size()} - action size:{self.action_size()}")

    def draw_cumulative_rewards(self, data):
        test_data = data[:20100][::2]
        # print("cumulated rewards: {}".format(data[-200]))
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.plot(moving_average(test_data, 500))
        plt.show()

    def learning_phase(self):
        done = False

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size()])

            cumulated_reward = 0.0
            cumulated_rewards = []

            for time in range(self.n_steps):
                # env.render()
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                cumulated_reward+= reward
                next_state = np.reshape(next_state, [1, self.state_size()])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print("episode: {}/{}, score: {}, e:{:2}".format(e, self.n_episodes, time, self.agent.epsilon))
                    break
                        
            cumulated_rewards.append(cumulated_reward)

            if e % 50 == 0:
                self.agent.save(self.agent.output_dir + "weights_"+"{:05d}".format(int(self.agent.max_str)+e) + ".hdf5")

                params={
                    "cumulated_rewards": cumulated_rewards,
                    "epsilon": self.agent.epsilon
                }
                self.feedback.save(params)
                cumulated_rewards = []

        self.env.close()

if __name__ == '__main__':
    rospy.init_node('DQN_Agent', anonymous=True, log_level=rospy.WARN)
    rl = RL()
    rl.learning_phase()