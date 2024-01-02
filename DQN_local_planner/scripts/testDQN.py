#!/usr/bin/env python3

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import gym
import numpy as np
from collections import deque

from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as k
import testenv
import rospy
import rospkg
from utils import moving_average

import pickle
import matplotlib.pyplot as plt

import gc



class Feedback():
    def __init__(self):
        
        self.parent_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/models/"
        self.filename = "t2.pkl"
        self.file_path = self.parent_dir + self.filename
                
        # assign_params if not created
        self.params={
            "cumulated_rewards": [],
            "epsilon": 0.99,
            "goal_reached_count": 0,
            'histories': []
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
                self.params["histories"]+= params["histories"]
                self.params["epsilon"] = params["epsilon"]
                self.params['goal_reached_count']+= params['goal_reached_count']
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

class NNModel():
    def __init__(self, state_size, action_size):
        """
        create sequential model
        """
        pass

    def fit(self, state, target_f, epochs=1, verbose=0, callbacks=ClearMemory()):
        return self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=ClearMemory())

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_weights(self, name):
        self.model.load_weights(name)

    def predict(self, state):
        return self.model.predict(state)


class DQNAgent():
    def __init__(self, state_size, action_size, epsilon):
        self.output_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/tw2/" 
                
        # Parameters
        self.gamma = 0.95
        self.epsilon_decay = 0.94
        self.epsilon_min = 0.05
        self.replay_memory_size = 150
        self.batch_size = 64
        self.update_target_every = 10

        self.state_size = state_size
        self.action_size = action_size
        self.max_str = '00000'        
        self.histories = []
        self.epsilon = epsilon
        self.memory = deque(maxlen=self.replay_memory_size)

        # create neural network model
        self.model = self.create_model()

        self.target_model =self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.model.summary()

        self.target_update_counter = 0

    def create_model(self):
        """
        fully_connected_model
        """
        model = Sequential()
        
        # Layers
        model.add(Dense(64, input_dim = self.state_size, activation='relu'))        
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))        
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr = 0.001))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        # print(f'act values: {act_values}')
        return np.argmax(act_values[0])        

    def replay(self):
        if len(self.memory) <self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)

        # current states
        states = np.array([transition[0][0] for transition in minibatch])
        qs_list = self.model.predict(states)

        # future states
        next_states = np.array([transition[3][0] for transition in minibatch])
        future_qs_list = self.target_model.predict(next_states)

        X = []
        y = []

        for index,(state, action, reward, next_state, done) in enumerate(minibatch):
            new_q = reward
            if not done:
                new_q = reward + self.gamma * np.max(future_qs_list[index])

            # update Q value for given state
            qs = qs_list[index]
            qs[action] = new_q

            # append to training data
            X.append(state)
            y.append(qs)

        self.history = self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, shuffle=False, epochs=1, verbose=0, callbacks=ClearMemory())
        self.histories.append(self.history.history['loss'][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        
        self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save_weights(self, name):
        self.target_model.save_weights(name)

    def load_last_model(self):
        """
        load lastest model
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        files = os.listdir(self.output_dir)
        
        if not files:
            return 
        
        weights = []
        for fn in files:
            weights.append(int(fn[8:13]))

        max_num = max(weights)
        self.max_str = f'{max_num:05d}'
        filename = "weights_"+self.max_str+".hdf5"
        self.model.load_weights(self.output_dir + filename)        
        print("[Agent] {} file has loaded.".format(filename))


class RL():
    def __init__(self):

        self.n_episodes = 10_000
        self.n_steps = 10_000

        self.env = gym.make('LocalPlannerWorld-v4')
        self.print_env()

        self.feedback = Feedback()
        print(f'cumulated rewards: {self.latest_feedback()["cumulated_rewards"]}')
        # self.draw_cumulative_rewards(self.latest_feedback()["cumulated_rewards"])
        # self.draw_loss(self.latest_feedback()["histories"])
        epsilon = self.latest_feedback()["epsilon"]
        print(f'epsilon: {epsilon}')
        
        self.agent = DQNAgent(self.state_size(), self.action_size(), epsilon)
        self.agent.load_last_model()
        
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
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.plot(moving_average(data, 50))
        plt.show()

    def draw_loss(self, data):
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.plot(moving_average(data, 500))
        plt.show()

    def learning_phase(self):
        done = False

        cumulated_rewards = []
        self.goal_reached_count = 0
        self.cumulative_goal_reached_count = 0

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size()])
            cumulated_reward = 0.0
            
            for time in range(self.n_steps):
                # env.render()
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size()])
                cumulated_reward+= reward
                
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                
                print(f"action: {action}")
                print(f'episode: {time}/{self.n_steps}')
                
                if done:
                    if self.env.is_goal_reached:
                        self.goal_reached_count+=1
                    print(f'cumulated rewards: {self.latest_feedback()["cumulated_rewards"][-200:]}')
                    print(f'histores rewards: {self.latest_feedback()["histories"][-200:]}')
                    rospy.logwarn("********************************************")
                    print("episode: {}/{}, score: {}, e:{:2} goal reached: {}".format(e, self.n_episodes, time, self.agent.epsilon, self.cumulative_goal_reached_count))
                    rospy.logwarn("********************************************")
                    break
                        
            cumulated_rewards.append(cumulated_reward)
            
            # replay agent
            self.agent.replay()

            if e % 10 == 0:
                self.agent.save_weights(self.agent.output_dir+ "weights_"+"{:05d}".format(int(self.agent.max_str)+e) + ".hdf5")

                self.cumulative_goal_reached_count+= self.goal_reached_count
                params={
                    "cumulated_rewards": cumulated_rewards,
                    "epsilon": self.agent.epsilon,
                    "goal_reached_count": self.goal_reached_count,
                    'histories': self.agent.histories
                }
                self.feedback.save(params)
                cumulated_rewards = []
                self.goal_reached_count = 0
                self.agent.histories = []
        self.env.close()

if __name__ == '__main__':
    rospy.init_node('DQN_Agent', anonymous=True, log_level=rospy.WARN)
    rl = RL()
    rl.learning_phase()