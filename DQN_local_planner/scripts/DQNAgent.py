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
import environment
import rospy
import rospkg
from utils import moving_average, min_max_normalize, tt

import pickle
import matplotlib.pyplot as plt

import gc
from enum import Enum

class DoneReason(Enum):
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION_DETECTED = "COLLISION_DETECTED"
    DIST_EXCEEDED = "DIST_EXCEEDED"
    ANGLE_EXEEDED = "ANGLE_EXEEDED"
    N_STEPS_DONE = "N_STEPS_DONE"

class Feedback():
    def __init__(self):
        
        self.parent_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/models/"
        self.filename = "robot_dqn.pkl"
        self.file_path = self.parent_dir + self.filename
                
        # assign_params if not created
        self.params={
            "cumulated_rewards": [],
            "epsilon": 0.99,
            "done_reasons": [],
            'histories': [],
            'memory': deque(maxlen=1),
            'memory_len': 1000}
            
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
                self.params["done_reasons"]+= params["done_reasons"]
                self.params["histories"]+= params["histories"]
                self.params["epsilon"] = params["epsilon"]
                self.params['memory'] = deque(params['memory'], maxlen=params['memory_len'])
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
        # Parameters
        self.learning_rate = 0.001

        self.state_size = state_size
        self.action_size = action_size

        self.model = self.fully_connected_model()
        # self.model = self.alexnet_cnn_model()

    def fully_connected_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))        
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr = self.learning_rate))

        return model

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def summary(self):
        return self.model.summary()

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_weights(self, name):
        self.model.load_weights(name)

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, X, y, batch_size=32):
        return self.model.fit(X, y, batch_size=batch_size, shuffle=False, epochs=1, callbacks=ClearMemory())

class DQNAgent():
    def __init__(self, state_size, action_size, epsilon, memory):
        self.state_size = state_size
        self.action_size = action_size
                
        # Q-Learning parameters
        self.gamma = 0.95 
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.min_replay_memory = 1_000
        self.max_replay_memory = 50_000
        self.memory = deque(memory, maxlen=self.max_replay_memory)
        self.epsilon = epsilon
        self.update_target_every = 1
        self.target_update_counter = 0

        # other parameters
        self.batch_size = 64
        self.max_str = '00000'
        self.histories = []
        self.is_model_fit = False

        self.output_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/robot_dqn/" 

        if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # create neural network model
        self.model = NNModel(self.state_size, self.action_size)

        self.target_model = NNModel(self.state_size, self.action_size)
        self.target_model.set_weights(self.model.get_weights())

        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])        

    def replay(self):
        if len(self.memory) <self.min_replay_memory:
            return
        self.is_model_fit = True

        minibatch = random.sample(self.memory, self.batch_size)

        state_list = []
        next_state_list = []
        for index,(state, action, reward, next_state, done) in enumerate(minibatch):
            state_list.append(state[0])
            next_state_list.append(next_state[0])

        qs_list = self.model.predict(np.array(state_list))
        qs_next_list = self.target_model.predict(np.array(next_state_list))
        
        for index,(state, action, reward, next_state, done) in enumerate(minibatch):
            new_q = reward
            if not done:
                new_q = reward + self.gamma * np.max(qs_next_list[index])
 
            qs_list[index][action] = new_q

        print("****************************************************************")
        self.history = self.model.fit(np.array(state_list), np.array(qs_list), self.batch_size)
        print("****************************************************************")

        self.histories.append(self.history.history['loss'][0])
                
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        # self.model.save_weights(name)
        self.target_model.save_weights(name)

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
        self.n_steps = 10_000

        self.env = gym.make('LocalPlannerWorld-v4')
        self.print_env()

        self.feedback = Feedback()
        print(f'cumulated rewards: {self.latest_feedback()["cumulated_rewards"]}')
        self.print_done_counts()
        self.draw_cumulative_rewards(self.latest_feedback()["cumulated_rewards"])
        self.draw_loss(self.latest_feedback()["histories"])
        
        done_reasons = self.latest_feedback()["done_reasons"]
        self.draw(tt(done_reasons, 'GOAL_REACHED'), 500, 'Episode', 'Goal Reached')
        self.draw(tt(done_reasons, 'COLLISION_DETECTED'), 500, 'Episode', 'Collision Detected')
        self.draw(tt(done_reasons, 'DIST_EXCEEDED'), 500, 'Episode', 'Distance Exceeded')
        self.draw(tt(done_reasons, 'ANGLE_EXEEDED'), 500, 'Episode', 'Angle Exceeded')
 
        epsilon = self.latest_feedback()["epsilon"]
        print(f'epsilon: {epsilon}')
        
        self.agent = DQNAgent(self.state_size(), self.action_size(), epsilon, self.latest_feedback()["memory"])
        self.agent.load_latest_model()



    def draw(self, data, ma=100, xlabel="Episode", ylabel="Cumulative Reward"):
        if len(data)>ma:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.plot(moving_average(data, ma))
            plt.show()

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
        size = 500
        if len(data)>size:
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.plot(moving_average(data, size))
            plt.show()

    def draw_loss(self, data):
        # data = [x for x in data if x <= 5_000]
        size = 5000
        if len(data)>size:
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.plot(moving_average(data, size))
            plt.show()

    def print_done_counts(self):
        goal_reached_count = self.latest_feedback()["done_reasons"].count(DoneReason.GOAL_REACHED.value)
        collision_detected_count = self.latest_feedback()["done_reasons"].count(DoneReason.COLLISION_DETECTED.value)
        dist_exceeded_count = self.latest_feedback()["done_reasons"].count(DoneReason.DIST_EXCEEDED.value)
        angle_exceeded_count = self.latest_feedback()["done_reasons"].count(DoneReason.ANGLE_EXEEDED.value)
        nsteps_done_count = self.latest_feedback()["done_reasons"].count(DoneReason.N_STEPS_DONE.value)
        
        print(f"GOAL_REACHED:{goal_reached_count}, COLLISION_DETECTED:{collision_detected_count}, DIST_EXCEEDED:{dist_exceeded_count}, ANGLE_EXEEDED:{angle_exceeded_count}, N_STEPS_DONE:{nsteps_done_count}")


    def learning_phase(self):
        done = False

        cumulated_rewards = []
        done_reasons = []

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size()])

            cumulated_reward = 0.0

            for time in range(self.n_steps):
                action = self.agent.act(state)
                # print(f"action: {action}")
                next_state, reward, done, _ = self.env.step(action)
                cumulated_reward+= reward

                next_state = np.reshape(next_state, [1, self.state_size()])
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay()

                state = next_state
                print(f'step: {time}/{self.n_steps} episode: {e}/{self.n_episodes} target_update_counter: {self.agent.target_update_counter} memory size: {len(self.agent.memory)}')
                self.print_done_counts()

                if done:
                    self.agent.target_update_counter += 1
            
                    if self.agent.epsilon > self.agent.epsilon_min and self.agent.is_model_fit:
                        self.agent.epsilon *= self.agent.epsilon_decay

                    if self.env.is_goal_reached:
                        done_reasons.append(DoneReason.GOAL_REACHED.value)
                    elif self.env.is_dist_exceed:
                        done_reasons.append(DoneReason.DIST_EXCEEDED.value)
                    elif self.env.is_angle_exceed:
                        done_reasons.append(DoneReason.ANGLE_EXEEDED.value)
                    elif self.env.is_collision_detected:
                        done_reasons.append(DoneReason.COLLISION_DETECTED.value)
                    elif self.env.nsteps_done:
                        done_reasons.append(DoneReason.N_STEPS_DONE.value)
                    else:
                        print("Unexpected Done Reason.")

                    print(f'cumulated rewards: {self.latest_feedback()["cumulated_rewards"][-200:]}')
                    # print(f'histores rewards: {self.latest_feedback()["histories"][-200:]}')
                    self.print_done_counts()
                    rospy.logwarn("********************************************")
                    print("episode: {}/{}, score: {}, e:{:2}".format(e, self.n_episodes, time, self.agent.epsilon))
                    rospy.logwarn("********************************************")
                    break
            
            cumulated_rewards.append(cumulated_reward)
            
            # if e % 10 == 0 and self.agent.is_model_fit:
            if self.agent.is_model_fit:
            # if self.agent.target_update_counter == 0:
                self.agent.save(self.agent.output_dir + "weights_"+"{:05d}".format(int(self.agent.max_str)+e) + ".hdf5")

                params={
                    "cumulated_rewards": cumulated_rewards,
                    "epsilon": self.agent.epsilon,
                    'histories': self.agent.histories,
                    'memory': self.agent.memory,
                    'memory_len': self.agent.max_replay_memory,
                    'done_reasons': done_reasons
                }
                self.feedback.save(params)
                cumulated_rewards = []
                done_reasons = []
                self.agent.histories = []
        self.env.close()

if __name__ == '__main__':
    rospy.init_node('DQN_Agent', anonymous=True, log_level=rospy.WARN)
    rl = RL()
    rl.learning_phase()