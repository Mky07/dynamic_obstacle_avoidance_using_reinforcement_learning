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
from utils import moving_average

import pickle
import matplotlib.pyplot as plt

import gc

class Feedback():
    def __init__(self):
        
        self.parent_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/models/"
        self.filename = "m8.pkl"
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
        # Parameters
        self.learning_rate = 0.0001

        self.state_size = state_size
        self.action_size = action_size

        # self.model = self.fully_connected_model()
        self.model = self.alexnet_cnn_model()

    def fully_connected_model(self):
        model = Sequential()
        
        # input layer
        model.add(Dense(64, input_dim = self.state_size, activation='relu'))
        
        #hidden layer 1
        model.add(Dense(128, activation='relu'))
        
        #hidden layer 2
        model.add(Dense(32, activation='relu'))
        
        #output layer
        model.add(Dense(self.action_size, activation='softmax'))

        model.compile(loss="mse", optimizer=Adam(learning_rate = self.learning_rate))

        return model

    def cnn_model(self):
        # define two sets of inputs
        scan_input_size = self.state_size - 2
        scan_inputs = Input(shape=(scan_input_size,1))
        other_inputs = Input(shape=(self.state_size - scan_input_size,))
        
        # the first branch operates on the first input
        x = Conv1D(filters=32, kernel_size=3, padding= "same", activation='relu')(scan_inputs)
        x = Conv1D(filters=64, kernel_size=3, padding= "same", activation='relu')(x)
        x = MaxPooling1D(pool_size=(3,), padding="same")(x)
        x = Conv1D(filters=64, kernel_size=3, padding= "same", activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Conv1D(filters=128, kernel_size=3, padding= "same", activation='relu')(x)
        x = MaxPooling1D(pool_size=(3,), padding="same")(x)
        x = Flatten()(x)
        x = Model(inputs=scan_inputs, outputs=x)
        
        # the second branch opreates on the second input

        ####################################
        ##### Spatial embedding ##
        ####################################
        y = Dense(16, activation="relu")(other_inputs)
        # y = Dense(16, activation="relu")(y)
        y = Model(inputs=other_inputs, outputs=y)
        
        # combine the output of the two branches
        combined = concatenate([x.output, y.output])

        # apply a FC layer and then a regression prediction on the
        
        # combined outputs
        z = Dense(128, activation="relu")(combined)
        z = Dense(64, activation="relu")(z)
        z = Dense(self.action_size, activation="linear")(z)
        
        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)

        # compile model
        model.compile(loss="mse", optimizer=Adam(learning_rate = self.learning_rate))

        return model

    def alexnet_cnn_model(self):
        #https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/
        # define two sets of inputs
        scan_input_size = self.state_size - 2
        scan_inputs = Input(shape=(scan_input_size,1))
        other_inputs = Input(shape=(self.state_size - scan_input_size,))
        
        # the first branch operates on the first input
        x = Conv1D(filters=32, kernel_size=11, strides=4, padding= "same", activation='relu')(scan_inputs)
        x = MaxPooling1D(pool_size=(3,), strides=2, padding="same")(x)
        x = Conv1D(filters=64, kernel_size=5, strides=1, padding= "same", activation='relu')(x)
        x = MaxPooling1D(pool_size=(3,), strides=2, padding="same")(x)
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding= "same", activation='relu')(x)
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding= "same", activation='relu')(x)
        x = Conv1D(filters=16, kernel_size=3, strides=1, padding= "same", activation='relu')(x)
        x = MaxPooling1D(pool_size=(3,), strides=2, padding="same")(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Model(inputs=scan_inputs, outputs=x)

        # the second branch opreates on the second input
        y = Dense(2, activation="relu")(other_inputs)
        y = Dense(16, activation="relu")(y)
        y = Dense(32, activation="relu")(y)
        y = Model(inputs=other_inputs, outputs=y)
        
        # combine the output of the two branches
        combined = concatenate([x.output, y.output])

        # apply a FC layer and then a regression prediction on the
        
        # combined outputs
        z = Dense(64, activation="relu")(combined)
        # z = Dropout(0.5)(z)
        z = Dense(64, activation="relu")(z)
        #https://ai.stackexchange.com/questions/34589/using-softmax-non-linear-vs-linear-activation-function-in-deep-reinforceme#:~:text=The%20normal%20use%20case%20for,are%20estimates%20for%20some%20measurement.
        # z = Dense(self.action_size, activation="linear")(z)
        z = Dense(self.action_size, activation="softmax")(z)
        
        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)

        # compile model categorical_hinge
        model.compile(loss="categorical_hinge", optimizer=Adam(learning_rate = self.learning_rate))
        # model.compile(loss="mse", optimizer=Adam(learning_rate = self.learning_rate))

        return model


    def summary(self):
        return self.model.summary()

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_weights(self, name):
        self.model.load_weights(name)

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target_f, epochs=1, verbose=0, callbacks=ClearMemory()):
        return self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=ClearMemory())

class DQNAgent():
    def __init__(self, state_size, action_size, epsilon):
        self.state_size = state_size
        self.action_size = action_size
                
        # Q-Learning parameters
        self.gamma = 0.90
        self.epsilon_decay = 0.992
        self.epsilon_min = 0.05
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon

        # other parameters
        self.batch_size = 32
        self.max_str = '00000'
        self.histories = []

        self.output_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/w8/" 

        if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        # create neural network model
        self.model = NNModel(self.state_size, self.action_size)
        self.model.summary()

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

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.history = self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=ClearMemory())
            self.histories.append(self.history)

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
        self.n_steps = 10_000

        self.env = gym.make('LocalPlannerWorld-v4')
        self.print_env()

        self.feedback = Feedback()
        print(f'cumulated rewards: {self.latest_feedback()["cumulated_rewards"]}')
        self.draw_cumulative_rewards(self.latest_feedback()["cumulated_rewards"])

        # history = self.latest_feedback()['histories'][-10]
        # self.draw_loss(history)

        epsilon = self.latest_feedback()["epsilon"]
        print(f'epsilon: {epsilon}')
        
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
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.plot(moving_average(data, 500))
        plt.show()

    def draw_loss(self, history):
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def learning_phase(self):
        done = False

        cumulated_rewards = []
        self.goal_reached_count = 0
        self.is_dist_exceed_count = 0
        self.is_angle_exceed_count = 0
        self.cumulative_goal_reached_count = 0

        # rush_count = 0
        # rush_count_limit = 2
        for e in range(self.n_episodes):
            state = self.env.reset()
            # state = np.reshape(state, [1, self.state_size()])
            scan_states = state[2:]
            other_states = state[:2]
            state1 = np.reshape(scan_states, [1, len(scan_states,)])
            state2 = np.reshape(other_states, [1, len(other_states,)])
            state = [state1, state2]
            cumulated_reward = 0.0

            for time in range(self.n_steps):
                # env.render()
                action = self.agent.act(state)
                print(f"action: {action}")
                # if rush_count<rush_count_limit:
                #     action = 28
                next_state, reward, done, _ = self.env.step(action)
                cumulated_reward+= reward

                next_scan_states = next_state[2:]
                next_other_states = next_state[:2]
                next_state1 = np.reshape(next_scan_states, [1, len(next_scan_states,)])
                next_state2 = np.reshape(next_other_states, [1, len(next_other_states,)])
                next_state = [next_state1, next_state2]

                # next_state = np.reshape(next_state, [1, self.state_size()])
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    if self.env.is_goal_reached:
                        self.goal_reached_count+=1
                    if self.env.is_dist_exceed:
                        self.is_dist_exceed_count+=1
                    if self.env.is_angle_exceed:
                        self.is_angle_exceed_count+=1

                    print(f'cumulated rewards: {self.latest_feedback()["cumulated_rewards"][-200:]}')
                    rospy.logwarn("********************************************")
                    print("episode: {}/{}, score: {}, e:{:2} goal reached: {} angle_exceed:{} dist_exceed:{}".format(e, self.n_episodes, time, self.agent.epsilon, self.cumulative_goal_reached_count, self.is_angle_exceed_count, self.is_dist_exceed_count))
                    rospy.logwarn("********************************************")
                    break
                        
            cumulated_rewards.append(cumulated_reward)
            
            # replay agent
            self.agent.replay()

            if e % 10 == 0:
                self.agent.save(self.agent.output_dir + "weights_"+"{:05d}".format(int(self.agent.max_str)+e) + ".hdf5")

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