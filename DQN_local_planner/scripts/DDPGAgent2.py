#!/usr/bin/env python3

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import gym
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as k
import DDPG_env
import rospy
import rospkg
from utils import moving_average, min_max_normalize

import pickle
import matplotlib.pyplot as plt

import gc

class Feedback():
    def __init__(self):
        
        self.parent_dir = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/models/"
        self.filename = "ddpg6.pkl"
        self.file_path = self.parent_dir + self.filename
                
        # assign_params if not created
        self.params={
            "cumulated_rewards": [],
            "goal_reached_count": 0,
            'memory': deque(maxlen=1),
            'memory_len': 1000,
            # "actor_loss_list": [],
            # "critic_loss_list": [],
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
                self.params['goal_reached_count']+= params['goal_reached_count']
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

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class ReplayBuffer:
    def __init__(self, memory, max_size):
        self.max_size = max_size
        self.buffer = deque(memory, maxlen=self.max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

class Actor:
    def __init__(self, state_dim, action_dim, learning_rate, lower_bound, upper_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.learning_rate = learning_rate
        self.optimizer = Adam(self.learning_rate)
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=[self.state_dim])
        h1 = Dense(64, activation='relu')(input_layer)
        h2 = Dense(32, activation='relu')(h1)
        output_layer = Dense(self.action_dim, activation=self.custom_activation)(h2)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def train(self, gradients):        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def custom_activation(self, x):
        outputs_0 = self.lower_bound[0] + (tf.nn.tanh(x[:, 0]) + 1.0) / 2.0 * (self.upper_bound[0] - self.lower_bound[0])
        outputs_1 = self.lower_bound[1] + (tf.nn.tanh(x[:, 1]) + 1.0) / 2.0 * (self.upper_bound[1] - self.lower_bound[1])
        outputs = tf.stack([outputs_0, outputs_1], axis=1)
        return outputs

class Critic:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.learning_rate)
        self.critic_loss = None
        self.model = self.build_model()

    def build_model(self):
        state_input_layer = Input(shape=[self.state_dim])
        action_input_layer = Input(shape=[self.action_dim])
        concatenated = Concatenate()([state_input_layer, action_input_layer])
        h1 = Dense(64, activation='relu')(concatenated)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(32, activation='relu')(h2)
        output_layer = Dense(1, activation='linear')(h3)
        model = Model(inputs=[state_input_layer, action_input_layer],
                      outputs=output_layer)

        model.compile(loss="mse", optimizer=self.optimizer)
        return model

    def loss(self, states, actions, q_targets):
        q_values = self.model([states, actions])
        return tf.reduce_mean(tf.square(q_targets - q_values))

    def train(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            self.critic_loss = self.loss(states, actions, q_targets)
        gradients = tape.gradient(self.critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def predict(self, states, actions):
        return self.model([states, actions])

class DDPGAgent:
    def __init__(self, env, buffer_size=1000000, batch_size=64):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.001
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = batch_size
        self.min_batch_size = 1_000
        
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        self.feedback = Feedback()
        self.draw_cumulative_rewards(self.feedback.params["cumulated_rewards"])

        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        print("Size of State Space ->  {}".format(self.state_dim))
        print(f"env.action_space: {env.action_space}")
        print("Size of Action Space ->  {}".format(self.action_dim))
        print("Max Value of Action ->  {}".format(self.upper_bound))
        print("Min Value of Action ->  {}".format(self.lower_bound))

        self.actor = Actor(self.state_dim, self.action_dim, self.actor_learning_rate, self.lower_bound, self.upper_bound)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.actor_learning_rate, self.lower_bound, self.upper_bound)
        self.target_actor.model.set_weights(self.actor.model.get_weights())

        self.critic = Critic(self.state_dim, self.action_dim, self.critic_learning_rate)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.critic_learning_rate)
        self.target_critic.model.set_weights(self.critic.model.get_weights())

        ################################# MODEL LOAD WEIGHTS ##############################
        folder_name = "ddpg6"
        self.max_str = "00000"
        base_weight_folder = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/DQN_local_planner/"+folder_name+"/" 
        self.actor_folder = base_weight_folder + "actor/"
        self.critic_folder = base_weight_folder + "critic/"

        if not os.path.exists(self.actor_folder):
                os.makedirs(self.actor_folder)

        if not os.path.exists(self.critic_folder):
                os.makedirs(self.critic_folder)
        
        # # load weights
        aw = self.load_model_weights(self.actor_folder)
        print(f"aw: a{aw}")
        if aw !=None:
            self.actor.model.load_weights(aw)
        
        cw = self.load_model_weights(self.critic_folder)
        if cw !=None:
            self.critic.model.load_weights(cw)
        ##################################################################################

        self.replay_buffer = ReplayBuffer(self.feedback.params["memory"],max_size=buffer_size)

    def draw_cumulative_rewards(self, data):
        size = 500
        if len(data)>size:
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.plot(moving_average(data, size))
            plt.show()

    def load_model_weights(self, out_dir):
        dir_list = os.listdir(out_dir)
        
        if dir_list:
            weights = []
            for fn in dir_list:
                weights.append(int(fn[8:13]))

            max_num = max(weights)
            self.max_str = f'{max_num:05d}'
            filename = "weights_"+self.max_str+".hdf5"
            print("[Agent] {} file has loaded.".format(out_dir + filename))

            return out_dir+filename
        return None

    def select_action(self, state, noise):
        acts = self.actor.predict(np.reshape(state, [1, self.state_dim]))[0]
        print(f"action actor:{acts}")

        # acts = np.clip(acts + noise, -1, 1)
        act_0 = np.clip(acts[0] + noise(), 0.1, 1)
        act_1 = np.clip(acts[1] + noise(), -0.4, 0.4)
        # acts = (acts +1.0)/2.0
        # act_0 = self.lower_bound[0] + acts[0] * (self.upper_bound[0] - self.lower_bound[0])
        # act_1 = self.lower_bound[1] + acts[1] * (self.upper_bound[1] - self.lower_bound[1])
        acts = np.array([act_0[0], act_1[0]])
        return acts
    
    def update_target(self, model, target_model):
        model_weights = model.get_weights()
        target_model_weights = target_model.get_weights()
        new_weights = [self.tau * model_weight + (1 - self.tau) * target_model_weight
                    for model_weight, target_model_weight in zip(model_weights, target_model_weights)]
        return new_weights

    def train(self, num_episodes=1000, max_steps=1000):
        # actor_loss_list = []
        # critic_loss_list = []

        for ep in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            goal_reached_count = 0
            is_dist_exceed_count = 0
            is_angle_exceed_count = 0
            cumulative_goal_reached_count = 0
            cumulated_rewards = []

            for step in range(max_steps):
                action = self.select_action(state, self.ou_noise)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add((state, action, reward, next_state, done))

                if len(self.replay_buffer.buffer) > self.min_batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    q_target = rewards + self.gamma * self.target_critic.model([next_states, self.target_actor.model(next_states)]) * (1 - dones)
                    self.critic.train(states, actions, q_target)

                    with tf.GradientTape() as tape:
                        actor_q_values = self.critic.model([states, self.actor.model(states)])
                        actor_loss = -tf.reduce_mean(actor_q_values)
                    gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
                    self.actor.train(gradients)

                    self.target_critic.model.set_weights(self.update_target(self.critic.model, self.target_critic.model))
                    self.target_actor.model.set_weights(self.update_target(self.actor.model, self.target_actor.model))

                print(f'step: {step}/{1000} episode: {ep}/{num_episodes} goal_reached:{cumulative_goal_reached_count} memory size: {len(self.replay_buffer.buffer)}')

                total_reward += reward
                state = next_state
                if done:
                    if env.is_goal_reached:
                        goal_reached_count+=1
                    if env.is_dist_exceed:
                        is_dist_exceed_count+=1
                    if env.is_angle_exceed:
                        is_angle_exceed_count+=1
                    print(f'cumulated rewards: {self.feedback.params["cumulated_rewards"][-200:]}')
                    # print(f'actor_loss_list: {actor_loss_list[-200:]}')
                    # print(f'critic_loss_list: {critic_loss_list[-200:]}')

                    rospy.logwarn("********************************************")
                    print("episode: {}/{}, score: {}, goal reached: {} angle_exceed:{} dist_exceed:{}".format(ep, num_episodes, step, cumulative_goal_reached_count, is_angle_exceed_count, is_dist_exceed_count))
                    rospy.logwarn("********************************************")

                    break
            print("Episode:", ep + 1, "Total Reward:", total_reward)

            # save models
            weight_filename = "weights_"+"{:05d}".format(int(self.max_str)+ep) + ".hdf5"

            self.actor.model.save_weights(self.actor_folder+weight_filename)
            self.critic.model.save_weights(self.critic_folder+weight_filename)

            cumulated_rewards.append(total_reward)
            # actor_loss_list.append(actor_loss)
            # critic_loss_list.append(self.critic.critic_loss)

            cumulative_goal_reached_count+= goal_reached_count
            params={
                "cumulated_rewards": cumulated_rewards,
                # "actor_loss_list": actor_loss_list,
                # "critic_loss_list": critic_loss_list,
                "goal_reached_count": goal_reached_count,
                'memory': self.replay_buffer.buffer,
                'memory_len': self.replay_buffer.max_size}
            
            self.feedback.save(params)
            cumulated_rewards = []
            goal_reached_count = 0

# Kullanacağımız ortamı oluşturalım
rospy.init_node('DDPG_Agent', anonymous=True, log_level=rospy.WARN)
env = gym.make("LocalPlannerWorld-v4")

# DDPG ajanını oluşturup eğitelim
agent = DDPGAgent(env, buffer_size=50_000, batch_size=64)
agent.train(num_episodes=10_000)