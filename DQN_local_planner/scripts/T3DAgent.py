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
import json
from enum import Enum

# FILES
# import action_noise
# from action_noise import OUActionNoise

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
            
class DoneReason(Enum):
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION_DETECTED = "COLLISION_DETECTED"
    DIST_EXCEEDED = "DIST_EXCEEDED"
    ANGLE_EXEEDED = "ANGLE_EXEEDED"
    N_STEPS_DONE = "N_STEPS_DONE"

class T3DAgent:
    def __init__(self, env_name, parent_dir, buffer_capacity=50_000, batch_size=64, critic_lr=0.001, actor_lr = 0.0001):
        self.parent_dir = parent_dir

        ########## environment parameters ######################
        self.env = gym.make(env_name)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.upper_bound = self.env.action_space.high
        self.lower_bound = self.env.action_space.low
        print("Size of State Space ->  {}".format(self.num_states))
        print("Size of Action Space ->  {}".format(self.num_actions))
        print("Max Value of Action ->  {}".format(self.upper_bound))
        print("Min Value of Action ->  {}".format(self.lower_bound))
        ########################################################
        self.buffer_capacity = buffer_capacity
        
        # load config
        self.config=self.load_config()

        self.draw(self.config["cumulative_rewards"], 50)
        self.draw(self.config["critic_loss"],100,"Step" ,"Critic Loss")
        self.draw(self.config["actor_loss"],100,"Step" ,"Actor Loss")

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = self.config["state_buffer"]
        self.action_buffer = self.config["action_buffer"]
        self.reward_buffer = self.config["reward_buffer"]
        self.next_state_buffer = self.config["next_state_buffer"]

        # Its tells us num of times record() was called.
        self.buffer_counter = self.config["buffer_counter"]

        # Noise Object
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Model parameters
        self.batch_size = batch_size
        self.gamma = 0.99
        self.tau = 0.005
        self.update_frequency = 20
        self.update_counter = 0

        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.critic_model = self.get_critic()
        self.target_critic = self.get_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.critic_model2 = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.critic_optimizer2 = tf.keras.optimizers.Adam(critic_lr)

        self.load_weights()

    def load_weights(self):
        weight_filename = "weights_"+"{:05d}".format(self.config["ep"]) + ".hdf5"
        if not os.path.exists(self.parent_dir+"actor/"):
            os.makedirs(self.parent_dir+"actor/")
        if not os.path.exists(self.parent_dir+"critic/"):
            os.makedirs(self.parent_dir+"critic/")
        
        if os.listdir(self.parent_dir+"actor/"):
            lf = self.parent_dir+"actor/" +weight_filename
            self.actor_model.load_weights(lf)
            print(f"actor model: {lf} loaded")
            self.target_actor.set_weights(self.actor_model.get_weights())

        if os.listdir(self.parent_dir+"critic/"):
            lf = self.parent_dir+"critic/" +weight_filename
            print(f"critic model: {lf} loaded")
            self.critic_model.load_weights(lf)
            self.target_critic.set_weights(self.critic_model.get_weights())
    
    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(128, activation="relu")(out)
        out = layers.Dense(32, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)

        outputs = self.lower_bound + 0.5*(outputs+1.0) * (self.upper_bound-self.lower_bound)
        # outputs_0 = self.lower_bound[0] + (outputs[:, 0] + 1.0) / 2.0 * (self.upper_bound[0] - self.lower_bound[0])
        # outputs_1 = self.lower_bound[1] + (outputs[:, 1] + 1.0) / 2.0 * (self.upper_bound[1] - self.lower_bound[1])
        # outputs = tf.stack([outputs_0, outputs_1], axis=1)

        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        action_input = layers.Input(shape=(self.num_actions))
        concat = layers.Concatenate()([state_input, action_input])

        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(128, activation="relu")(out)
        out = layers.Dense(32, activation="relu")(out)
        outputs = layers.Dense(1, activation="linear")(out)
        
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        # print(f"sampled action:{sampled_actions}")
        return self.clipped_noisy_action(sampled_actions)

    def clipped_noisy_action(self, action):
        action = action.numpy() + self.ou_noise()
        legal_action = np.clip(action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)][0]

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    # @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # state_batch, action_batch, reward_batch, next_state_batch = self.sample()
        critic_loss, actor_loss=self.update(state_batch, action_batch, reward_batch, next_state_batch)
        self.config["critic_loss"].append(critic_loss.numpy())
        if actor_loss != None:
            self.config["actor_loss"].append(actor_loss.numpy())
        # print(f"cl: {critic_loss.numpy()} al:{actor_loss.numpy()}")

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def draw(self, data, ma=100, xlabel="Episode", ylabel="Cumulative Reward"):
        if len(data)>ma:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.plot(moving_average(data, ma))
            plt.show()

    # @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        self.update_counter+=1
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            target_actions = self.clipped_noisy_action(target_actions) # T3D Özelliği. Add noise to target action. Then clip it. 
            q1 = self.target_critic([next_state_batch, target_actions], training=True)
            q2 = self.target_critic2([next_state_batch, target_actions], training=True)
            
            q = np.concatenate((q1.numpy(), q2.numpy()), axis=1)
            q = np.min(q, axis=1)
            q = np.reshape(q, (len(q), 1)) # T3D Özelliği

            y = reward_batch + self.gamma * q
            critic_value1 = self.critic_model([state_batch, action_batch], training=True)
            critic_value2 = self.critic_model2([state_batch, action_batch], training=True)
            
            critic_loss1 = tf.math.reduce_mean(tf.math.square(y - critic_value1))
            critic_loss2 = tf.math.reduce_mean(tf.math.square(y - critic_value2))

        critic_grad = tape.gradient(critic_loss1, self.critic_model.trainable_variables)
        critic_grad2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)

        del tape

        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
        self.critic_optimizer2.apply_gradients(zip(critic_grad2, self.critic_model2.trainable_variables))

        actor_loss = None
        if self.update_counter % self.update_frequency == 0: # T3D Özelliği
            with tf.GradientTape() as tape:
                actions = self.actor_model(state_batch, training=True)
                critic_value = self.critic_model([state_batch, actions], training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        return critic_loss1, actor_loss

    def train(self, num_episodes=10_000):
        self.config["num_episodes"] = num_episodes

        for ep in range(self.config["ep"]+1, self.config["num_episodes"]):
            cumulative_reward = 0
            step=0
            prev_state = self.env.reset()
            
            while True:
                step+=1

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state)
                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)

                self.record((prev_state, action, reward, state))
                
                cumulative_reward += reward
                self.learn()

                self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
                prev_state = state
                print(f"episode: {ep}/{num_episodes}, score: {step},buffer_counter:{self.config['buffer_counter']}")
                print(f"cumulative rewards: {self.config['cumulative_rewards'][-200:]}")
                self.print_done_counts()
                # End this episode when `done` is True
                if done:
                    rospy.logwarn("********************************************")
                    print(f"cumulative rewards: {self.config['cumulative_rewards'][-200:]}")
                    print(f"episode: {ep}/{num_episodes}, score: {step}, total reward: {np.mean(self.config['cumulative_rewards'][-40:])} buffer_counter:{self.config['buffer_counter']}")
                    self.print_done_counts()
                    rospy.logwarn("********************************************")

                    self.save_config(cumulative_reward, ep, step)
                    self.save_models()
                    break

    def save_config(self, cumulative_reward, ep, step):
        if self.env.is_goal_reached:
            self.config["done_reasons"].append(DoneReason.GOAL_REACHED.value)
        elif self.env.is_dist_exceed:
            self.config["done_reasons"].append(DoneReason.DIST_EXCEEDED.value)
        elif self.env.is_angle_exceed:
            self.config["done_reasons"].append(DoneReason.ANGLE_EXEEDED.value)
        elif self.env.is_collision_detected:
            self.config["done_reasons"].append(DoneReason.COLLISION_DETECTED.value)
        elif self.env.nsteps_done:
            self.config["done_reasons"].append(DoneReason.N_STEPS_DONE.value)
        else:
            print("Unexpected Done Reason.")

        self.config["cumulative_rewards"].append(cumulative_reward)
        self.config["ep"] = ep
        self.config["nstep"] = step
        self.config["buffer_counter"] = self.buffer_counter
        self.config["state_buffer"] = self.state_buffer
        self.config["action_buffer"] = self.action_buffer
        self.config["reward_buffer"] = self.reward_buffer
        self.config["next_state_buffer"] = self.next_state_buffer

        self._save_config()
        
    def save_models(self):
        weight_filename = "weights_"+"{:05d}".format(self.config["ep"]) + ".hdf5"
        self.actor_model.save_weights(self.parent_dir + "actor/" + weight_filename)
        self.critic_model.save_weights(self.parent_dir + "critic/" + weight_filename)


    def _save_config(self):
        filename = "config.pkl"
        fn = self.parent_dir+"config/" + filename
        with open(fn, 'wb') as f:
            if not os.path.exists(fn):
                pass
            else:
                pickle.dump(self.config, f)

    def load_config(self):
        filename = "config.pkl"
        fn = self.parent_dir+"config/" + filename
        config = {}

        if not os.path.exists(self.parent_dir+"config/"):
            os.makedirs(self.parent_dir+"config/")

        if not os.path.exists(fn):
            with open(fn, 'wb') as f:
                config = {
                "done_reasons": [],
                "cumulative_rewards": [],
                "buffer_counter": 0,
                "state_buffer":np.zeros((self.buffer_capacity, self.num_states)),
                "action_buffer":np.zeros((self.buffer_capacity, self.num_actions)),
                "reward_buffer":np.zeros((self.buffer_capacity, 1)),
                "next_state_buffer":np.zeros((self.buffer_capacity, self.num_states)),
                "critic_loss": [],
                "actor_loss": [],
                "ep": 0,
                "num_episodes": 10_000,
                "nstep":[]
                }
                pickle.dump(config, f)

        with open(fn, 'rb') as f:
            config = pickle.load(f)

        return config

    def print_done_counts(self):
        goal_reached_count = self.config["done_reasons"].count(DoneReason.GOAL_REACHED.value)
        collision_detected_count = self.config["done_reasons"].count(DoneReason.COLLISION_DETECTED.value)
        dist_exceeded_count = self.config["done_reasons"].count(DoneReason.DIST_EXCEEDED.value)
        angle_exceeded_count = self.config["done_reasons"].count(DoneReason.ANGLE_EXEEDED.value)
        nsteps_done_count = self.config["done_reasons"].count(DoneReason.N_STEPS_DONE.value)
        
        print(f"GOAL_REACHED:{goal_reached_count}, COLLISION_DETECTED:{collision_detected_count}, DIST_EXCEEDED:{dist_exceeded_count}, ANGLE_EXEEDED:{angle_exceeded_count}, N_STEPS_DONE:{nsteps_done_count}")

if __name__ == '__main__':
    rospy.init_node('T3D_Agent', anonymous=True, log_level=rospy.WARN)
    env_name = "LocalPlannerWorld-v4"

    ######################### CONFIG #########################################
    parent_folder = "robot_t3d"
    critic_lr = 0.001
    actor_lr = 0.0001
    batch_size = 64
    buffer_capacity = 50_000

    ##########################################################################
    current_file_location = os.path.abspath(__file__)
    main_dir = os.path.dirname(current_file_location)
    main_dir = os.path.dirname(main_dir)
    parent_dir = main_dir+"/"+parent_folder+"/"
    ##########################################################################

    agent = T3DAgent(env_name, parent_dir,buffer_capacity=buffer_capacity, batch_size=batch_size, critic_lr=critic_lr, actor_lr=actor_lr)
    agent.train(10_000)
    agent.draw(agent.config["cumulative_rewards"], 40)
    agent.draw(agent.config["critic_loss"],100,"Step" ,"Critic Loss")
    agent.draw(agent.config["actor_loss"],100,"Step" ,"Actor Loss")