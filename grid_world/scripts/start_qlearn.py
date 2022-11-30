#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
import grid_world
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from geometry_msgs.msg import Pose


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == '__main__':

    rospy.init_node('grid_world', anonymous=True, log_level=rospy.WARN)
    filename = "/home/mky/rl_ws/src/openai_examples_projects/grid_world/models/model14.pkl"

    # log_varaiables = defineLogVariables()
    log_values = {
        "qlearn.q": {},
        "current_episode": 0,
        "nepisodes": 0,
        "cumulated_reward": []
    }
    
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            pickle.dump(log_values, f)     

    # Create the Gym environment
    env = gym.make('GridWorld-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtle2_openai_ros_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    # load qlearn valeus from file
    with open(filename, 'rb') as f:
        log_values = pickle.load(f)
        qlearn.q = log_values['qlearn.q']

    last_episode = log_values['current_episode']
    plt.plot(moving_average(log_values['cumulated_reward'], 15))
    plt.show()
    start_time = time.time()
    highest_reward = 0
    
    print("q valeus: {}".format(qlearn.q))
    print("last episode: {}".format(log_values['current_episode']))
    print("cumulated_reward: {}".format(log_values['cumulated_reward']))


    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        print_q = {key : round(qlearn.q[key], 2) for key in qlearn.q}
        # rospy.logerr("*****************************************\nq values:\n {}\n*****************************************".format(print_q))
        print("cumulated reward: {}".format(log_values['cumulated_reward'][-100:]))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.1:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(str( observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            infos ="episode:" + str(log_values['current_episode']) + ' step:' + str(i)
            print(infos)
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(str(observation))

            # Make the algorithm learn based on the results
            infos= "reward:"+ str(round(reward,3)) + " cumulated reward:" + str(round(cumulated_reward,3))
            
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                # rospy.logwarn("NOT DONE")
                infos+= '\t\t NOT DONE'
                state = nextState
            else:
                # rospy.logwarn("DONE")
                infos+= '\t\t DONE'
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            infos += '\n\n'
            
            print(infos)

        # save qlearn values to file
        with open(filename, 'wb') as f:
            log_values['qlearn.q'] = qlearn.q
            log_values['current_episode'] = last_episode + x
            log_values['nepisodes'] = nepisodes
            log_values['cumulated_reward'].append(cumulated_reward)
            pickle.dump(log_values, f)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))
    
    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()