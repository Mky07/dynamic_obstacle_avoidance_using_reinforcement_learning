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
import obstacle_avoidance
import numpy as np
import pickle

import os

# def defineLogVariables():
#     return {
#         "qlearn.q": {},
#         "current_episode": 0,
#         "nepisodes": 0   
#     }


if __name__ == '__main__':

    rospy.init_node('obstacle_avoidance', anonymous=True, log_level=rospy.WARN)
    filename = "/home/mky/rl_ws/src/openai_examples_projects/dynamic_obstacle_avoidance_using_reinforcement_learning/models/model4.pkl"

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
    env = gym.make('ObstacleAvoidance-v0')
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
    
    start_time = time.time()
    highest_reward = 0
    
    print("q valeus: {}".format(qlearn.q))
    print("last episode: {}".format(log_values['current_episode']))
    print("cumulated_reward: {}".format(log_values['cumulated_reward']))
    
    cumulated_reward_10_episode = []
    mean_val = 0
    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### WALL START EPISODE=>" + str(log_values['current_episode']))
        # rospy.logerr("*****************************************\nq values:\n {}\n*****************************************".format(qlearn.q))
        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start" + " episode: " +str(log_values['current_episode']) +"  step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(nextState))
            
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            
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

        mean_val+= cumulated_reward
        # if x%10 == 0:
        #     cumulated_reward_10_episode.append(mean_val/10.0)
        #     mean_val = 0
        cumulated_reward_10_episode.append(cumulated_reward)

        print("cumulated_reward: {}".format(log_values['cumulated_reward']))

        # print("Cumulated_reward_10_episode: {}".format(cumulated_reward_10_episode))
    
    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))
    

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()