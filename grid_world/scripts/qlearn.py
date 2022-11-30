'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
#!/usr/bin/env python3
import random
import rospy
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = 0
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)        

        info = "[" + str(state) + "]"
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            info +="[ R ]"
        else:
            indices = [i for i, x in enumerate(q) if x == maxQ]
            i = random.choice(indices)
            info +=" [M] "
            action = self.actions[i] 

        # print
        info += ' '.join(["a:"+str(a) + " q:" + str(round(self.getQ(state, a), 2)) for a in self.actions])
        info += " MaxQ:" + str(round(maxQ, 2)) + " Next A:" + str(action)
        print(info)
        
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)