import numpy as np
import math
import matplotlib.patches as patch
import matplotlib.pyplot as plt

class Agent(object):

    def __init__(self, state, radius, actions, n_time_step, time_step, color):
        self.state = np.array(state)        # [x_1; x_2]
        self.radius = radius      # limited sensing range

        self.time_step = time_step
        # print(self.time_step)

        self.traj = [np.array(state)]
        self.action_hist = []

        # Action Space
        self.actions = actions
        self.n_actions = len(self.actions)
        self.action_indices = np.array(range(self.n_actions))
        self.next_action_index = 0

        # Plotting
        self.color = color

        # OSG
        self.J = math.ceil(math.log2(n_time_step)) # number of experts
        self.g = np.sqrt(math.log(self.J) / n_time_step) 
        self.beta = 1 / n_time_step
        self.gamma = [np.sqrt(math.log(self.n_actions * n_time_step) / 2 ** (j - 1)) for j in range(1, self.J + 1)]   # J x 1
        self.expert_weight = [[1.0 / self.J for i in range(self.J)] for j in range(n_time_step+1)]                                   # weights of experts        
        self.action_weight = [[[1.0 / self.n_actions for k in range(self.n_actions)] for i in range(self.J)] for j in range(n_time_step+1)]  # weights of actions for all experts        
        self.action_prob_dist = [[0.0 for i in range(self.n_actions)] for j in range(n_time_step)] 
        self.loss = [[0.0 for i in range(self.n_actions)] for j in range(n_time_step)] 
        

    def get_losses(self, agents_considered, targets, t):
        """
        Returns the losses of all possible actions based on the estimation result of just executed actions
        :return: The losses of all possible actions.
        """
        losses = np.zeros(self.n_actions)
        obj_action = []

        # f(empty set) = 0 before any agent's moving

        # f(a): distance with action a without any other agent's action
        for i in range(self.n_actions):
            action = self.action_indices[i]
            state = self.motion_model(self.state, self.actions[action])
            obj_action.append(sum([1 / (np.linalg.norm(tar.state - state) + 0.001) for tar in targets]))

        # f(A_{i-1}) 
        if len(agents_considered) == 0:
            losses = -np.array(obj_action) / max(np.array(obj_action))
        else:
            curr_obj = sum([1 / min([np.linalg.norm(tar.state - ag.state) + 0.001 for ag in agents_considered]) for tar in targets])

            for i in range(self.n_actions):
                action = self.action_indices[i]
                state = self.motion_model(self.state, self.actions[action])
                # f(A_{i-1} U a)
                temp_obj = 0
                for tar in targets:
                    temp_obj += 1 / (min([np.linalg.norm(tar.state - state), tar.min_dist]) + 0.001)

                losses[i] = (curr_obj - temp_obj) / max(np.array(obj_action)) #obj_action[i]
        self.loss[t] = losses 

    def get_action_prob_dist(self, t):
        """
        Returns the output of FSF* (the predicted action probability distribution)
        :param t: The index of time step.
        :return: None.
        """
        q = self.expert_weight[t] # J x 1
        p = self.action_weight[t] # J x m
        self.action_prob_dist[t] = np.dot(q, p).tolist() # m x 1

    def apply_next_action(self):
        """
        Applies the next action to modify the agent state.
        :param t: Time step.
        :return: None
        """
        self.state = self.motion_model(self.state, self.actions[self.next_action_index])

    def update_experts(self, t):
        """
        Updates the parameters of experts after getting losses (from t to t + 1)
        :param t: The index of time step
        :return: None
        """
        for j in range(self.J):
            v = [self.action_weight[t][j][i] * np.exp(-self.gamma[j] * self.loss[t][i]) for i in range(self.n_actions)]
            self.action_weight[t + 1][j] = [self.beta * np.sum(v) / self.n_actions + (1 - self.beta) * v[i] for i in range(self.n_actions)]
            self.expert_weight[t + 1][j] = self.expert_weight[t][j] * np.exp(-self.g * np.dot(np.array(self.loss[t]), np.array(self.action_weight[t][j])) / np.linalg.norm(np.array(self.action_weight[t][j]), ord=1))

        self.action_weight[t + 1] = [self.action_weight[t + 1][j] / np.linalg.norm(self.action_weight[t + 1][j], ord=1) for j in range(self.J)]
        self.expert_weight[t + 1] = self.expert_weight[t + 1] / np.linalg.norm(np.array(self.expert_weight[t + 1]), ord=1)

    def motion_model(self, state, action):
        '''
        :param state: The current state at time t.
        :param action: The current action at time t.
        :return: The resulting state x_{t+1}
        '''
        return state[0] + action[0] * self.time_step, state[1] + action[1] * self.time_step