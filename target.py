import numpy as np
import math
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class Target(object):

    def __init__(self, init_state, time_step, idx, color):
        self.state = np.array(init_state)
        self.u = np.array([1.0, 0])

        self.min_dist = math.inf # distance to the closest agent
        self.min_dist_hist = []

        self.idx = idx

        self.traj = [np.array(init_state)]

        self.A = np.eye(2)                          # system matrix Jacobian
        self.B = [] #np.array([[1], [0]])           # input matrix Jacobian

        self.agent_meas_cov = []        # agents that observe the target and corresponding measurements and covariances                                                 
        self.agent_meas_cov_hist = []

        self.time_step = time_step

        # Plotting
        self.color = color

        self.n_adversarial_trigger = 0
        self.adversarial_trigger = -100
        if self.idx == 0:
            self.u1 = -np.array([0, 2])
            self.u2 = -np.array([0, -40])
        else:
            self.u1 = np.array([0, 2])
            self.u2 = np.array([0, -40])


    def update_state(self, t, H):
        '''
        Updates the state of the target
        :param t: Index of time step.
        :param H: Horizon.
        :return: None.
        '''
        u = np.array([1.0, 0])
        # u = np.array([1.0, np.sin(t)])

        # box
        # if self.idx == 0:
        #     if t * self.time_step < H / 4:
        #         u = np.array([0, 1.0])
        #     elif t * self.time_step < H / 2:
        #         u = np.array([1.0, 0])
        #     elif t * self.time_step < H * 4/5:
        #         u = np.array([0, -1.0])
        #     else:
        #         u = np.array([-1.0, 0])
        # else:
        #     if t * self.time_step < H / 4:
        #         u = np.array([0, 1.0])
        #     elif t * self.time_step < H / 2:
        #         u = np.array([-1.0, 0])
        #     elif t * self.time_step < H * 4/5:
        #         u = np.array([0, -1.0])
        #     else:
        #         u = np.array([1.0, 0])

        # if t % 2 == 0:
        #     u = np.array([0.5, 2.5])
        # else:
        #     u = np.array([0.5, -2.5])
        # if self.min_dist < 5:
        #     self.u = np.dot(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]), self.u)
        self.state = np.dot(self.A, self.state) + u * self.time_step
        # self.state = np.dot(self.A, self.state) + np.array([self.time_step, self.time_step*2*np.sin(t * np.pi/200)])


    def adversarial_update_state(self, t, threshold):
        '''
        Updates the state of the target
        :param t: Index of time step.
        :param threshold: Threshold for triggering adversarial maneuvering.
        :return: None.
        ''' 
        # adversarial
        if t >= self.adversarial_trigger + 20 and self.min_dist < threshold:
            self.adversarial_trigger = t
            self.n_adversarial_trigger += 1
            self.u1 = -self.u1
            self.u2 = -self.u2

        if self.adversarial_trigger <= t < self.adversarial_trigger + 19:
            u = self.u1
        elif t < self.adversarial_trigger + 20:
            u = self.u2 + np.array([30, 0])
        else:
            u = np.array([1, 0])
        self.state = np.dot(self.A, self.state) + u * self.time_step

    def random_update_state(self, t, H, bar):
        '''
        Updates the state of the target with truncated input noise
        :param t: Index of time step.
        :param H: Time horizon of simulation.
        :param threshold: Threshold for triggering adversarial maneuvering.
        :return: None.
        ''' 
        # # random + line
        # u = np.array([1.0, truncnorm.rvs(-bar, bar, loc=0, scale=3)])

        # random + box
        var = 2
        if self.idx == 0:
            if t * self.time_step < H / 4:
                u = np.array([truncnorm.rvs(-bar, bar, loc=0, scale=var), 1.0])
            elif t * self.time_step < H / 2:
                u = np.array([1.0, truncnorm.rvs(-bar, bar, loc=0, scale=var)])
            elif t * self.time_step < H * 4/5:
                u = np.array([truncnorm.rvs(-bar, bar, loc=0, scale=var), -1.0])
            else:
                u = np.array([-1.0, truncnorm.rvs(-bar, bar, loc=0, scale=var)])
        else:
            if t * self.time_step < H / 4:
                u = np.array([truncnorm.rvs(-bar, bar, loc=0, scale=var), 1.0])
            elif t * self.time_step < H / 2:
                u = np.array([-1.0, truncnorm.rvs(-bar, bar, loc=0, scale=var)])
            elif t * self.time_step < H * 4/5:
                u = np.array([truncnorm.rvs(-bar, bar, loc=0, scale=var), -1.0])
            else:
                u = np.array([1.0, truncnorm.rvs(-bar, bar, loc=0, scale=var)])

        self.state = np.dot(self.A, self.state) + u * self.time_step
