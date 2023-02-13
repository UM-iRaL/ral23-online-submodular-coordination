import simulator
from agent import Agent
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
# from planner import Planner
import time
import psutil
# from memory_profiler import memory_usage
import networkx as nx
from target import Target
import math
import itertools

N_AGENTS = 2
N_TARGETS = 2
WIDTH = 100
HEIGHT = 100
RADIUS = 10
ACTIONS = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [2,0], [0,2], [-2,0], [0,-2],[3,0], [0,3], [-3,0], [0,-3]])
HORIZON = 50    # seconds
N_STEPS = 2500  # rounds of actions
N_TRIAL = 50    # number of Monte-Carlo simulations


def OSG():
    all_min_dist = [[], []]

    for idx in range(N_TRIAL):
        np.random.seed(idx)
        # initialize agents/FSF*
        # line
        targets = create_target([[2,3], [2,-3]])
        agents = create_agent([[0,1], [0,-1]])
        # box
        # targets = create_target([[3,1], [-3,1]])
        # agents = create_agent([[1,0], [-1,0]])
    

        # at every step, each agent gets feedback and outputs
        for t in range(N_STEPS):
            for agent in agents:
                # output action probability distribution by FSF*
                agent.get_action_prob_dist(t)

                # sample the next action
                next_action_index = np.random.choice(agent.action_indices, 1, p=agent.action_prob_dist[t])[0]
                agent.next_action_index = next_action_index
                agent.action_hist.append(next_action_index)

            for target in targets:
                # all target move for one step
                target.update_state(t, HORIZON)
                target.traj.append(target.state)
                target.min_dist = math.inf  

            agents_considered = []
            for agent in agents:
                # get loss vector
                agent.get_losses(agents_considered, targets, t)

                # update experts
                agent.update_experts(t)

                # apply the next action 
                agent.apply_next_action()
                agent.traj.append(agent.state)

                agents_considered.append(agent)

                for target in targets:
                    target.min_dist = min([np.linalg.norm(ag.state - target.state) for ag in agents_considered])

            # update min_dist for all targets after this agent's action
            for target in targets:
                target.min_dist_hist.append(target.min_dist)

        for tar_idx, tar in enumerate(targets):
            all_min_dist[tar_idx].append(tar.min_dist_hist)

    np.save('all_'+str(HORIZON)+'_'+str(N_STEPS)+'.npy', all_min_dist)

    plt.figure(1)
    for idx,tar in enumerate(targets):
        xx = [x for (x,y) in tar.traj]
        yy = [y for (x,y) in tar.traj]
        plt.plot(xx, yy, linestyle="--", label="Target "+str(idx+1), linewidth=2) 
        plt.scatter(xx[0], yy[0], marker='x', s=70)  
    for idx,agent in enumerate(agents):
        xx = [x for (x,y) in agent.traj]
        yy = [y for (x,y) in agent.traj]
        plt.plot(xx, yy, label="Drone "+str(idx+1), linewidth=2)
        plt.scatter(xx[0], yy[0], marker='o', s=70)  
        # if idx == 0:
        #     plt.plot(xx, yy, marker="x", label="Drone "+str(idx+1))
        # else:
        #     plt.plot(xx, yy, marker="*", label="Drone "+str(idx+1))
    plt.axis('square')
    # plt.legend(prop={'family':'Times New Roman', 'size':20})
    # plt.title(label='OSG (Horizon = '+str(HORIZON)+'s, '+str(N_STEPS)+' Time steps)',family='Times New Roman',size=18)
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    # line
    plt.ylim(-5, 5) 
    plt.xlabel("X", fontname="Times New Roman", fontsize=20)
    plt.ylabel("Y", fontname="Times New Roman", fontsize=20)
    plt.savefig("line_"+str(HORIZON)+"s_"+str(N_STEPS)+"step", bbox_inches='tight')
    # box
    # plt.xlim(-35, 35)
    # plt.ylim(-10, 35) 
    # plt.xlabel("X", fontname="Times New Roman", fontsize=20)
    # plt.ylabel("Y", fontname="Times New Roman", fontsize=20)
    # plt.savefig("box_"+str(HORIZON)+"s_"+str(N_STEPS)+"step", bbox_inches='tight')
    plt.show()

    # plt.figure(2)
    # # for idx, tar in enumerate(targets):
    # for idx in range(N_TARGETS):
    #     mean = []
    #     std = []
    #     for i in range(N_STEPS):
    #         data = [all_min_dist[idx][j][i] for j in range(N_TRIAL)]
    #         mean.append(sum(data) / N_TRIAL)
    #         std.append(np.std(data))
    #     plt.plot(np.linspace(1,N_STEPS,N_STEPS) * HORIZON/N_STEPS, mean, label="Target "+str(idx+1))
    #     plt.fill_between(np.linspace(1,N_STEPS,N_STEPS) * HORIZON/N_STEPS, np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.3)
    # # plt.legend(prop={'family':'Times New Roman', 'size':20}, loc='upper right')#, frameon=False)
    # plt.xticks(fontname="Times New Roman", fontsize=20)
    # plt.yticks(fontname="Times New Roman", fontsize=20)
    # plt.xlabel("Time (s)", fontname="Times New Roman", fontsize=20)
    # plt.ylabel("Minimum Distance", fontname="Times New Roman", fontsize=20)
    # # line
    # plt.ylim(0, 5)
    # plt.gca().set_aspect(aspect=4.3)
    # plt.savefig("dist_"+str(N_TRIAL)+"trial_line_"+str(HORIZON)+"s_"+str(N_STEPS)+"step", bbox_inches='tight')
    # # box
    # # plt.ylim(0, 25) 
    # # plt.gca().set_aspect(aspect=1.5)
    # # plt.savefig("dist_"+str(N_TRIAL)+"trial_box_"+str(HORIZON)+"s_"+str(N_STEPS)+"step", bbox_inches='tight')
    # plt.show()


def create_agent(states):
    agents = []
    for i in range(len(states)):
        agents.append(Agent(state=states[i], radius=RADIUS, actions=ACTIONS, n_time_step=N_STEPS, time_step=HORIZON/N_STEPS, color=np.random.rand(3)))
    return agents


def create_target(states):
    targets = []
    for i in range(len(states)):
        targets.append(Target(states[i], time_step=HORIZON/N_STEPS, idx=i, color=[1,1,1]))
    return targets


def main():
    OSG()
    plt.show()


if __name__ == "__main__":
    OSG()
    # main()