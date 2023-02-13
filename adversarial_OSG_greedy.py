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

WIDTH = 100
HEIGHT = 100
RADIUS = 10
ACTIONS = np.array([[1,0], [0,1], [-1,0], [0,-1], [0,-2], [2,0], [0,2], [-2,0]])
HORIZON = 50    # seconds
N_STEPS = 1000  # rounds of actions
N_TRIAL = 50    # number of Monte-Carlo simulations
ADVERSARIAL_THRESHOLD = 1.5
TARGET_POS = [[0,3], [0,-3]]
AGENT_POS = [[0,1], [0,-1]] 
N_AGENTS = len(AGENT_POS)
N_TARGETS = len(TARGET_POS)

def adversarial_OSG_greedy():

    all_min_dist1 = [[], []]
    n_adversarial_trigger = np.array([0, 0])

    for idx in range(N_TRIAL):
        np.random.seed(idx)

        # initialize agents/FSF*
        targets1 = create_target(TARGET_POS)
        agents1 = create_agent(AGENT_POS)

        # at every step, each agent gets feedback and outputs
        for t in range(N_STEPS):
            for agent in agents1:
                # output action probability distribution by FSF*
                agent.get_action_prob_dist(t)

                # sample the next action
                next_action_index = np.random.choice(agent.action_indices, 1, p=agent.action_prob_dist[t])[0]
                agent.next_action_index = next_action_index
                agent.action_hist.append(next_action_index)

            for target in targets1:
                # all target move for one step
                target.adversarial_update_state(t, ADVERSARIAL_THRESHOLD)
                target.traj.append(target.state)
                target.min_dist = math.inf  

            agents_considered1 = []
            for agent in agents1:
                # get loss vector
                agent.get_losses(agents_considered1, targets1, t)

                # update experts
                agent.update_experts(t)

                # apply the next action 
                agent.apply_next_action()
                agent.traj.append(agent.state)

                agents_considered1.append(agent)

                for target in targets1:
                    target.min_dist = min([np.linalg.norm(ag.state - target.state) for ag in agents_considered1])

            # update min_dist for all targets after this agent's action
            for target in targets1:
                target.min_dist_hist.append(target.min_dist)

        for tar_idx, tar in enumerate(targets1):
            all_min_dist1[tar_idx].append(tar.min_dist_hist)
            n_adversarial_trigger[tar_idx] += tar.n_adversarial_trigger

    targets2 = create_target(TARGET_POS)
    agents2 = create_agent(AGENT_POS)
    for t in range(N_STEPS):
        agents_considered2 = []
        obj = 0
        for agent in agents2:
            for i in range(agent.n_actions):
                action = agent.action_indices[i]
                curr_state = agent.state
                agent.state = agent.motion_model(agent.state, agent.actions[action])
                agents_considered2.append(agent)
                temp_obj = sum([max([1 / np.linalg.norm(tar.state - ag.state) for ag in agents_considered2]) for tar in targets2])
                if temp_obj > obj:
                    obj = temp_obj
                    agent.next_action_index = i
                agents_considered2.remove(agent)
                agent.state = curr_state

            # apply the next action 
            agent.action_hist.append(agent.next_action_index)
            agent.apply_next_action()
            agent.traj.append(agent.state)

            # then consider this agent at the next round of greedy
            agents_considered2.append(agent)

        # for idx, target in enumerate(targets2):
        for target in targets2:
            # all target move for one step
            target.adversarial_update_state(t, ADVERSARIAL_THRESHOLD)
            target.traj.append(target.state)
            target.min_dist = min([np.linalg.norm(target.state - agent.state) for agent in agents2])   
            target.min_dist_hist.append(target.min_dist)


    print("OSG # of adversarial triggers: "+str(n_adversarial_trigger / N_TRIAL))

    plt.figure(1)
    for idx,tar in enumerate(targets1):
        xx = [x for (x,y) in tar.traj]
        yy = [y for (x,y) in tar.traj]
        plt.plot(xx, yy, linestyle="--", label="Target "+str(idx+1), linewidth=2) 
        plt.scatter(xx[0], yy[0], marker='x', s=70)  
    for idx,agent in enumerate(agents1):
        xx = [x for (x,y) in agent.traj]
        yy = [y for (x,y) in agent.traj]
        plt.plot(xx, yy, label="Drone "+str(idx+1)+" OSG", linewidth=2)
        plt.scatter(xx[0], yy[0], marker='o', s=70)  
    # ax1.legend(prop={'family':'Times New Roman', 'size':8}, loc='lower right')
    # plt.title(label='OSG (Horizon = '+str(HORIZON)+'s, '+str(N_STEPS)+' Time steps)',family='Times New Roman',size=18)
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    plt.xlim(-5, 70) 
    plt.ylim(-6, 6) 
    plt.xlabel("X", fontname="Times New Roman", fontsize=20)
    plt.ylabel("Y", fontname="Times New Roman", fontsize=20)
    plt.gca().set_aspect(aspect=4)
    plt.savefig("adversarial_OSG_traj", bbox_inches='tight')
    plt.show()

    # ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.figure(2)
    for idx,tar in enumerate(targets2):
        xx = [x for (x,y) in tar.traj]
        yy = [y for (x,y) in tar.traj]
        plt.plot(xx, yy, linestyle="--", label="Target "+str(idx+1), linewidth=2) 
        plt.scatter(xx[0], yy[0], marker='x', s=70)  
    for idx,agent in enumerate(agents2):
        xx = [x for (x,y) in agent.traj]
        yy = [y for (x,y) in agent.traj]
        plt.plot(xx, yy, label="Drone "+str(idx+1)+" Greedy", linewidth=2)
        plt.scatter(xx[0], yy[0], marker='o', s=70)  
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    plt.xlim(-5, 70) 
    plt.ylim(-6, 6) 
    plt.xlabel("X", fontname="Times New Roman", fontsize=20)
    plt.ylabel("Y", fontname="Times New Roman", fontsize=20)
    plt.gca().set_aspect(aspect=4)
    plt.savefig("adversarial_greedy_traj", bbox_inches='tight')
    plt.show()

    plt.figure(3)
    for idx in range(N_TARGETS):
        plt.plot(np.linspace(1,N_STEPS,N_STEPS) * HORIZON/N_STEPS, targets2[idx].min_dist_hist, linewidth=2, label="Target "+str(idx+1)+" ($\widehat{\sf{SG}}$)")
    for idx in range(N_TARGETS):   
        mean = []
        std = []
        for i in range(N_STEPS):
            data = [all_min_dist1[idx][j][i] for j in range(N_TRIAL)]
            mean.append(sum(data) / N_TRIAL)
            std.append(np.std(data))
        plt.plot(np.linspace(1,N_STEPS,N_STEPS) * HORIZON/N_STEPS, mean, label="Target "+str(idx+1)+" ($\sf{OSG}$)")
        plt.fill_between(np.linspace(1,N_STEPS,N_STEPS) * HORIZON/N_STEPS, np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.3)
    plt.legend(prop={'family':'Times New Roman', 'size':20}, loc=(0,0.7), ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.7, labelspacing=0.1, handlelength=2)
    plt.ylim(bottom=0.5)
    plt.gca().set_aspect(aspect=7.3)
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    plt.xlabel("Time (s)", fontname="Times New Roman", fontsize=20)
    plt.ylabel("Minimum Distance", fontname="Times New Roman", fontsize=20)
    plt.savefig("adversarial_OSG_greedy_dist.png", bbox_inches='tight')
    plt.show()

    # plt.figure(3)
    # # for idx, tar in enumerate(targets):
    # for idx in range(N_TARGETS):
    #     mean = []
    #     std = []
    #     for i in range(N_STEPS):
    #         data = [all_min_dist2[idx][j][i] for j in range(N_TRIAL)]
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
    # # plt.ylim(0, 5)
    # # plt.gca().set_aspect(aspect=5)
    # plt.savefig("final_greedy.png", bbox_inches='tight')
    # plt.show()

def greedy_no_prediction():
    aver = 0
    for idx in range(1):
        # np.random.seed(idx)
        targets = create_target([[2,5]])#, [2,-5]])
        agents = create_agent([[0,5]])#, [0,-5]])

        for t in range(N_STEPS):
            agents_considered = []
            obj = 0
            for agent in agents:
                for i in range(agent.n_actions):
                    action = agent.action_indices[i]
                    curr_state = agent.state
                    agent.state = agent.motion_model(agent.state, agent.actions[action])
                    agents_considered.append(agent)
                    temp_obj = sum([max([1 / np.linalg.norm(tar.state - ag.state) + 0.001 for ag in agents_considered]) for tar in targets])
                    if temp_obj > obj:
                        obj = temp_obj
                        agent.next_action_index = i
                    agents_considered.remove(agent)
                    agent.state = curr_state

                # apply the next action 
                agent.action_hist.append(agent.next_action_index)
                agent.apply_next_action()
                agent.traj.append(agent.state)

                # then consider this agent at the next round of greedy
                agents_considered.append(agent)

            for target in targets:
                # all target move for one step
                target.final_update_state(t, HORIZON)
                target.traj.append(target.state)
                target.min_dist = min([np.linalg.norm(target.state - agent.state) for agent in agents])   
                target.min_dist_hist.append(target.min_dist)
    

    plt.figure(2)
    for tar in targets:
        xx = [x for (x,y) in tar.traj]
        yy = [y for (x,y) in tar.traj]
        plt.plot(xx, yy, linestyle=":")   
    for agent in agents:
        xx = [x for (x,y) in agent.traj]
        yy = [y for (x,y) in agent.traj]
        plt.plot(xx, yy)
    plt.title(label='Greedy no prediction',family='Times New Roman',size=18)
    plt.axis('equal')
    plt.show()    


def create_agent(states):
    agents = []
    for i in range(len(states)):
        agents.append(Agent(state=states[i], radius=RADIUS, actions=ACTIONS, n_time_step=N_STEPS, time_step=HORIZON/N_STEPS, color=np.random.rand(3)))
    return agents


def create_target(states):
    '''
    for tracking targets that follow a route
    '''
    targets = []
    for i in range(len(states)):
        targets.append(Target(states[i], time_step=HORIZON/N_STEPS, idx=i, color=[1,1,1]))
    return targets

if __name__ == "__main__":
    adversarial_OSG_greedy()
