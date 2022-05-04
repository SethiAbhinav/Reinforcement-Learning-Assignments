from environment import bandit_env
import numpy as np


def argmax(reward):
    max_reward = reward[0]
    ties = [0]

    for idx, i in enumerate(reward[1:]):
        if i > max_reward:
            max_reward = i
            ties = [idx+1]
        if i == max_reward:
            ties.append(idx+1)
    
    if len(ties) > 1:
        index_of_max = np.random.choice(ties)
    else:
        index_of_max = ties[0]
    
    return index_of_max

# create bandit_env
q_mean = [2.5, -3.5, 1.0, 5.0, -2.5]
q_stddev = [0.33, 1.0, 0.66, 1.98, 1.65] 

arm_bandit_env = bandit_env(q_mean, q_stddev)

print("Number of levers: ",arm_bandit_env.n)
print("Mean of each lever: ",arm_bandit_env.r_mean)
print("Standard Deviation of each lever: ",arm_bandit_env.r_stddev)

# implement e greedy policy
def e_greedy(rewards, e):
    # choose random action with probability e
    if np.random.rand() < e:
        return np.random.randint(0, len(rewards))

    # choose greedy action with probability 1-e
    else:
        return argmax(rewards)

average_rewards = np.zeros(6)

epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]

for idx, e in enumerate(epsilons):
    actual_reward = 0
    actions = [0, 0, 0, 0, 0]
    rewards = [0, 0, 0, 0, 0]

    for i in range(1000):   
        np.random.seed(i)

        action = e_greedy(rewards, e)
        reward = arm_bandit_env.pull(action)
        actual_reward += reward
        
        actions[action] += 1

        rewards[action] = rewards[action] + (1/(actions[action])) * (reward - rewards[action])
    average_rewards[idx] = actual_reward/1000
print(average_rewards)
