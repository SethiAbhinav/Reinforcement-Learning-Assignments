from environment import bandit_env
import numpy as np

# np.random.seed(3)
cnt = 0
rewards_main = [0, 0, 0, 0, 0]
avg_rewards = [0, 0, 0, 0, 0]
# q follows gausian distribution
q_mean = [2.5, -3.5, 1.0, 5.0, -2.5]
q_stddev =  [0.33, 1.0, 0.66, 1.98, 1.65]

for i in range(100):
    reward_bandit = [0, 0, 0, 0, 0]
    action_bandit = [0, 0, 0, 0, 0]

    rewards = [0, 0, 0, 0, 0]
    # apply e greedy policy with e = 0.1
    e = 0.1

    # initialize the bandit environment
    env = bandit_env(q_mean, q_stddev)

    # initialize the number of rounds
    n_rounds = 1000

    # e greedy policy
    def e_greedy(q_mean, q_stddev, e):
        """
        e greedy policy
        
        :params:
        q_mean: takes a list of reward mean
        q_stddev: takes a list of reward standard deviation
        e: the probability of choosing a random arm
        """
        # choose random action with probability e
        if np.random.rand() < e:
            return np.random.randint(0, len(q_mean))
        # choose greedy action with probability 1-e
        else:
            return np.argmax(q_mean)
        
    # call the e greedy policy 1000 times
    for i in range(n_rounds):
        action = e_greedy(q_mean, q_stddev, e)
        # print(action)    
        reward = env.pull(action)
        # print reward
        reward_bandit[action] += reward
        action_bandit[action] += 1

        rewards[action] = reward_bandit[action] / action_bandit[action]

    print(rewards)
    cnt += 1
    for i in range(5):
        rewards_main[i] += rewards[i]
        avg_rewards[i] = rewards_main[i] / cnt
    print(avg_rewards)