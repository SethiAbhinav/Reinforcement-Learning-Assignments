from environment import bandit_env

# create bandit_env
q_mean = [2.5, -3.5, 1.0, 5.0, -2.5]
q_stddev = [0.33, 1.0, 0.66, 1.98, 1.65] 

arm_bandit_env = bandit_env(q_mean, q_stddev)

print("Number of levers: ",arm_bandit_env.n)
print("Mean of each lever: ",arm_bandit_env.r_mean)
print("Standard Deviation of each lever: ",arm_bandit_env.r_stddev)
print()


