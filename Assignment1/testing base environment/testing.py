from environment import bandit_env
import numpy as np

np.random.seed(10)


states = [0, 1, 2] # Hostel, Canteen, Academic Building
rewards = [-1, +1, +3]
actions = [0, 1] # attend, eat

gamma = 0.9


data = [
        
        # Hostel
        [
         #attend
         [
          [0.5, 0, -1], # prob, next state, reward
          [0.5, 2, +3]
         ],
        
         #eat
         [
          [1, 1, +1]
         ]
        ],
        
        # Canteen
        [
         #attend
         [
          [0.3, 0, -1],
          [0.6, 2, +3],
          [0.1, 1, +1]
         ],
         
         #eat
         [
          [1,1,+1]
         ]
        ],
        
        # Academic Building
        [
         #attend
         [
          [0.3, 1, +1],
          [0.7, 2, +3]
         ],
         #eat
         [
          [0.8, 1, +1],
          [0.2, 2, +3]
         ]
        ]
]

# initialize v
v = [0, 0, 0]

# initialize policy
policy = [0, 1, 0]

# call policy iteration with gamma 0.9 and theta 0.001
print(policy)
