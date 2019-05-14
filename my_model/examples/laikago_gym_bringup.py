import gym
import numpy as np
import math


from my_model import *

env = gym.make('LaikagoTorqueBulletEnv-v1', render=True  )

sum_reward = 0
steps = 20000
amplitude_1_bound = 0.1
amplitude_2_bound = 0.1
speed = 1

env.reset()
init_angle = np.array([-0.09090909, -0.48609111,  0.40011803, -0.09090909, -0.48609111,
                      0.40011803, -0.09090909, -0.48609111,  0.40011803, -0.09090909,
                        -0.58609111,  0.20011803])

#

for step_counter in range(steps):
    time_step = 0.01
    t = step_counter * time_step

    action = env.action_space.sample()

    print('action: ',action)

    _, reward, done, _ = env.step(action)
    sum_reward += reward
    if done:
        break


print("Test is over!")