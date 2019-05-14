import gym
import numpy as np
import math


from my_model import *

env = gym.make('KinovaBulletEnv-v1', render=True,  )

sum_reward = 0
steps = 20000


env.reset()
init_angle = np.array([-0.09090909, -0.48609111,  0.40011803, -0.09090909, -0.48609111,
                      0.40011803, -0.09090909, -0.48609111,  0.40011803, -0.09090909,
                        -0.58609111,  0.20011803])

#

# for step_counter in range(steps):
#     time_step = 0.01
#     t = step_counter * time_step
#
#
#     action = env.action_space.sample()
#
#     print(action)
#     # action = init_angle
#     _, reward, done, _ = env.step(action)
#     sum_reward += reward
#     if done:
#         break
episode = 0
print('episode = ', episode)
while True:
    action = env.action_space.sample()


    _, reward, done, _ = env.step(action)
    sum_reward += reward
    if done:
        episode += 1
        env.reset()
        done = False
        print('episode = ', episode)
print("Test is over!")