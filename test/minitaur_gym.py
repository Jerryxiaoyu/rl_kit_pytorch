
import math
import numpy as np
from pybullet_envs.bullet import minitaur_gym_env
import argparse
from pybullet_envs.bullet import minitaur_env_randomizer



def MinitaurTest():
    randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    environment = minitaur_gym_env.MinitaurBulletEnv(
      render=True,
      motor_velocity_limit=np.inf,
      pd_control_enabled=True,
      hard_reset=False,
      env_randomizer =  randomizer,
      on_rack=False)

    sum_reward = 0
    steps = 20000
    amplitude_1_bound = 0.1
    amplitude_2_bound = 0.1
    speed = 1

    for step_counter in range(steps):
      time_step = 0.01
      t = step_counter * time_step

      amplitude1 = amplitude_1_bound
      amplitude2 = amplitude_2_bound
      steering_amplitude = 0
      if t < 10:
        steering_amplitude = 0.1
      elif t < 20:
        steering_amplitude = -0.1
      else:
        steering_amplitude = 0

      # Applying asymmetrical sine gaits to different legs can steer the minitaur.
      a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)
      a2 = math.sin(t * speed + math.pi) * (amplitude1 - steering_amplitude)
      a3 = math.sin(t * speed) * amplitude2
      a4 = math.sin(t * speed + math.pi) * amplitude2
      action = [a1, a2, a2, a1, a3, a4, a4, a3]
      _, reward, done, _ = environment.step(action)
      sum_reward += reward
      if done:
        break
    environment.reset()



# randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
# environment = minitaur_gym_env.MinitaurBulletEnv(
#     render=True,
#     motor_velocity_limit=np.inf,
#     pd_control_enabled=True,
#     hard_reset=False,
#     env_randomizer=randomizer,
#     on_rack=False)

import gym
environment = gym.make('MinitaurBulletEnv-v0', render=True,
      motor_velocity_limit=np.inf,
      pd_control_enabled=True,
      hard_reset=False,
      on_rack=False)

sum_reward = 0
steps = 200
amplitude_1_bound = 0.1
amplitude_2_bound = 0.1
speed = 1

environment.reset()

for step_counter in range(steps):
    time_step = 0.01
    t = step_counter * time_step

    amplitude1 = amplitude_1_bound
    amplitude2 = amplitude_2_bound
    steering_amplitude = 0
    if t < 10:
        steering_amplitude = 0.1
    elif t < 20:
        steering_amplitude = -0.1
    else:
        steering_amplitude = 0

    # Applying asymmetrical sine gaits to different legs can steer the minitaur.
    a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)
    a2 = math.sin(t * speed + math.pi) * (amplitude1 - steering_amplitude)
    a3 = math.sin(t * speed) * amplitude2
    a4 = math.sin(t * speed + math.pi) * amplitude2
    action = [a1, a2, a2, a1, a3, a4, a4, a3]
    _, reward, done, _ = environment.step(action)
    sum_reward += reward
    if done:
        break

print('Task is over!')