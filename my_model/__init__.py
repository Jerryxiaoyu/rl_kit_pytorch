import gym
from gym.envs.registration import registry, make, spec
from my_model.laikago import *
from my_model.jaco2 import *


def register(id,*args,**kvargs):
	if id in registry.env_specs:
		return
	else:
		return gym.envs.registration.register(id,*args,**kvargs)



# ------------bullet-------------

register(
        id='LaikagoBulletEnv-v1',
        entry_point='my_model.laikago:LaikagoBulletEnv',
        max_episode_steps=1000,
        reward_threshold=20000.0,
)
register(
        id='LaikagoTorqueBulletEnv-v1',
        entry_point='my_model.laikago:LaikagoTorqueBulletEnv',
        max_episode_steps=1000,
        reward_threshold=20000.0,
)



register(
        id='KinovaBulletEnv-v1',
        entry_point='my_model.jaco2:KinovaGymEnv',
        max_episode_steps=1000,
        reward_threshold=20000.0,
)

def getList():
	btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet')>=0]
	return btenvs
