from my_model.laikago.laikago import Laikago
from my_model.laikago import bullet_client
import pybullet_data
import numpy as np


pybullet_client = bullet_client.BulletClient()
urdf_root=pybullet_data.getDataPath()
time_step = 0.01
self_collision_enabled =  True

motor_velocity_limit = np.inf
pd_control_enabled = False
acc_motor = True
motor_kp=1.0
motor_kd=0.02
torque_control_enabled = False
motor_protect = True
on_rack = False
kd_for_pd_controllers = 0.3


robot = Laikago(
          pybullet_client=pybullet_client,
         # urdf_root=urdf_root,
          time_step= time_step,
          self_collision_enabled= self_collision_enabled,
          motor_velocity_limit= motor_velocity_limit,
          pd_control_enabled= pd_control_enabled,
          accurate_motor_model_enabled=acc_motor,
          motor_kp=  motor_kp,
          motor_kd=  motor_kd,
          torque_control_enabled=  torque_control_enabled,
          motor_overheat_protection=motor_protect,
          on_rack= on_rack,
           )



print("Test is over!")