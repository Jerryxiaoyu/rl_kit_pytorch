import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet
import numpy as np
import copy
import math
import pybullet_data

## [0] coresponding to position noise stv, [1] -- velocity noise stv, [2] -- torque noise stv
SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0)
#pybullet_data.getDataPath()
INIT_ENDEFFORTPOSITION = [0.537, 0.0, 0.5]
INIT_ENDEFFORTANGLE = 0

class Kinova:
  #TODO urdf path
  def __init__(self,
               pybullet_client,
               robot_type = 'j2n6s300',
               urdfRootPath=os.path.abspath('../model'),
               timeStep=0.01,
               building_env = True,
               useInverseKinematics = False,
               torque_control_enabled = False,
               is_fixed = True):

    self.robot_type = robot_type
    self._pybullet_client = pybullet_client
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35  # TODO not use this, because use URDF info
    self.maxForce = 200.

    self._basePosition = [-0.000000, 0.000000, 0.000000]
    self._baseOrientation = [0.000000, 0.000000, 0.000000, 1.000000]
    self._init_jointPositions = [0, math.pi, math.pi, 0, 0, 0, 1, 1, 1]

    self._torque_control_enabled = torque_control_enabled
    self._observation_noise_stdev = SENSOR_NOISE_STDDEV


    self.useInverseKinematics = useInverseKinematics # 0
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 1

    self.EndEffectorLinkName = '{}_joint_end_effector'.format(self.robot_type)
    self.numFingers = int(self.robot_type[5])# to test
    self.OnlyEndEffectorObervations = useInverseKinematics
    self._is_fixed = is_fixed

    if building_env:
      self.build_env()
    self.reset()

    #lower limits for null space
    self.ll = self.jointLowerLimit[:self.numMotors-self.numFingers]
    #upper limits for null space
    self.ul = self.jointUpperLimit[:self.numMotors-self.numFingers]
    #joint ranges for null space
    self.jr=[5.8,4,5.8,4,5.8,4 ]
    #restposes for null space
    self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66 ]
    #joint damping coefficents
    self.jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]

    self.ee_X_upperLimit = 0.9
    self.ee_X_lowerLimit = -0.9
    self.ee_Y_upperLimit = 0.9
    self.ee_Y_lowerLimit = -0.9
    self.ee_Z_upperLimit = 1.18
    self.ee_Z_lowerLimit = -0.9

    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2

    if self.useInverseKinematics:
      self.endEffectorPos =  [0.537, 0.0, 0.5]
      self.endEffectorAngle = 0

  def build_env(self):
    # build env
    self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
    self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
    self._pybullet_client.setGravity(0, 0, -9.81)

  def reset(self, reload_urdf=True):
    if reload_urdf:
      self.kinovaUid = self._pybullet_client.loadURDF(
                          os.path.join(self.urdfRootPath,"jaco2/urdf/j2n6s300.urdf"),
                          self._basePosition,self._baseOrientation,
                          useFixedBase=self._is_fixed,
                          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)

      self._BuildJointNameToIdDict()
      self._GetJointInfo()
      self._ResetJointState()

      # reset joint state
      for i in range(self.numMotors):
        self._SetDesiredMotorAngleById(self.motorIndices[i], self._init_jointPositions[i], max_velocity= 10)
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.kinovaUid, self._basePosition, self._baseOrientation)
      self._pybullet_client.resetBaseVelocity(self.kinovaUid, [0, 0, 0], [0, 0, 0])

      self._ResetJointState()
      # reset joint state
      for i in range(self.numMotors):
        self._SetDesiredMotorAngleById(self.motorIndices[i], self._init_jointPositions[i], max_velocity=10)

    if self.useInverseKinematics:
      self.endEffectorPos = [0.537, 0.0, 0.5]
      self.endEffectorAngle = 0

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.kinovaUid)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.kinovaUid, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _GetJointInfo(self):
    self.actuator_joint = []
    self.motorNames = []
    self.motorIndices = []
    self.joint_q_index = []
    self.joint_u_index = []
    self.jointLowerLimit = []
    self.jointUpperLimit = []
    self.jointMaxForce = []
    self.jointMaxVelocity = []

    for i in range(self._pybullet_client.getNumJoints(self.kinovaUid)):
      joint_info = self._pybullet_client.getJointInfo(self.kinovaUid, i)
      qIndex = joint_info[3]
      if qIndex > - 1:  # JOINT_FIXED
        self.motorNames.append(joint_info[1].decode("UTF-8"))
        self.motorIndices.append(i)
        self.joint_q_index.append(joint_info[3])
        self.joint_u_index.append(joint_info[4])
        self.jointLowerLimit.append(joint_info[8])
        self.jointUpperLimit.append(joint_info[9])
        self.jointMaxForce.append(joint_info[10])
        self.jointMaxVelocity.append(joint_info[11])
    self.numMotors = len(self.motorNames)

    self.EndEffectorIndex = self._joint_name_to_id[self.EndEffectorLinkName]

  def _ResetJointState(self):

    for i in range(self.numMotors):
      self._pybullet_client.resetJointState(
        self.kinovaUid,
        self._joint_name_to_id[self.motorNames[i]],
        self._init_jointPositions[i],
        targetVelocity=0)

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.kinovaUid,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle, max_velocity = None):
    if max_velocity is None:
        max_velocity = self.jointMaxVelocity[motor_id]
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.kinovaUid,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.POSITION_CONTROL,
        targetPosition=desired_angle,
        positionGain=0.3,
        velocityGain=1,
        maxVelocity = max_velocity,
        force=self.maxForce)


  def GetActionDimension(self):
    if (self.useInverseKinematics):
      return 5  #position x,y,z angle and finger angle
    return len(self.motorIndices)

  def GetObservationDimension(self):
    return len(self.GetObservation())

  def _AddSensorNoise(self, sensor_values, noise_stdev):
    if noise_stdev <= 0:
      return sensor_values
    observation = sensor_values + np.random.normal( scale=noise_stdev, size=sensor_values.shape)
    return observation

  def GetTrueMotorAngles(self):
    """Gets the joints angles at the current moment.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.kinovaUid, motor_id)[0]
        for motor_id in self.motorIndices]
    return motor_angles

  def GetMotorAngles(self):
    """Gets the actual joint angles with noise.
    This function mimicks the noisy sensor reading and adds latency. The motor
    angles that are delayed, noise polluted, and mapped to [-pi, pi].
    Returns:
      Motor angles polluted by noise and latency, mapped to [-pi, pi].
    """
    motor_angles = self._AddSensorNoise(np.array(self.GetTrueMotorAngles()[0:self.numMotors]),
        self._observation_noise_stdev[0])
    return motor_angles  # delete maping

  def GetTrueMotorVelocities(self):
    """Get the velocity of all joints.
    Returns:
      Velocities of all joints.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.kinovaUid, motor_id)[1]
        for motor_id in self.motorIndices]

    return motor_velocities

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Velocities of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array( self.GetTrueMotorVelocities()[0:self.numMotors]),
        self._observation_noise_stdev[1])

  def GetTrueMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    #TODO if considering motor dynamics, need to add a coversion function of motor torques.
    motor_torques = [
          self._pybullet_client.getJointState(self.kinovaUid, motor_id)[3]
          for motor_id in self.motorIndices ]

    return motor_torques
  def GetMotorTorques(self):
    """Get the amount of torque the motors are exerting.
    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Motor torques of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise( np.array( self.GetTrueMotorTorques()[0: self.numMotors]),
                                 self._observation_noise_stdev[2])

  def GetObservation(self):
    """Get the observations of minitaur.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []

    if self.OnlyEndEffectorObervations:
      state = self._pybullet_client.getLinkState(self.kinovaUid,
                                                 self.EndEffectorIndex)
      ee_pos = state[4]
      ee_orn = state[5]
      ee_euler = self._pybullet_client.getEulerFromQuaternion(ee_orn)

      observation.extend(list(ee_pos))
      observation.extend(list(ee_euler))
    else:
      observation.extend(self.GetMotorAngles().tolist())
      observation.extend(self.GetMotorVelocities().tolist())
      observation.extend(self.GetMotorTorques().tolist())

    return observation

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    if self.OnlyEndEffectorObervations:
      raise print('need to developed!')
    else:
      upper_bound = np.array([0.0] * self.GetObservationDimension())
      upper_bound[0:self.numMotors] = self.jointUpperLimit  # Joint angle.
      upper_bound[self.numMotors:2 * self.numMotors] = self.jointMaxVelocity  #    Joint velocity.
      upper_bound[2 * self.numMotors:3 * self.numMotors] = self.jointMaxForce # Joint torque.

    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    if self.OnlyEndEffectorObervations:
      raise print('need to developed!')
    else:
      lower_bound = np.array([0.0] * self.GetObservationDimension())
      lower_bound[0:self.numMotors] = self.jointLowerLimit  # Joint angle.
      lower_bound[self.numMotors:2 * self.numMotors] = self.jointMaxVelocity*(-1)  # Joint velocity.
      lower_bound[2 * self.numMotors:3 * self.numMotors] = self.jointMaxForce*(-1)  # Joint torque.

    return lower_bound

  def ApplyAction(self, motorCommands):

    #print ("self.numJoints")
    #print (self.numJoints)
    if (self.useInverseKinematics):
      
      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]
      
      # EndEffectorStates = self._pybullet_client.getLinkState(self.kinovaUid, self.EndEffectorIndex)
      # actualEndEffectorPos = EndEffectorStates[4]  # pos in world frame
      # actualEndEffectorOrn = EndEffectorStates[5]  # orientation in world frame

      #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
      #print(actualEndEffectorPos[2])



      self.endEffectorPos[0] = self.endEffectorPos[0]+dx
      if (self.endEffectorPos[0]>self.ee_X_upperLimit):
        self.endEffectorPos[0]=self.ee_X_upperLimit
      if (self.endEffectorPos[0]< self.ee_X_lowerLimit):
        self.endEffectorPos[0]= self.ee_X_lowerLimit

      self.endEffectorPos[1] = self.endEffectorPos[1]+dy
      if (self.endEffectorPos[1] < self.ee_Y_lowerLimit):
        self.endEffectorPos[1] = self.ee_Y_lowerLimit
      if (self.endEffectorPos[1] > self.ee_Y_upperLimit):
        self.endEffectorPos[1] = self.ee_Y_upperLimit
      
      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      self.endEffectorPos[2] = self.endEffectorPos[2]+dz

      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = self._pybullet_client.getQuaternionFromEuler([0,-math.pi, self.endEffectorAngle]) # -math.pi,yaw])

      if (self.useNullSpace==1):
        if (self.useOrientation==1):
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.kinovaUid,self.EndEffectorIndex,pos,orn,self.ll,self.ul,self.jr,self.rp)
        else:
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.kinovaUid,self.EndEffectorIndex,pos,lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
      else:
        if (self.useOrientation==1):
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.kinovaUid,self.EndEffectorIndex,pos,orn,jointDamping=self.jd)
        else:
          jointPoses = self._pybullet_client.calculateInverseKinematics(self.kinovaUid,self.EndEffectorIndex,pos)

      #print("jointPoses")
      #print(jointPoses)
      #print("self.kukaEndEffectorIndex")
      #print(self.kukaEndEffectorIndex)
      if (self.useSimulation):
        for i in range(self.numMotors - self.numFingers):
          motor_id = self.motorIndices[i]
          self._SetDesiredMotorAngleById(motor_id, jointPoses[i])
      else:
        #TODO test
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range (self.numJoints):
          self._pybullet_client.resetJointState(self.kukaUid,i,jointPoses[i])
      #fingers
      for i in range(self.numFingers):
        finger_id = self.motorIndices[self.numMotors - self.numFingers + i]
        self._pybullet_client.setJointMotorControl2(self.kinovaUid,finger_id,self._pybullet_client.POSITION_CONTROL,targetPosition= fingerAngle,  force=self.fingerTipForce)

    else:
      assert np.array(motorCommands).size == self.numMotors
      for i in range (self.numMotors):

        motor_id = self.motorIndices[i]
        self._SetDesiredMotorAngleById(motor_id, motorCommands[i])
      
