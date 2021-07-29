# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,/
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
import numpy as np
import random
from random import randrange
from dm_control import viewer


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_ACTION=['B0_0','B0_5','B5_0','B5_5']
CORNER_INDEX_POSITION=['G0_0','G0_5','G5_0','G5_5']
INDEX_ACTION=['B0_0','B0_1','B0_2','B0_3','B0_4','B0_5','B1_0','B2_0','B3_0','B4_0','B5_0','B1_1','B1_2','B1_3','B1_4','B1_5','B2_1','B2_2','B2_3','B2_4','B2_5','B3_1','B3_2','B3_3','B3_4','B3_5','B4_1','B4_2','B4_3','B4_4','B4_5','B5_1','B5_2','B5_3','B5_4','B5_5']
INDEX_POSITION=['G0_0','G0_1','G0_2','G0_3','G0_4','G0_5','G1_0','G2_0','G3_0','G4_0','G5_0','G1_1','G1_2','G1_3','G1_4','G1_5','G2_1','G2_2','G2_3','G2_4','G2_5','G3_1','G3_2','G3_3','G3_4','G3_5','G4_1','G4_2','G4_3','G4_4','G4_5','G5_1','G5_2','G5_3','G5_4','G5_5']
SEAM_INDEX_ACTION=['B0_0','B0_1','B0_2']
SEAM_INDEX_POSITION=['G0_0','G0_1','G0_2']
INDEX_NONSEAM_ACTION=['B0_4','B0_5','B3_0','B4_0','B5_0','B1_4','B1_5','B2_4','B2_5','B3_1','B3_2','B3_3','B3_4','B3_5','B4_1','B4_2','B4_3','B4_4','B4_5','B5_1','B5_2','B5_3','B5_4','B5_5']
INDEX_NONSEAM_POSITION=['G0_3','G0_4','G0_5','G3_0','G4_0','G5_0','G1_3','G1_4','G1_5','G2_3','G2_4','G2_5','G3_1','G3_2','G3_3','G3_4','G3_5','G4_1','G4_2','G4_3','G4_4','G4_5','G5_1','G5_2','G5_3','G5_4','G5_5']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_sewts_intermediate.xml'),common.ASSETS

W=64

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit,**environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

class Cloth(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, random_location=True,
               pixels_only=False, maxq=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._random_location = random_location
    self._pixels_only = pixels_only

    self._maxq = maxq

    print('random_location', self._random_location, 'pixels_only', self._pixels_only, 'maxq', self._maxq)

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    # (3 is one hot encoding for seam point selection, x,y,z for three seam points)
    return specs.BoundedArray(
          shape=(7,), dtype=np.float, minimum=[-1.0] * 7 , maximum=[1.0] * 7)
  def initialize_episode(self,physics):

    physics.named.data.xfrc_applied['B2_3', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B3_3', :3] = np.array([0,0,-2])
    
    for i in range(0,20):
        point = randrange(0,len(INDEX_ACTION))
        corner = randrange(0,len(CORNER_INDEX_ACTION))
        print(INDEX_ACTION[point])

        physics.named.data.xfrc_applied[INDEX_ACTION[point],:3] = np.random.uniform(-.5,.5,size=3) * 2
        physics.step()

        physics.named.data.xfrc_applied[CORNER_INDEX_ACTION[corner],:3] = np.random.uniform(-1,1,size=3) * 2
        physics.step()
    
    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #   Support legacy internal code.

      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))

      one_hot = action[:4] # selection of one of the 4 corners using one hot encoding, e.g. [0,1,0,0] for corner 2
      index = np.argmax(one_hot) # selects the corner from encoding, e.g. for [0,0,0,1], position 3 has max value and hence index for corner is 3, i.e. last corner
      print("Corner")
      print(index+1,"\n")
      action = action[4:] # ultimately action is just of size 3 , i.e. x,y,z for pick position

      goal_position = action * 0.05 * 0.5 # multiplying by 0.05 for right scaling perhaps
      # action is a numpy array of size 4 and the next three terms are just x,y,z position of the particle
      corner_action = CORNER_INDEX_ACTION[index]
      corner_geom = CORNER_INDEX_POSITION[index]

      # apply consecutive force to move the point to the target position
      position = goal_position + physics.named.data.geom_xpos[corner_geom]
      # geom_xpos gives the x, y, z position of the selected corner point
      # add the movement of the corner given by goal_position (scaled action) to the current corner position
      dist = position - physics.named.data.geom_xpos[corner_geom]
      # distance movement
      loop = 0
      # while Frobenius norm or 2-norm is greater than a certain value
      # Frobenius norm is just taking squares of each of the term, adding them and taking a square root
      # https://www.youtube.com/watch?v=yiSKsLcniGw
      # https://youtu.be/Gt56YxMBlVA
      # We are taking this because we want to ensure that the distance moved is significant, i.w. freater than a certain minimum amount
      # and we represent it using this term
      while np.linalg.norm(dist) > 0.0025:
        loop += 1
        if loop > 40:
        # why loop ?
          break
        physics.named.data.xfrc_applied[corner_action, :3] = dist * 20
        # Force applied on the corner proportional to the distance to be moved, i.e. more force for more distance
        physics.step()
        # get observations after applying actions, new position perhaps
        self.after_step(physics)
        dist = position - physics.named.data.geom_xpos[corner_geom]      
      
      
  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = []
    obs_ = []
    for i in range(len(INDEX_POSITION)):
        obs_.append(physics.named.data.geom_xpos[INDEX_POSITION[i]])
    obs_ = np.array(obs_)
    obs['position'] = obs_.reshape(-1).astype('float32')
    # obs.append(physics.named.data.geom_xpos['G0_0'].reshape(-1).astype('float32'))
    # print(obs)

    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    
    # DEFINING THE CURRENT POSITIONS OF 4 POINTS, G0_0, G0_1, G0_2, G0_3
    
    current_pos_G0_0 = physics.named.data.geom_xpos['G0_0']
    current_pos_G0_1 = physics.named.data.geom_xpos['G0_1']
    current_pos_G0_2 = physics.named.data.geom_xpos['G0_2']
    
    # Defining the radius
    
    reward1 = 0
    reward2 = 0
    reward3 = 0
    reward4 = 0

    # Rewarding if seam points are on ground
    
    if current_pos_G0_0[2] <= 0.01:
        reward1 += 0.1
    if current_pos_G0_1[2] <= 0.01:
        reward1 += 0.1
    if current_pos_G0_2[2] <= 0.01:
        reward1 += 0.1
    print("Stage 1 reward :", reward1)
    # Rewarding if seam points form a line
    # Taking Corner point x,y coordinates, G0_0
    
    x_G0_0 = current_pos_G0_0[0] 
    y_G0_0 = current_pos_G0_0[1]
    
    # Taking the point next to corner, G0_1
    
    x_G0_1 = current_pos_G0_1[0]
    y_G0_1 = current_pos_G0_1[1]
    
    # Taking the point second to corner, G0_2 
    
    x_G0_2 = current_pos_G0_2[0]
    y_G0_2 = current_pos_G0_2[1]
    if abs(x_G0_1 - x_G0_0) > 0.01:
        slope_G00_G01 =(y_G0_1 - y_G0_0) / (x_G0_1 - x_G0_0)
    else:
        slope_G00_G01 = 10 * np.sign(y_G0_1 - y_G0_0)

    if abs(x_G0_2 - x_G0_1) > 0.01:
        slope_G01_G02 = (y_G0_2- y_G0_1) / (x_G0_2 - x_G0_1)    
    else:
        slope_G01_G02 = 10 * np.sign(y_G0_2 - y_G0_1)

    if abs(x_G0_2 - x_G0_0) > 0.01:
        slope_G00_G02 = (y_G0_2- y_G0_0) / (x_G0_2 - x_G0_0)
    else:
        slope_G00_G02 = 10 * np.sign(y_G0_2 - y_G0_0)

    if ((abs(abs(slope_G00_G01) - abs(slope_G01_G02))) < 0.5 and (abs(abs(slope_G00_G02) - abs(slope_G01_G02))) < 0.5):
        print("1 : ", abs(slope_G00_G01) - abs(slope_G01_G02))
        print("2 : ", abs(slope_G00_G02) - abs(slope_G01_G02))
        reward2 += 0.35
    print("slope_G00_G01 :", slope_G00_G01)      
    print("slope_G01_G02 :", slope_G01_G02)      
    print("slope_G00_G02 :", slope_G00_G02)        
    print("Stage 2 reward :", reward2)      
    # Rewarding if neighbouring particles are also on ground   
    
    current_pos_G1_0 = physics.named.data.geom_xpos['G1_0']
    current_pos_G1_1 = physics.named.data.geom_xpos['G1_1']
    current_pos_G1_2 = physics.named.data.geom_xpos['G1_2']
    
    # Rewarding if seam points are on ground

    if current_pos_G1_0[2] <= 0.01:
        reward3 += 0.05

    if current_pos_G1_1[2] <= 0.01:
        reward3 += 0.05

    if current_pos_G1_2[2] <= 0.01:
        reward3 += 0.05
    print("Stage 3 reward :", reward3)
    # DEFINE REGION OF NON-OVERLAP. AROUND THE SEAM AREA THERE SHOULD BE NO OTHER CLOTH PORTION ON TOP

    x_min = current_pos_G0_0[0] - 0.06
    x_max = current_pos_G0_0[0] + 0.06

    y_min = current_pos_G0_0[1] - 0.06
    y_max = current_pos_G0_0[1] + 0.06
    
    for x in range(len(INDEX_NONSEAM_POSITION)):
        if physics.named.data.geom_xpos[INDEX_NONSEAM_POSITION[x]][0] < x_min or physics.named.data.geom_xpos[INDEX_NONSEAM_POSITION[x]][0] > x_max or physics.named.data.geom_xpos[INDEX_NONSEAM_POSITION[x]][1] > y_max or physics.named.data.geom_xpos[INDEX_NONSEAM_POSITION[x]][1] < y_min: 
            reward4 += 0.007407 # 0.2/27
    print("Stage 4 reward :", reward4)       
    reward = reward1 + reward2 + reward3 + reward4
    return reward
