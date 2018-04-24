# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.toy_text.frozen_lake import FrozenLakeEnv as _FrozenLakeEnv
from gym.envs.mujoco.hopper import HopperEnv as _HopperEnv
from gym import spaces
from gym.wrappers.time_limit import TimeLimit

import numpy as np


def get_env(env_name, safety_param=0):
    # Reset reward should be in [-1, 0]
    if env_name in ['small-gridworld', 'large-gridworld']:
        if env_name == 'small-gridworld':
            map_name = '4x4'
            max_episode_steps = 30
            num_training_iterations = 6000
        else:
            map_name = '8x8'
            max_episode_steps = 100
            num_training_iterations = 20000
        env = FrozenLakeEnv(map_name=map_name)
        done_state = np.zeros(env.nS)
        done_state[0] = 1

        def reset_done_fn(s):
            return np.all(s == done_state)

        def reset_reward_fn(s):
            float(reset_done_fn(s)) - 1.0

        agent_type = 'DDQNAgent'
    elif env_name == 'hopper':
        env = HopperEnv()

        def reset_done_fn(s):
            height = s[0]
            ang = s[1]
            return (height > .7) and (abs(ang) < .2)

        def reset_reward_fn(s):
            return float(reset_done_fn(s)) - 1.0

        agent_type = 'DDPGAgent'
        max_episode_steps = 1000
        num_training_iterations = 1000000

    else:
        raise ValueError('Unknown environment: %s' % env_name)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    q_min = -1 * (1. - safety_param) * env._max_episode_steps
    lnt_params = {
        'reset_reward_fn': reset_reward_fn,
        'reset_done_fn': reset_done_fn,
        'q_min': q_min,
    }
    agent_params = {
        'num_training_iterations': num_training_iterations,
        'agent_type': agent_type,
    }
    return (env, lnt_params, agent_params)


class FrozenLakeEnv(_FrozenLakeEnv):
    """Modified version of FrozenLake-v0.

    1. Convert integer states to one hot encoding.
    2. Make the goal state reversible
    """
    def __init__(self, map_name):
        super(FrozenLakeEnv, self).__init__(map_name=map_name,
                                            is_slippery=False)
        self.observation_space = spaces.Box(low=np.zeros(self.nS),
                                            high=np.ones(self.nS))
        # Make the goal state not terminate
        goal_s = self.nS - 1
        left_s = goal_s - 1
        up_s = goal_s - int(np.sqrt(self.nS))

        self.P[goal_s] = {
            0: [(1.0, left_s, 0.0, False)],
            1: [(1.0, goal_s, 1.0, True)],
            2: [(1.0, goal_s, 1.0, True)],
            3: [(1.0, up_s, 0.0, True)],
        }

    def _s_to_one_hot(self, s):
        one_hot = np.zeros(self.nS)
        one_hot[s] = 1.
        return one_hot

    def step(self, a):
        (s, r, done, info) = super(FrozenLakeEnv, self).step(a)
        done = (s == self.nS - 1)  # Assume we can't detect dangerous states
        one_hot = self._s_to_one_hot(s)
        r -= 1  # Make the reward be in {-1, 0}
        return (one_hot, r, done, info)

    def reset(self):
        s = super(FrozenLakeEnv, self).reset()
        one_hot = self._s_to_one_hot(s)
        return one_hot


class HopperEnv(_HopperEnv):
    """Modified version of Hopper-v1."""

    def step(self, action):
        (obs, r, done, info) = super(HopperEnv, self).step(action)
        return (obs, r, False, info)
