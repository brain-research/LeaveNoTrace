import sys
import os
sys.path.append(os.path.join(sys.path[0], 'coach'))

from coach.agents import DQNAgent as _DQNAgent
from coach.agents import DDQNAgent as _DDQNAgent
from coach.agents import DDPGAgent as _DDPGAgent
from coach.configurations import Preset, DQN, GymVectorObservation, ExplorationParameters
from coach.memories.memory import Transition
import tensorflow as tf
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv as _FrozenLakeEnv
import numpy as np
from gym import spaces
from coach.environments.gym_environment_wrapper import GymEnvironmentWrapper as _GymEnvironmentWrapper
from coach.environments.environment_wrapper import EnvironmentWrapper
import gym
from gym.wrappers.time_limit import TimeLimit
from gym import Wrapper

def get_env(env_name, safety_param):
    if env_name == 'FrozenLake-v0':
        env = FrozenLakeEnv()
        done_state = np.zeros(env.nS)
        done_state[0] = 1
        reset_done_fn = lambda s: np.all(s == done_state)
        reset_reward_fn = lambda s: float(reset_done_fn(s))
        max_reset_reward = 1.0
        max_episode_steps = 30
    else:  # TODO: add more environments
        raise ValueError('Unknown environment: %s' % env_name)
    # Copy the max steps into environment metadata so we can still
    # access it after removing the environment wrapper.
    env = NeverDoneWrapper(env)  # In the real world, we don't when when agent breaks
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    q_min = safety_param * max_reset_reward * env._max_episode_steps
    env_params = {
        'reset_reward_fn': reset_reward_fn,
        'reset_done_fn': reset_done_fn,
        'q_min': q_min,
    }
    return (env, env_params)

# Overwrite the agents to automatically use the default parameters
class DQNAgent(_DQNAgent):

    def __init__(self, env, name):
        tuning_params = Preset(agent=DQN, env=GymVectorObservation, exploration=ExplorationParameters)
        tuning_params.sess = tf.Session()  # TODO: check for collisions between graphs, sessions of multiple agents
        tuning_params.agent.discount = 1.0 - 1.0 / env._max_episode_steps
        tuning_params.visualization.dump_csv = False
        tuning_params.num_training_iterations = 10
        # tuning_params.num_heatup_steps = 10 * tuning_params.batch_size
        # obs_shape = env.observation_space.shape
        # assert len(obs_shape) == 1, 'Observation space: %s' % str(obs_shape)
        # tuning_params.env.desired_observation_width = obs_shape[0]
        # tuning_params.env.desired_observation_height = 1  # TODO: DOES THIS MATTER?
        # tuning_params.env.action_space_size = env.action_space.n
        # print('tuning_parameters', tuning_params)

        # Single-thread runs
        tuning_params.task_index = 0

        env = GymEnvironmentWrapper(env, tuning_params)

        super(DQNAgent, self).__init__(env, tuning_params, name=name)


class DDQNAgent(_DDQNAgent):

    def __init__(self, env, name):
        tuning_params = Preset(agent=DQN, env=GymVectorObservation, exploration=ExplorationParameters)
        tuning_params.sess = tf.Session()  # TODO: check for collisions between graphs, sessions of multiple agents
        tuning_params.agent.discount = 1.0 - 1.0 / env._max_episode_steps
        tuning_params.visualization.dump_csv = False
        tuning_params.num_training_iterations = 5000
        tuning_params.num_heatup_steps = env._max_episode_steps * tuning_params.batch_size
        tuning_params.exploration.epsilon_decay_steps = 2000


        # Single-thread runs
        tuning_params.task_index = 0

        env = GymEnvironmentWrapper(env, tuning_params)

        super(DDQNAgent, self).__init__(env, tuning_params, name=name)


class DDPGAgent(_DDPGAgent):

    def __init__(self, env):
        tuning_params = None  # get these values
        super(DDPGAgent, self).__init__(env, tuning_params)

class FrozenLakeEnv(_FrozenLakeEnv):
    def __init__(self):
        super(FrozenLakeEnv, self).__init__(is_slippery=False)
        self.observation_space = spaces.Box(low=np.zeros(self.nS),
                                            high=np.ones(self.nS))

    def _s_to_one_hot(self, s):
        one_hot = np.zeros(self.nS)
        one_hot[s] = 1.
        return one_hot

    def step(self, a):
        (s, r, done, info) = super(FrozenLakeEnv, self).step(a)
        one_hot = self._s_to_one_hot(s)
        return (one_hot, r, done, info)

    def reset(self):
        s = super(FrozenLakeEnv, self).reset()
        one_hot = self._s_to_one_hot(s)
        return one_hot

class GymEnvironmentWrapper(_GymEnvironmentWrapper):
    def __init__(self, env, tuning_parameters):
        EnvironmentWrapper.__init__(self, tuning_parameters)

        self.env = env

        if self.seed is not None:
            self.env.seed(self.seed)

        # self.env_spec = gym.spec(self.env_id)
        self.env.frameskip = self.frame_skip
        self.discrete_controls = type(self.env.action_space) != gym.spaces.box.Box
        self.random_initialization_steps = 0
        self.state = self.reset(True)['state']
        print('state: %s' % str(self.state))

        # render
        if self.is_rendered:
            image = self.get_rendered_image()
            scale = 1
            if self.human_control:
                scale = 2
            self.renderer.create_screen(image.shape[1]*scale, image.shape[0]*scale)

        if isinstance(self.env.observation_space, gym.spaces.Dict):
            if 'observation' not in self.env.observation_space:
                raise ValueError((
                    'The gym environment provided {env_id} does not contain '
                    '"observation" in its observation space. For now this is '
                    'required. The environment does include the following '
                    'keys in its observation space: {keys}'
                ).format(
                    env_id=self.env_id,
                    keys=self.env.observation_space.keys(),
                ))

        # TODO: collect and store this as observation space instead
        self.is_state_type_image = len(self.state['observation'].shape) > 1
        if self.is_state_type_image:
            self.width = self.state['observation'].shape[1]
            self.height = self.state['observation'].shape[0]
        else:
            self.width = self.state['observation'].shape[0]

        # action space
        self.actions_description = {}
        if hasattr(self.env.unwrapped, 'get_action_meanings'):
            self.actions_description = self.env.unwrapped.get_action_meanings()
        if self.discrete_controls:
            self.action_space_size = self.env.action_space.n
            self.action_space_abs_range = 0
        else:
            self.action_space_size = self.env.action_space.shape[0]
            self.action_space_high = self.env.action_space.high
            self.action_space_low = self.env.action_space.low
            self.action_space_abs_range = np.maximum(np.abs(self.action_space_low), np.abs(self.action_space_high))
        self.actions = {i: i for i in range(self.action_space_size)}
        self.key_to_action = {}
        if hasattr(self.env.unwrapped, 'get_keys_to_action'):
            self.key_to_action = self.env.unwrapped.get_keys_to_action()

        # measurements
        if self.env.spec is not None:
            self.timestep_limit = self.env.spec.timestep_limit
        else:
            self.timestep_limit = None
        self.measurements_size = len(self.step(0)['info'].keys())
        self.random_initialization_steps = self.tp.env.random_initialization_steps

class NeverDoneWrapper(Wrapper):
    def step(self, action):
        (obs, r, done, info) = self.env.step(action)
        return (obs, r, False, info)

    def reset(self):
        return self.env.reset()
