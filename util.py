from lnt import MetricsWrapper
from coach.agents import _DQNAgent, _DDPGAgent
from coach.configurations import Preset, DQN, GymVectorObservation, ExplorationParameters
import tensorflow as tf

def get_env(env_name, safety_param):
    env = MetricsWrapper(gym.make(env_name))
    if env_name == 'FrozenLake-v0':
        reset_reward_fn = lambda s: float(s == 0)
        reset_done_fn = lambda s: s == 0
        max_reset_reward = 1.0
    else:  # TODO: add more environments
        raise ValueError('Unknown environment: %s' % env_name)
    # Copy the max steps into environment metadata so we can still
    # access it after removing the environment wrapper.
    env.metadata['max_episode_steps'] = env.spec.max_episode_steps

    q_min = safety_param * max_reset_reward * env.spec.max_episode_steps
    env_params = {
        'reset_reward_fn': reset_reward_fn,
        'reset_done_fn': reset_done_fn,
        'max_episode_steps': env.spec.max_episode_steps,
        'q_min': q_min,
    }
    return (env, env_params)

# Overwrite the agents to automatically use the default parameters
class DQNAgent(_DQNAgent):

    def __init__(env):
        tuning_params = Preset(agent=DQN, env=GymVectorObservation, exploration=ExplorationParameters)
        tuning_params.sess = tf.Session()  # TODO: check for collisions between graphs, sessions of multiple agents
        tuning_params.agent.discount = 1.0 - 1.0 / env.metadata['max_episode_steps']
        print('tuning_parameters', tuning_params)

        # Single-thread runs
        tuning_params.task_index = 0

        super(DQNAgent, self).__init__(env, tuning_params)

class DDPGAgent(_DDPGAgent):

    def __init__(env):
        tuning_params = None  # get these values
        super(DDPGAgent, self).__init__(env, tuning_params)
