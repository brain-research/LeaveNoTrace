import sys
import os
sys.path.append(os.path.join(sys.path[0], 'coach'))

from coach.agents import DQNAgent as _DQNAgent
from coach.agents import DDQNAgent as _DDQNAgent
from coach.agents import DDPGAgent as _DDPGAgent
from coach.configurations import Preset, DQN, DDPG, GymVectorObservation, ExplorationParameters, OUExploration
from coach.memories.memory import Transition

from coach.utils import RunPhase
from coach.environments.gym_environment_wrapper import GymEnvironmentWrapper
from coach.environments.environment_wrapper import EnvironmentWrapper
import tensorflow as tf



def Agent(env, **kwargs):
    agent_type = kwargs.pop('agent_type')
    if agent_type == 'DDQNAgent':
        return DDQNAgent(env, **kwargs)
    elif agent_type == 'DDPGAgent':
        return DDPGAgent(env, **kwargs)
    else:
        raise ValueError('Unknown agent_type: %s' % agent_type)

# Overwrite the agents to automatically use the default parameters
class DDQNAgent(_DDQNAgent):

    def __init__(self, env, name, num_training_iterations=10000):
        tuning_params = Preset(agent=DQN, env=GymVectorObservation, exploration=ExplorationParameters)
        tuning_params.sess = tf.Session()
        tuning_params.agent.discount = 0.99
        tuning_params.visualization.dump_csv = False
        tuning_params.num_training_iterations = num_training_iterations
        tuning_params.num_heatup_steps = env._max_episode_steps * tuning_params.batch_size
        tuning_params.exploration.epsilon_decay_steps = 0.66 * num_training_iterations
        env = GymEnvironmentWrapper(tuning_params, env)
        super(DDQNAgent, self).__init__(env, tuning_params, name=name)

    def get_q(self, obs, action):
        inputs = {'observation': obs[None, :, None]}
        outputs = self.main_network.target_network.predict(inputs)
        return outputs[0, action]


class DDPGAgent(_DDPGAgent):

    def __init__(self, env, name, num_training_iterations=1000000):
        tuning_params = Preset(agent=DDPG, env=GymVectorObservation, exploration=OUExploration)
        tuning_params.sess = tf.Session()
        tuning_params.agent.discount = 0.999
        tuning_params.visualization.dump_csv = False
        tuning_params.num_training_iterations = num_training_iterations
        env = GymEnvironmentWrapper(tuning_params, env)
        super(DDPGAgent, self).__init__(env, tuning_params, name=name)

    def get_q(self, obs, action):
        inputs = {'observation': obs[None, :, None],
                  'action': action[None, :]}
        outputs = self.main_network.target_network.predict(inputs)
        return outputs[0, 0]
