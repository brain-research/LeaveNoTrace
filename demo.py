from lnt import SafetyWrapper
from util import get_env, DQNAgent, DDPGAgent, DDQNAgent
import argparse
import tensorflow as tf


def learn_safely(env_name, safety_param):
    (env, env_params) = get_env(env_name, safety_param)

    # 1. Create a reset agent that will reset the environment
    reset_agent = DDQNAgent(env=env, name='reset_agent')  # TODO: don't hard code learning alg

    # 2. Create a wrapper around the environment to protect it
    safe_env = SafetyWrapper(env=env, reset_agent=reset_agent, **env_params)

    # 3. Safely learn to solve the task.
    DDQNAgent(env=safe_env, name='forward_agent').improve()

    # Plot the reward and resets throughout training
    safe_env.plot_metrics()

def learn_dangerously(env_name):
    (env, _) = get_env(env_name, 0)
    DDQNAgent(env=env, name='agent').improve()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Let\'s do safe RL with Leave No Trace!')
    parser.add_argument('--env_name', type=str, default='FrozenLake-v0',
                        help=('Name of the Gym environment. The currently supported '
                              'environments are: FrozenLake-v0'))
    parser.add_argument('--safety_param', type=float, default=0.2,
                        help=('Increasing the safety_param from 0 to 1 makes the '
                              'agent safer. A reasonable value is 0.2'))
    args = parser.parse_args()
    assert 0 < args.safety_param < 1, 'The safety_param should be between 0 and 1.'
    # learn_dangerously(args.env_name)
    learn_safely(args.env_name, args.safety_param)
