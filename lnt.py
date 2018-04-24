from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit

from coach_util import Transition, RunPhase
import numpy as np


class SafetyWrapper(Wrapper):
    # TODO: allow user to specify number of reset attempst. Currently fixed at 1.
    def __init__(self, env, reset_agent, reset_reward_fn, reset_done_fn, q_min):
        assert isinstance(env, TimeLimit)
        super(SafetyWrapper, self).__init__(env)
        self._reset_agent = reset_agent
        self._reset_reward_fn = reset_reward_fn
        self._reset_done_fn = reset_done_fn
        self._max_episode_steps = env._max_episode_steps
        self._q_min = q_min
        self._obs = env.reset()  # TODO: describe that we do one reset here


        self._total_resets = 0  # Total resets taken during training
        self._episode_rewards = []  # Rewards for the current episode
        self._reset_history = []
        self._reward_history = []

        self._reset_agent.exploration_policy.change_phase(RunPhase.TRAIN)

    def _reset(self):
        obs = self._obs
        # TODO: handle case where agent is already reset
        obs_vec = [np.argmax(obs)]
        for t in range(self._max_episode_steps):
            # TODO: investigate what extra stuff is returned here
            (reset_action, _) = self._reset_agent.choose_action({'observation': obs[:, None]}, phase=RunPhase.TRAIN)
            (next_obs, r, _, info) = self.env.step(reset_action)
            reset_reward = self._reset_reward_fn(next_obs)
            reset_done = self._reset_done_fn(next_obs)
            # print('Reset transition:', np.argmax(obs), np.argmax(next_obs),
            #       reset_action, reset_reward)
            transition = Transition({'observation': obs[:, None]}, reset_action,
                                    reset_reward, {'observation': next_obs[:, None]},
                                    reset_done)
            self._reset_agent.memory.store(transition)
            obs = next_obs
            obs_vec.append(np.argmax(obs))
            if self._reset_agent.memory.num_transitions_in_complete_episodes() > self._reset_agent.tp.batch_size:
                loss = self._reset_agent.train()  # Do one training iteration of the reset agent
                # print('Reset agent Bellman error: %f' % loss)
            if reset_done:
                break
        if not reset_done:
            # print('Failed reset:', np.argmax(obs))
            obs = self.env.reset()
            self._total_resets += 1

        print(obs_vec)
        # print('Reset agent exploration:', self._reset_agent.exploration_policy.get_control_param())

        ### Log metrics
        self._reset_history.append(self._total_resets)
        self._reward_history.append(np.mean(self._episode_rewards))
        self._episode_rewards = []

        done = False  # If the agent takes an action that causes an early abort
                      # the agent shouldn't believe that the episode terminates.
                      # Because the reward is negative, the agent would be
                      # incentivized to do early aborts as quickly as possible.
        # Reset the elapsed steps back to 0
        self.env._elapsed_steps = 0
        return (obs, r, done, info)

    def reset(self):
        '''Return only the current state for external calls to reset.'''
        (obs, r, done, info) = self._reset()
        return obs

    def step(self, action):
        reset_q = self._reset_agent.get_q(self._obs, action)
        # print('Safety check: %.2f vs %.2f (%s, %s)' % (reset_q, self._q_min, np.argmax(self._obs), action))
        if reset_q < self._q_min:
            # print('Early abort:', np.argmax(self._obs))
            (obs, r, _, info) = self._reset()
            done = False  # Treat early aborts as episode termination
        else:
            (obs, r, done, info) = self.env.step(action)
            self._episode_rewards.append(r)
        self._obs = obs
        return (obs, r, done, info)

    def plot_metrics(self, output_dir):
        import matplotlib.pyplot as plt
        import json
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data = {
            'reward_history': self._reward_history,
            'reset_history': self._reset_history
        }
        with open(os.path.join(output_dir, 'data.json'), 'w') as f:
            json.dump(data, f)
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        rewards = np.array(self._reward_history)
        lnt_resets = np.array(self._reset_history)
        num_episodes = len(rewards)
        baseline_resets = np.arange(num_episodes)
        episodes = np.arange(num_episodes)
        ax1.plot(episodes, rewards, 'g.')
        ax2.plot(episodes, lnt_resets, 'b-')
        ax2.plot(episodes, baseline_resets, 'b--')

        ax1.set_ylabel('reward', color='g', fontsize=20)
        ax1.tick_params('y', colors='g')
        ax2.set_ylabel('num. resets', color='b', fontsize=20)
        ax2.tick_params('y', colors='b')
        ax1.set_xlabel('num. episodes', fontsize=20)
        plt.savefig(os.path.join(output_dir, 'plot.png'))

        plt.show()
