from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit

from util import Transition
import traceback
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


        self._total_steps = 0  # Total steps taken during training
        self._total_resets = 0  # Total resets taken during training
        self._total_reward = 0  # Cumulative reward for the current episode
        self._reset_history = []
        self._reward_history = []

    def _reset(self):
        obs = self._obs
        # TODO: handle case where agent is already reset
        for t in range(self._max_episode_steps):
            # TODO: investigate what extra stuff is returned here
            (reset_action, _) = self._reset_agent.choose_action({'observation': obs[:, None]})
            (next_obs, r, _, info) = self.env.step(reset_action)
            self._total_steps += 1
            reset_reward = self._reset_reward_fn(next_obs)
            reset_done = self._reset_done_fn(next_obs)
            # print('Reset transition:', np.argmax(obs), np.argmax(next_obs),
            #       reset_action, reset_reward, reset_done)
            transition = Transition({'observation': obs[:, None]}, reset_action,
                                    reset_reward, {'observation': next_obs[:, None]},
                                    reset_done)
            self._reset_agent.memory.store(transition)
            obs = next_obs
            if self._reset_agent.memory.num_transitions_in_complete_episodes() > self._reset_agent.tp.batch_size:
                loss = self._reset_agent.train()  # Do one training iteration of the reset agent
        if not reset_done:
            obs = self.env.reset()
            self._total_resets += 1

        # Log metrics
        self._reward_history.append((self._total_steps, self._total_reward))
        self._reset_history.append((self._total_steps, self._total_resets))
        self._total_reward = 0

        done = False
        # Reset the elapsed steps back to 0
        self.env._elapsed_steps = 0
        return (obs, r, done, info)

    def reset(self):
        '''Return only the current state for external calls to reset.'''
        (obs, r, done, info) = self._reset()
        return obs

    def step(self, action):
        inputs = {'observation': self._obs[None, :, None]}
        outputs = self._reset_agent.main_network.target_network.predict(inputs)
        reset_q = outputs[0, action]
        # print('Safety check: %.2f vs %.2f (%s, %s)' % (reset_q, self._q_min, np.argmax(self._obs), action))
        if reset_q < self._q_min:
            # print('Early abort:', np.argmax(self._obs))
            (obs, r, done, info) = self._reset()
        else:
            (obs, r, done, info) = self.env.step(action)
            self._total_steps += 1
            self._total_reward += r
        self._obs = obs
        return (obs, r, done, info)

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        reward_history = np.array(self._reward_history)
        reset_history = np.array(self._reset_history)
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        ax1.plot(reward_history[:, 0], reward_history[:, 1], 'g.')
        ax2.plot(reset_history[:, 0], reset_history[:, 1], 'b-')

        ax1.set_ylabel('reward', color='g', fontsize=20)
        ax1.tick_params('y', colors='g')
        ax2.set_ylabel('num. resets', color='b', fontsize=20)
        ax2.tick_params('y', colors='b')
        ax1.set_xlabel('num. steps', fontsize=20)

        plt.show()


# class MetricsWrapper(Wrapper):
# 
#     def __init__(self, env):
#         super(MetricsWrapper, self).__init__(env)
#         self._total_steps = 0  # Total steps taken during training
#         self._total_resets = 0  # Total resets taken during training
#         self._total_reward = 0  # Cumulative reward for the current episode
#         self._reset_history = []
#         self._reward_history = []
# 
#     def step(self, action):
#         (obs, r, done, info) = self.env.step(action)
#         self._total_steps += 1
#         self._total_reward += r
#         if done:
#             self._reset_history.append((self._total_steps, self._total_resets))
#             self._reward_history.append((self._total_steps, self._total_reward))
#             self._total_reward = 0  # Reset the cumulative reward to 0 for the next episode
#         return (obs, r, done, info)
# 
#     def reset(self):
#         self._total_resets += 1
#         return self.env.reset()
# 
#     def plot_metrics(self):
#         import matplotlib.pyplot as plt
#         import numpy as np
#         reward_history = np.array(self._reward_history)
#         reset_history = np.array(self._reset_history)
#         fig = plt.figure(figsize=(8, 6))
#         ax1 = fig.gca()
#         ax2 = ax1.twinx()
#         ax1.plot(reward_history[:, 0], reward_history[:, 1], 'g.')
#         ax2.plot(reset_history[:, 0], reset_history[:, 1], 'b-')
# 
#         ax1.set_ylabel('reward', color='g', fontsize=20)
#         ax1.tick_params('y', colors='g')
#         ax2.set_ylabel('num. resets', color='b', fontsize=20)
#         ax2.tick_params('y', colors='b')
#         ax1.set_xlabel('num. steps', fontsize=20)
# 
#         plt.show()
