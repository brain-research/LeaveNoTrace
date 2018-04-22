from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit

from coach.memories.memory import Transition

class SafetyWrapper(Wrapper):
    # TODO: allow user to specify number of reset attempst. Currently fixed at 1.
    def __init__(self, env, reset_agent, reset_reward_fn, reset_done_fn, max_episode_steps, q_min):
        # Remove the time limit wrapper
        if isinstance(env, TimeLimit):
            env = env.env
        super(SafetyWrapper, self).__init__(self, env)
        self._reset_agent = reset_agent
        self._reset_reward_fn = reset_reward_fn
        self._reset_done_fn = reset_done_fn
        self._max_episode_steps = max_episode_steps
        self._q_min = q_min
        self._obs = None

    def _reset(self):
        obs = self._obs
        if self._reset_done_fn(obs):
            return obs
        for t in range(max_episode_steps):
            reset_action = self._reset_agent.choose_action(obs)
            (next_obs, r, done, info) = self.env.step(reset_action)
            reset_reward = self._reset_reward_fn(next_obs)
            reset_done = self._is_reset_fn(next_obs)
            transition = Transition(obs, reset_action, reset_reward, next_obs, reset_done)
            self._reset_agent.memory.store(transition)
            loss = self._reset_agent.train()  # Do one training iteration of the reset agent
            obs = next_obs
            if reset_done:
                break
        if not reset_done:
            obs = self.env.reset()
        return (obs, r, done, info)

    def reset(self):
        '''Return only the current state for external calls to reset.'''
        (obs, r, done, info) = self._reset()
        return obs
        

    def step(self, action):
        inputs = {'actions': action[None, :], 'states': self._obs[None, :]}
        reset_q = self._reset_agent.main_network.target_network.predict(inputs)
        if reset_q < q_min:
            (obs, r, done, info) = self._reset()
        else:
            (obs, r, done, info) = self.env.step(action)
        self._obs = obs
        return (obs, r, done, info)


class MetricsWrapper(Wrapper):

    def __init__(self, env):
        super(MetricsWrapper, self).__init__(self, env)
        self._total_steps = 0  # Total steps taken during training
        self._total_resets = 0  # Total resets taken during training
        self._total_reward = 0  # Cumulative reward for the current episode
        self._reset_history = []
        self._reward_history = []

    def step(self, action):
        (obs, r, done, info) = self.env.step(action)
        self._total_steps += 1
        self._total_reward += r
        if done:
            self._reset_history.append((self._total_steps, self._total_resets))
            self._reward_history.append((self._total_steps, self._total_reward))
            self._total_reward = 0  # Reset the cumulative reward to 0 for the next episode
        return (obs, r, done, info)

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        import numpy as np
        reward_history = np.array(self._reward_history)
        reset_history = np.array(self._reset_history)
        plt.figure(figsize=(8, 6))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        ax1.plot(reward_history[:, 0], reward_history[:, 1], 'gx', label='reward')
        ax2.plot(reset_history[:, 0], reset_history[:, 1], 'bo', label='num. resets')
        plt.legend()
        plt.show()


class AssertNeverTouchedWrapper(Wrapper):
    # TODO: delete this once we're sure the environment is not touched

    def reset(self):
        assert False, 'Never call reset() on AssertNeverTouchedWrapper'

    def step(self, action):
        assert False, 'Never call step() on AssertNeverTouchedWrapper'
