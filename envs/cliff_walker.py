
@SUITE.add
def cliff(random=None, version=0, eval=False):
  if eval:
    model = common.read_model('walker.xml')
  else:
    model = common.read_model('walker_cliff.xml')
  physics = Physics.from_string(model, common.ASSETS)
  # Note: reward functions specified here are not actually used.
  task = PlanarWalkerCliff(reward_function=stand_sparse_reward,
                                 evaluation_function=stand_sparse_reward,
                                 start_standing=True,
                                 random=random,
                                 version=version,
                                 eval=eval)
  return control.Environment(physics, task, time_limit=float('inf'),
                             control_timestep=_CONTROL_TIMESTEP)

@SUITE.add
def multi_reward(random=None, version=0, frail=False):
  if version == 0:
    model = common.read_model('walker.xml')
  elif version == 77:
    model = common.read_model('walker_77.xml')
  else:
    raise ValueError('Unknown model for walker: %d' % version)

  physics = Physics.from_string(model, common.ASSETS)
  # Note: reward functions specified here are not actually used.
  task = PlanarWalkerMultiReward(reward_function=stand_sparse_reward,
                                 evaluation_function=stand_sparse_reward,
                                 start_standing=True,
                                 random=random,
                                 version=version,
                                 frail=frail)
  return control.Environment(physics, task, time_limit=float('inf'),
                             control_timestep=_CONTROL_TIMESTEP)


class PlanarWalkerCliff(PlanarWalker):

  def __init__(self, *args, **kwargs):
    self._version = kwargs.pop('version')
    self._eval = kwargs.pop('eval')
    super(PlanarWalkerCliff, self).__init__(*args, **kwargs)

  def _get_rewards(self, physics):
    z = physics.data.qpos[0]
    x = physics.data.qpos[1]

    stand_reward = np.clip(0.25 * losses.identity(physics.torso_upright()) +
                           0.25 * losses.identity(physics.standing()) +
                           0.5 * (1 - losses.identity(physics.falling())),
                           0, 1)
    control_reward = np.clip(1 - 0.1 * losses.quadratic(physics.control()), 0, 1)
    walk_reward = np.clip(1 - 0.5 * losses.pseudo_huber(physics.walking(), p=0.1), 0, 1)
    run_reward = np.clip(1 - 0.2 * losses.pseudo_huber(physics.running(), p=0.1), 0, 1)
    reset_location_reward = 0.8 * (np.abs(x) < 0.5) + 0.2 * (1 - 0.2 * np.abs(x))
    forward_location_reward = 0.8 * (np.abs(5 - x) < 0.5) + 0.2 * (1 - 0.2 * np.abs(5 - x))

    if self._version == 0:
      forward_reward = 0.5 * run_reward + 0.25 * stand_reward + 0.25 * control_reward
    elif self._version == 1:
      forward_reward = 0.5 * forward_location_reward + 0.25 * stand_reward + 0.25 * control_reward
    else:
      raise ValueError('Unknow version for walker_cliff: %s' % self._version)
    reset_reward = 0.5 * reset_location_reward + 0.25 * stand_reward + 0.25 * control_reward

    return (forward_reward, reset_reward)

  def get_observation(self, physics):
    (forward_reward, reset_reward) = self._get_rewards(physics)
    rewards = np.array([forward_reward, reset_reward])

    if self._eval:
      x = 0.0
    else:
      x = physics.data.qpos[1]
    obs = np.concatenate(([x],
                          np.atleast_1d(physics.torso_height()),
                          physics.velocities(),
                          physics.orientations(),
                          rewards))

    return {'Doubles': obs}

  def get_termination(self, physics):
    """Override default method to prevent terminating before end of episode."""
    pass

