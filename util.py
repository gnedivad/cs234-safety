import collections
import csv
import numpy as np
import random

from ast import literal_eval as make_tuple
from configparser import ConfigParser
from enum import Enum
from queue import PriorityQueue


class MDPAlgorithm(object):

  def solve(self, mdp):
    """
    Computes something about the MDP.
    """
    raise NotImplementedError("Override me")


class PolicyEvaluation(MDPAlgorithm):

  def solve(self, mdp, pi, epsilon=1e-8, svf_from_all=False):
    """
    Computes the ground-truth values of the policy and state-visitation
    frequencies.

    Inputs:
    - epsilon: A Python float that represents the error tolerance for state
      visitation frequencies.
    - svf_from_all: A Python bool that determines whether to compute the
      state visitation frequencies and output as {rho}.

    Outputs:
    - R: A numpy array of shape (9, 9) that represents the estimates of the
      values of the policy.
    - rho: A numpy array of shape (9, 9) that represents the estimates of the
      state visitation frequencies.
    """
    states = mdp.states
    T = np.zeros((81, 81))
    for state in mdp.states:
      if state is None: continue  # ignore None state
      action = pi[state] if state is not None else None
      transitions = mdp.succAndProbReward(state, action)
      assert len(transitions) > 0
      for newState, prob, reward in transitions:
        if newState == None:
          assert prob == 1.0
          # We want probability in terminal states to go away on the next move
          T[mdp.stateToIdx(state)][mdp.stateToIdx(state)] = 0.0
        else:
          T[mdp.stateToIdx(state)][mdp.stateToIdx(newState)] += prob

    gamma = mdp.discount()
    R = np.zeros((9, 9))
    rho = np.zeros((9, 9))
    for state in mdp.states:
      if state is None: continue  # ignore None state
      # Computes the value of being in {state} for the MDP
      p = np.zeros(81)
      p[mdp.stateToIdx(state)] = 1.0
      svf = np.zeros(81)  # state visitation frequencies

      num_moves = 0
      total_reward = 0.0
      new_reward = None
      while sum(p) > epsilon:
        if svf_from_all or state == mdp.startState():
          svf += (gamma ** num_moves) * p
        new_reward = (gamma ** num_moves) * p[mdp.stateToIdx(mdp.goalState())]
        total_reward += new_reward
        p = p.dot(T)
        num_moves += 1
      R[state] = total_reward
      for idx, freq in enumerate(svf):
        rho[mdp.idxToState(idx)] += freq

    return R, rho


class ValueIteration(MDPAlgorithm):

  def solve(self, mdp, epsilon=0.001):
    """
    Computes the optimal policy of the MDP using value iteration.

    Inputs:
    - epsilon: A Python float that represents the error tolerance; value
      iteration continues until all of the values change by less than epsilon.

    Sets:
    - self.pi: optimal policy (mapping from state to action)
    - self.V: values (mapping from state to best values)
    """
    def computeQ(mdp, V, state, action):
      # Returns Q(state, action) based on V(state).
      return sum(prob * (reward + mdp.discount() * V[newState]) \
                 for newState, prob, reward in mdp.succAndProbReward(state, action))

    def computeOptimalPolicy(mdp, V):
      # Returns the optimal policy given the values V.
      pi = {}
      for state in mdp.states:
        pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
      return pi

    V = collections.defaultdict(float)  # state -> value of state
    numIters = 0
    while True:
      newV = {}
      for state in mdp.states:
        # This evaluates to zero for end states, which have no available actions (by definition)
        newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
      numIters += 1
      if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
        V = newV
        break
      V = newV

    # Computes the optimal policy now
    pi = computeOptimalPolicy(mdp, V)
    self.pi = pi
    self.V = V


class FollowerPolicy(MDPAlgorithm):
  """
  Implements the follower policy:
  - If its current location is along the planned path, then it tries to move to
    the next location along the planned path.
  - Otherwise, it takes the most direct path to the closest location along the
    planned path (doesn't consider whether this path is feasible). If multiple
    actions result in the same distance from the planned path, then it chooses
    the one that's closest to the latest location along the planned path.
  """
  def __init__(self):
    """
    Defines:
    - self.cache: A Python dict of dicts. The outer dict maps string
      representations of mdps to an inner dict. The inner dict maps states to
      actions of the policy.
    """
    self.cache = {}

  def solve(self, mdp, plannedPath):
    if str(mdp) in self.cache:
      return self.cache[str(mdp)]

    pi = {}
    for r in range(mdp.rows):
      for c in range(mdp.cols):
        state = (r, c)
        actions = mdp.actions(state)
        if len(actions) == 1:
          action = actions[0]
        else:
          try:
            # {state} is in {plannedPath}; this case assumes {plannedPath} was
            # constructed properly, such that it's feasible to get to the next
            # location in the plannedPath
            idx = plannedPath.index(state)
            newState = plannedPath[idx+1]
            action = tuple(np.array(newState) - np.array(state))
          except:
            # {state} is not in {plannedPath}
            # {newStates} is a numpy array
            plannedPathArr = np.array(plannedPath)
            newStatesArr = np.array(state) + np.array(actions)

            # A Python tuple consisting of:
            # - action
            # - distance from the resulting state to the closest location along
            #   the planned path
            # - index closest location along the planned path
            minAction = None
            for action, newStateArr in zip(actions, newStatesArr):
              norms = np.linalg.norm(newStateArr - plannedPathArr, axis=1)

              # Gets the index of the minimum norm (and the last occurence if
              # there are ties)
              minNormIdx = norms.shape[0] - 1 - np.argmin(norms[::-1])
              if (minAction is None
                  or norms[minNormIdx] < minAction[1]
                  or (norms[minNormIdx] < minAction[1] and minNormIdx > minAction[2])):
                minAction = (action, norms[minNormIdx], minNormIdx)
            action = minAction[0]

        pi[state] = action

    self.cache[str(mdp)] = pi
    return pi


class Dijkstra(object):
  """Implements Dijkstra's algorthm."""

  def __init__(self, costs_filename=None):
    """
    Inputs:
    - costs_filename: A Python string that represents the filename of the file
      containing the costs of each state; if None, then reads from config.ini.

    Defines:
    - self.cache: A Python dict mapping string representations of mdps to a list
      of tuples containing the shortest path in the mdp.
    - self.costs: A numpy array with shape of the grid that describes the costs
      of being in each state (higher costs correspond to worse states).
    """
    self.cache = {}

    cp = ConfigParser()
    cp.read("config.ini")

    # Prefers to use initialization argument, but falls back to config.ini
    if costs_filename is None:
      costs_filename = cp["GRID2D"].get("COSTS_FILENAME")

    costs = []
    with open(costs_filename, "r") as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        costs.append([int(elem.strip()) for elem in row])
    # Exponentiate to discourage unsafe paths (no matter how short)
    self.costs = 2 ** np.array(costs)

  def solve(self, mdp):
    if str(mdp) in self.cache:
      return self.cache[str(mdp)]

    done = np.zeros_like(self.costs, dtype=np.bool)
    totalCosts = np.inf * np.ones_like(self.costs)
    rows, cols = self.costs.shape
    shortestPaths = {}

    startState = mdp.startState()
    startCost = self.costs[startState]

    totalCosts[startState] = startCost
    pq = PriorityQueue()
    pq.put((startCost, [startState]))
    while not np.all(done):
      totalCost, locs = pq.get()
      loc = locs[-1]
      if done[loc]:
        continue

      done[loc] = True
      shortestPaths[loc] = locs
      r, c = loc
      for dr in range(-1, 2):
        for dc in range(-1, 2):
          if (0 <= r + dr < rows and 0 <= c + dc < cols
              and totalCost + self.costs[r+dr, c+dc] < totalCosts[r+dr, c+dc]):
            totalCosts[r+dr, c+dc] = totalCost + self.costs[r+dr, c+dc]
            pq.put((totalCosts[r+dr, c+dc], locs + [(r+dr, c+dc)]))

    # At this point, {totalCosts} is a numpy array with shape of the grid that
    # contains the cost of the shortest paths from the startState to each other
    # state in the grid; {shortestPaths} is a Python dict that maps each loc as
    # tuple to the shortest path of locs.
    shortestPath = shortestPaths[mdp.goalState()]
    self.cache[str(mdp)] = shortestPath
    return shortestPath


class MDP(object):
  """
  Defines an abstract class that represents a Markov Decision Process (MDP).
  """
  def startState(self):
    """Returns the start state."""
    raise NotImplementedError("Override me")

  def goalState(self):
    """Returns the goal state."""
    raise NotImplementedError("Override me")

  def actions(self, state):
    """Returns the set of actions possible from {state}."""
    raise NotImplementedError("Override me")

  def succAndProbReward(self, state, action):
    """
    Returns a list of (newState, prob, reward) tuples corresponding to edges
    coming out of {state}.

    Notation:
    - state: s
    - action: a
    - newState: s'
    - prob: T(s, a, s'),
    - reward: Reward(s, a, s')

    If IsEnd(state), return the empty list.
    """
    raise NotImplementedError("Override me")

  def discount(self):
    raise NotImplementedError("Override me")

  def computeStates(self):
    """
    Computes set of states reachable from startState. Helper function for
    MDPAlgorithms to know which states to compute values and policies for.
    
    Defines:
    - self.states: A Python set of all states reachable from the start state.
    """
    self.states = set()
    
    startState = self.startState()
    self.states.add(startState)

    queue = []
    queue.append(startState)
    while len(queue) > 0:
      state = queue.pop()
      for action in self.actions(state):
        for newState, prob, _ in self.succAndProbReward(state, action):
          if newState not in self.states:
            self.states.add(newState)
            queue.append(newState)

  def stateToIdx(self, state):
    raise NotImplementedError("Override me")

  def idxToState(self, state):
    raise NotImplementedError("Override me")


class GridMDPType(Enum):
  WALL = -1
  NONE = 0
  START = 1
  GOAL = 2


class Grid1DMDP(MDP):

  def __init__(self, n=1, safe=None):
    assert n > 0
    if safe is None:
      safe = [0]

    self.r = { i: 0.0 if i in safe else -1.0 for i in range(-n, n+1) }
    self.n = n

    # Defines {self.states}; must be defined after {self.r}
    self.computeStates()

  def startState(self): return 0

  def actions(self, state):
    # States with negative rewards are terminal
    return [0] if self.r[state] < 0.0 else [-1, +1]

  def succAndProbReward(self, state, action):
    newState = min(max(state + action, -self.n), +self.n)
    return [(state, 0.2, self.r[state]),
            (newState, 0.8, self.r[newState])]

  def discount(self): return 0.9


class Grid2DMDP(MDP):

  def __init__(self,
               grid_init=None,
               grid_filename=None,
               grid_shape=None,
               grid_goal_loc=None,
               R=None):
    cp = ConfigParser()
    cp.read("config.ini")

    # Prefers to use initialization argument, but falls back to config.ini
    if grid_init is None:
      grid_init = cp["GRID2D"].get("GRID_INIT")

    if grid_init == "file":
      # Prefers to use initialization argument, but falls back to config.ini
      if grid_filename is None:
        grid_filename = cp["GRID2D"].get("DEFAULT_FILENAME")

      # Reads the csv into a Python list of lists
      grid_str = []
      with open(grid_filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
          grid_str.append([elem.strip() for elem in row])

      # Converts the Python list of lists to numpy array
      self.grid_str = np.array(grid_str)

      # Converts numpy array to dtype GridMDPType
      self.grid = np.empty_like(self.grid_str, dtype=GridMDPType)
      self.grid[self.grid_str == "W"] = GridMDPType.WALL
      self.grid[self.grid_str == "N"] = GridMDPType.NONE
      self.grid[self.grid_str == "S"] = GridMDPType.START
      self.grid[self.grid_str == "G"] = GridMDPType.GOAL

      self.rows, self.cols = self.grid.shape
    elif grid_init == "config":
      # Prefers to use initialization argument, but falls back to config.ini
      if grid_shape is None:
        grid_shape = cp["GRID2D"].get("DEFAULT_SHAPE")
      if grid_goal_loc is None:
        grid_goal_loc = cp["GRID2D"].get("DEFAULT_GOAL_LOC")

      self.rows, self.cols = make_tuple(grid_shape)
      self.grid = np.array([
        [GridMDPType.NONE for _ in range(self.cols)] for _ in range(self.rows)])
      self.grid[make_tuple(grid_goal_loc)] = GridMDPType.GOAL

    if R is None:
      self.r = np.array([
        [0.0 for _ in range(self.cols)] for _ in range(self.rows)])
      for r in range(self.rows):
        for c in range(self.cols):
          if self.grid[r][c] == GridMDPType.GOAL:
            self.r[r][c] = 1.0
          elif self.grid[r][c] == GridMDPType.WALL:
            self.r[r][c] = 0.0
    else:
      self.r = np.array([
        [0.0 for _ in range(self.cols)] for _ in range(self.rows)])
      for r in range(self.rows):
        for c in range(self.cols):
          self.r[r][c] = R[self.stateToIdx((r, c))]

    # Defines {self.states}; must be defined after {self.r} gets initialized
    self.computeStates()

  def startState(self):
    startLocs = list(zip(*np.where(self.grid == GridMDPType.START)))
    if len(startLocs) == 1:
      return startLocs[0]
    return (8, 4)  # hardcoded for grids/9x9/*.csv

  def goalState(self):
    goalLocs = list(zip(*np.where(self.grid == GridMDPType.GOAL)))
    if len(goalLocs) == 1:
      return goalLocs[0]
    return (8, 4)  # hardcoded for grids/9x9/*.csv

  def randomStartState(self):
    feasibleStates = list(zip(
      *[list(l) for l in np.where(self.grid != GridMDPType.WALL)]))
    return random.choice(feasibleStates)

  def getFeasibleStates(self):
    if not hasattr(self, "feasibleStates"):
      self.feasibleStates = list(zip(
        *[list(l) for l in np.where(self.grid != GridMDPType.WALL)]))
    return self.feasibleStates

  def actions(self, state):
    if state is None:
      return [None]  # placeholder since {succAndProbReward} ignores {action}

    if (self.grid[state] == GridMDPType.GOAL
        or self.grid[state] == GridMDPType.WALL):
      return [None]  # placeholder since {succAndProbReward} ignores {action}

    r, c = state
    a = []
    for dr in range(-1, 2):
      for dc in range(-1, 2):
        if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols:
          a.append((dr, dc))
    return a

  def succAndProbReward(self, state, action):
    if state is None:
      # Terminal states do not have successors
      return []

    if (self.grid[state] == GridMDPType.GOAL
        or self.grid[state] == GridMDPType.WALL):
      # Collects the reward associated with being in this state before visiting
      # the terminal state, {None}
      return [(None, 1.0, self.r[state])]

    # With prob {P_ACTION} follows the transition implied by the action taken by
    # the agent; with prob {1 - P_ACTION} follows a random transition
    P_ACTION = 0.8
    actions = self.actions(state)
    return list(map(lambda a: (
      (state[0] + a[0], state[1] + a[1]),
      P_ACTION if a == action else (1.0 - P_ACTION) / (len(actions) - 1),
      self.r[state]
    ), actions))

  def discount(self): return 0.9

  def stateToIdx(self, state):
    if state is None:
      return self.rows * self.cols

    return state[0] * self.cols + state[1]

  def idxToState(self, idx):
    if idx == self.rows * self.cols:
      return None

    return (idx // self.cols, idx % self.cols)

  def __str__(self):
    return "\n".join([
      "".join([str(col) for col in row]) for row in self.grid_str])


class IRL(object):

  def compute_state_visitation_frequency(self, D, vi, mdp):
    """
    Inputs:
    - D: A Python list of lists of tuples, where each inner list represents a
      trajectory and each tuple represents a state.
    - vi: An instance of ValueIteration that contains information about the
      current estimates of the values of the states.
    - mdp: An instance of MDP that contains information about the states to
      indices mappings.

    Outputs:
    - 
    """
    # T is the number of timesteps
    T = len(D[0])

    # mu[s,t] is the probability of visiting state s at time t
    mu = np.zeros((mdp.rows * mdp.cols, T))

    # Sets initial probabilities
    for d in D:
      mu[mdp.stateToIdx(d[0]), 0] += 1
    mu[:,0] /= len(D)

    for t in range(T - 1):
      for s in range(mdp.rows * mdp.cols):
        state = mdp.idxToState(s)
        transitions = mdp.succAndProbReward(state, vi.pi[state])
        for newState, prob, _ in transitions:
          s_ = mdp.stateToIdx(newState)
          mu[s_, t+1] += mu[s, t] * prob

    return np.sum(mu, axis=1)


  def solve(self, D, mdp, f, iters=100, lr=0.01):
    """
    Tries to recover mdp.r from demonstrations D, given feature map f.

    Inputs:
    - f: A
    """
    f_D = np.zeros((mdp.rows * mdp.cols,))
    for d in D:
      for s in d:
        f_D += f[mdp.stateToIdx(s)]
    f_D /= len(D)

    # Train
    _lambda = np.random.uniform(size=(mdp.rows * mdp.cols,))
    for i in range(iters):
      # A Python numpy array of shape (|S|,), where |S| is the number of states
      R_hat = np.dot(f, _lambda)

      # Creates an MDP with rewards R, and solves it with value iteration
      curMdp = Grid2DMDP(R=R_hat)
      vi = ValueIteration()
      vi.solve(curMdp)

      svf = self.compute_state_visitation_frequency(D, vi, curMdp)

      grad = f_D - f.T.dot(svf)

      _lambda += lr * grad

    R_hat = np.dot(f, _lambda)
    return { mdp.idxToState(idx): reward for idx, reward in enumerate(R_hat) }


def print_values(values_dict, rows, cols):
  values_arr = np.zeros((rows, cols))
  for r in range(rows):
    for c in range(cols):
      if (r, c) in values_dict:
        values_arr[r][c] = values_dict[(r, c)]
  print(np.round(values_arr, 2))
