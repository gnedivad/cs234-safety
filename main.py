import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random

from collections import defaultdict
from configparser import ConfigParser
from scipy.stats import multivariate_normal
from tqdm import tqdm
from util import (
  Dijkstra, Grid2DMDP, FollowerPolicy, PolicyEvaluation, ValueIteration, print_values)


def generateMonteCarloDataset(n=1000):
  """
  Model-free Monte Carlo approach
  (1) Construct a dataset upfront, then (2) train:

  Construct a dataset mapping states to expected rewards (Python float) upfront
  - Sample goal state reachable from start state
  - Plan deterministic trajectory from start state to goal state based on
    Dijkstra's
  - Roll-out according to policy (conditioned on start state and planned
    trajectory) to find a reward R of the roll-out (since only terminal states
    have non-zero rewards in our MDP, it's exactly equal to the discounted
    reward r of the terminal state in the roll-out)
  - For each state s in the roll-out path p:
      Let X = (s, trajectory) and y = gamma^{len(p)} * r
      Add (X, y) to the dataset

  Train on (X, y) to predict y from X
  - Learn (or hand-craft) feature embedding for X
  """
  def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
      accum += prob
      if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)

  cp = ConfigParser()
  cp.read("config.ini")

  grids_root_dir = cp["EVAL"].get("GRIDS_ROOT_DIR")

  D = []
  filenames = [
    os.path.join(f"{grids_root_dir}", filename)
      for filename in os.listdir(f"{grids_root_dir}")]
  dijkstra = Dijkstra()
  follower = FollowerPolicy()

  for _ in tqdm(range(n)):
    # Sample mdp
    grid_filename = random.choice(filenames)
    mdp = Grid2DMDP(grid_init="file", grid_filename=grid_filename)

    # Our planned path is the safest path, as determined by Dijkstra's
    # (uses caching to prevent unnecessary searches for seen MDP's)
    plannedPath = dijkstra.solve(mdp)
    pi = follower.solve(mdp, plannedPath)

    # Performs roll-out of policy {pi}
    path = []  # actual path followed
    state = mdp.startState()
    totalReward = 0
    while True:
      if state is not None: path.append(state)
      action = pi[state] if state is not None else None
      transitions = mdp.succAndProbReward(state, action)
      if len(transitions) == 0:
        break
      i = sample([prob for _, prob, _ in transitions])
      newState, prob, reward = transitions[i]
      totalReward += reward
      state = newState

    for idx, state in enumerate(path):
      X = (state, plannedPath)
      y = totalReward * (mdp.discount() ** (len(path) - idx - 1))
      D.append((X, y))

  return D


def generateTdDataset(n=1000):
  """
  TD approach
  (1) Construct a dataset upfront, then (2) train:

  Construct an experience replay dataset of (s, a, r, s') tuples upfront, where
  r represents the immediate reward of the transition from s to s'; since we're
  performing policy evaluation, the action can be ignored
  - Sample goal state reachable from start state
  - Plan deterministic trajectory from start state to goal state based on
    Dijkstra's
  - Roll-out according to policy (conditioned on start state and planned
    trajectory)
  - For each state s in the roll-out path p:
      Let X = (s, r, s')
      Add X to the dataset

  Train an estimate of value function V_pi on (X,) to predict
  V_pi[s] = r + gamma * E[s'] from s
  - Learn (or hand-craft) feature embedding for X
  """
  def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
      accum += prob
      if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)

  cp = ConfigParser()
  cp.read("config.ini")

  grids_root_dir = cp["EVAL"].get("GRIDS_ROOT_DIR")

  D = []
  filenames = [
    os.path.join(f"{grids_root_dir}", filename)
      for filename in os.listdir(f"{grids_root_dir}")]
  dijkstra = Dijkstra()
  follower = FollowerPolicy()

  for _ in tqdm(range(n)):
    # Sample mdp
    grid_filename = random.choice(filenames)
    mdp = Grid2DMDP(grid_init="file", grid_filename=grid_filename)

    # Our planned path is the safest path, as determined by Dijkstra's
    # (uses caching to prevent unnecessary searches for seen MDP's)
    plannedPath = dijkstra.solve(mdp)
    pi = follower.solve(mdp, plannedPath)

    # Performs roll-out of policy {pi}
    path = []  # actual path followed
    state = mdp.startState()

    while True:
      if state is not None: path.append(state)
      action = pi[state] if state is not None else None
      transitions = mdp.succAndProbReward(state, action)
      if len(transitions) == 0:
        break
      i = sample([prob for _, prob, _ in transitions])
      newState, prob, reward = transitions[i]
      X = (state, plannedPath, reward, newState)
      D.append(X)
      state = newState

  return D


def generate_expected_rewards():
  cp = ConfigParser()
  cp.read("config.ini")

  grids_root_dir = cp["EVAL"].get("GRIDS_ROOT_DIR")
  truth_root_dir = cp["EVAL"].get("TRUTH_ROOT_DIR")

  filenames = [
    os.path.join(f"{grids_root_dir}", filename)
      for filename in os.listdir(f"{grids_root_dir}")]

  if not os.path.exists(f"{truth_root_dir}/svf"):
    os.makedirs(f"{truth_root_dir}/svf")

  dijkstra = Dijkstra()
  follower = FollowerPolicy()
  pe = PolicyEvaluation()

  # A Python dict mapping a Python tuple representing the goal state to a
  # numpy array representing the expected rewards from each of the states
  rewards = {}
  rhos = {}
  for filename in filenames:
    mdp = Grid2DMDP(grid_init="file", grid_filename=filename)
    plannedPath = dijkstra.solve(mdp)
    pi = follower.solve(mdp, plannedPath)
    R, rho = pe.solve(mdp, pi)
    rewards[mdp.goalState()] = R
    rhos[mdp.goalState()] = rho

    plt.figure()
    plt.imshow(rho, cmap="gray", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(mdp.goalState())
    plt.text(3.86, 8.1, "S", color="white", fontsize=14)
    plt.plot([mdp.goalState()[1]], [mdp.goalState()[0]], marker="*", markersize=14)
    plt.plot([3.5, 3.5], [7.5, 8.5], 'r-', lw=4)
    plt.plot([2.5, 3.5], [7.5, 7.5], 'r-', lw=4)
    plt.plot([2.5, 2.5], [5.5, 7.5], 'r-', lw=4)
    plt.plot([1.5, 2.5], [5.5, 5.5], 'r-', lw=4)
    plt.plot([1.5, 1.5], [4.5, 5.5], 'r-', lw=4)
    plt.plot([0.5, 1.5], [4.5, 4.5], 'r-', lw=4)
    plt.plot([0.5, 0.5], [1.5, 4.5], 'r-', lw=4)
    plt.plot([0.5, 1.5], [1.5, 1.5], 'r-', lw=4)
    plt.plot([1.5, 1.5], [0.5, 1.5], 'r-', lw=4)
    plt.plot([1.5, 2.5], [0.5, 0.5], 'r-', lw=4)
    plt.plot([2.5, 2.5], [0.5, 1.5], 'r-', lw=4)
    plt.plot([2.5, 3.5], [1.5, 1.5], 'r-', lw=4)
    plt.plot([3.5, 3.5], [1.5, 2.5], 'r-', lw=4)
    plt.plot([1.5, 3.5], [2.5, 2.5], 'r-', lw=4)
    plt.plot([1.5, 1.5], [2.5, 3.5], 'r-', lw=4)
    plt.plot([1.5, 2.5], [3.5, 3.5], 'r-', lw=4)
    plt.plot([2.5, 2.5], [3.5, 4.5], 'r-', lw=4)
    plt.plot([2.5, 3.5], [4.5, 4.5], 'r-', lw=4)
    plt.plot([3.5, 3.5], [4.5, 5.5], 'r-', lw=4)
    plt.plot([3.5, 4.5], [5.5, 5.5], 'r-', lw=4)
    plt.plot([4.5, 4.5], [0.5, 5.5], 'r-', lw=4)
    plt.plot([4.5, 7.5], [0.5, 0.5], 'r-', lw=4)
    plt.plot([7.5, 7.5], [0.5, 5.5], 'r-', lw=4)
    plt.plot([6.5, 7.5], [5.5, 5.5], 'r-', lw=4)
    plt.plot([6.5, 6.5], [6.5, 5.5], 'r-', lw=4)
    plt.plot([5.5, 6.5], [6.5, 6.5], 'r-', lw=4)
    plt.plot([5.5, 5.5], [7.5, 6.5], 'r-', lw=4)
    plt.plot([4.5, 5.5], [7.5, 7.5], 'r-', lw=4)
    plt.plot([4.5, 4.5], [7.5, 8.5], 'r-', lw=4)
    plt.plot([3.5, 4.5], [8.5, 8.5], 'r-', lw=4)
    plt.axis("off")
    plt.savefig(f"{truth_root_dir}/svf/{str(mdp.goalState()[0])},{str(mdp.goalState()[1])}_svf.png")

  pickle.dump(rewards, open(f"{truth_root_dir}/rewards.pkl", "wb"))
  pickle.dump(rhos, open(f"{truth_root_dir}/rhos.pkl", "wb"))


def generateTdherDatasetFromTdDataset(tdFuncApproxDatasetFilename):
  """
  Since there were several TD datasets that were experimented on before the
  definition of the HER dataset, we need to convert TD datasets into HER
  datasets.
  """
  with open(tdFuncApproxDatasetFilename, "rb") as tdFuncApproxDatasetFile:
    tdFuncApproxDataset = pickle.load(tdFuncApproxDatasetFile)

  herFuncApproxDataset = []
  for X in tdFuncApproxDataset:
    state, plannedPath, reward, newState = X

    # Append the original example to the HER dataset
    herFuncApproxDataset.append(X)

    try:
      curStateIdx = plannedPath.index(state)
      herFuncApproxDataset.append((
        state,
        plannedPath[:curStateIdx+1],
        1.0,
        None
      ))

      for idx in range(curStateIdx+1, len(plannedPath)):
        herFuncApproxDataset.append((
          state,
          plannedPath[:idx+1],
          0.0,
          newState
        ))
    except:
      # Current state not along path
      for idx in range(0, len(plannedPath)):
        herFuncApproxDataset.append((
          state,
          plannedPath[:idx+1],
          0.0,
          newState
        ))

  return herFuncApproxDataset


def generateUniqueHerDatasetFromTdDataset(tdFuncApproxDatasetFilename):
  """
  Since there were several TD datasets that were experimented on before the
  definition of the HER dataset, we need to convert TD datasets into HER
  datasets.
  """
  with open(tdFuncApproxDatasetFilename, "rb") as tdFuncApproxDatasetFile:
    tdFuncApproxDataset = pickle.load(tdFuncApproxDatasetFile)

  uniqueHerFuncApproxDataset = set()
  for X in tdFuncApproxDataset:
    state, plannedPath, reward, newState = X

    # Append the original example to the HER dataset
    uniqueHerFuncApproxDataset.add((
      state, tuple(plannedPath), reward, newState))

    try:
      curStateIdx = plannedPath.index(state)
      uniqueHerFuncApproxDataset.add((
        state,
        tuple(plannedPath[:curStateIdx+1]),
        1.0,
        None
      ))

      for idx in range(curStateIdx+1, len(plannedPath)):
        uniqueHerFuncApproxDataset.add((
          state,
          tuple(plannedPath[:idx+1]),
          0.0,
          newState
        ))
    except:
      # Current state not along path
      for idx in range(0, len(plannedPath)):
        uniqueHerFuncApproxDataset.add((
          state,
          tuple(plannedPath[:idx+1]),
          0.0,
          newState
        ))

  uniqueHerFuncApproxDataset = list(uniqueHerFuncApproxDataset)
  return ([(state, list(path), reward, newState) for state, path, reward, newState in uniqueHerFuncApproxDataset])


def generateDatasets(n=1000):
  def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
      accum += prob
      if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)

  cp = ConfigParser()
  cp.read("config.ini")

  grids_root_dir = cp["EVAL"].get("GRIDS_ROOT_DIR")

  mcRolloutDataset = defaultdict(list)
  mcFuncApproxDataset = []
  tdFuncApproxDataset = []
  filenames = [
    os.path.join(f"{grids_root_dir}", filename)
      for filename in os.listdir(f"{grids_root_dir}")]
  dijkstra = Dijkstra()
  follower = FollowerPolicy()

  for _ in tqdm(range(n)):
    # Sample mdp
    grid_filename = random.choice(filenames)
    mdp = Grid2DMDP(grid_init="file", grid_filename=grid_filename)

    # Our planned path is the safest path, as determined by Dijkstra's
    # (uses caching to prevent unnecessary searches for seen MDP's)
    plannedPath = dijkstra.solve(mdp)
    pi = follower.solve(mdp, plannedPath)

    # Performs roll-out of policy {pi}
    path = []  # actual path followed
    state = mdp.startState()
    totalReward = 0
    while True:
      if state is not None: path.append(state)
      action = pi[state] if state is not None else None
      transitions = mdp.succAndProbReward(state, action)
      if len(transitions) == 0:
        break
      i = sample([prob for _, prob, _ in transitions])
      newState, prob, reward = transitions[i]

      # Housekeeping for MC-FUNCAPPROX dataset
      totalReward += reward

      # Housekeeping for TD-FUNCAPPROX dataset
      X = (state, plannedPath, reward, newState)
      tdFuncApproxDataset.append(X)

      state = newState

    # Housekeeping for MC-ROLLOUT dataset
    mcRolloutDataset[tuple(plannedPath)].append(
      totalReward * (mdp.discount() ** (len(path) - 1)))

    # Housekeeping for MC-FUNCAPPROX dataset
    for idx, state in enumerate(path):
      X = (state, plannedPath)
      y = totalReward * (mdp.discount() ** (len(path) - idx - 1))
      mcFuncApproxDataset.append((X, y))

  return mcRolloutDataset, mcFuncApproxDataset, tdFuncApproxDataset


def solve_for_e_r(sigma=1.0):
  def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

  cp = ConfigParser()
  cp.read("config.ini")

  grids_root_dir = cp["EVAL"].get("GRIDS_ROOT_DIR")

  filenames = [
    os.path.join(f"{grids_root_dir}", filename)
      for filename in os.listdir(f"{grids_root_dir}")]
  dijkstra = Dijkstra()
  follower = FollowerPolicy()

  cached_probs = np.zeros((51, 51))
  for row in range(51):
    for col in range(51):
      cached_probs[row, col] = multivariate_normal.pdf(
        (row, col), mean=(25, 25), cov=[[sigma, 0], [0, sigma]])
  cached_probs = cached_probs / np.sum(cached_probs)


  e_r_list = []
  for grid_filename in filenames:
    mdp = Grid2DMDP(grid_init="file", grid_filename=grid_filename)

    # Our planned path is the safest path, as determined by Dijkstra's
    # (uses caching to prevent unnecessary searches for seen MDP's)
    plannedPath = dijkstra.solve(mdp)
    pi = follower.solve(mdp, plannedPath)

    feasibleStates = mdp.getFeasibleStates()  # mdp.states

    
    for o_row in range(0, 9):
      for o_col in range(0, 9):
        o = (o_row, o_col)

        p = np.zeros(81)

        # With constraint that s must be internal to the lung:
        for feasibleState in feasibleStates:
          p[mdp.stateToIdx(feasibleState)] = multivariate_normal.pdf(
            feasibleState, mean=o, cov=[[sigma, 0], [0, sigma]])

        # Without that constraint
        # for idx in range(81):
        #   state = mdp.idxToState(idx)
        #   dr = state[0] - o_row
        #   dc = state[1] - o_col
        #   p[idx] = cached_probs[25 + dr, 25 + dc]

        p_norm = p / np.sum(p)

        if o == mdp.goalState():
          # -r(o, pi(o)) = -1
          # r(s, pi(o)) = 0 for s != o
          # For all s != o: [-r(o, pi(o)) + r(s, pi(o))] = -1
          # The sum we desire is just -\sum_{s \neq o} p(s|o) i.e. -(1 - p(s=o | o))
          e_r = -(1 - p_norm[mdp.stateToIdx(o)])
          e_r_list.append(e_r)
        else:
          # -r(o, pi(o)) = 0
          # r(s, pi(o)) = 1 for s == mdp.goalState
          # For all s != o or goalState: r(s, pi(o)) = 0
          e_r = p_norm[mdp.stateToIdx(mdp.goalState())]
          e_r_list.append(e_r)


    # print(grid_filename)

  print("sigma=%.2f, Constrained" % sigma)
  print(np.mean(e_r_list))
  print(np.std(e_r_list))

  ax = plt.subplot(111)

  ax.hist(e_r_list, bins=20, log=True)
  plt.title("sigma=%.2f, Constrained" % sigma, fontsize=14)
  plt.xlabel("epsilon_r", fontsize=14)
  plt.ylabel("Count (log)", fontsize=14)

  ax.set_xlim([-1, 1])
  ax.set_ylim([1, 5e3])

  ax.spines["right"].set_visible(False)
  ax.spines["top"].set_visible(False)

  ax.xaxis.set_tick_params(labelsize=14)
  ax.yaxis.set_tick_params(labelsize=14)
  plt.show()

  import pdb; pdb.set_trace()



if __name__ == "__main__":
  # python main.py
  # D = generateMonteCarloDataset(n=4000)
  # os.makedirs("datasets/9x9/MC-0020-4k")
  # pickle.dump(D, open("datasets/9x9/MC-0020-4k/dataset.p", "wb"))

  # D = generateTdDataset(n=256)
  # os.makedirs("datasets/9x9/TD-0020-256")
  # pickle.dump(D, open("datasets/TD-0020-256/dataset.p", "wb"))

  # ns = [16, 64, 256]

  # for dataset_id in range(1, 10):
  #   for n in ns:
  #     mcRolloutDataset, mcFuncApproxDataset, tdFuncApproxDataset = generateDatasets(n=n)

  #     if not os.path.exists(f"datasets/9x9-0{str(dataset_id)}/MC-ROLLOUT-0020-{str(n)}"):
  #       os.makedirs(f"datasets/9x9-0{str(dataset_id)}/MC-ROLLOUT-0020-{str(n)}")
  #     if not os.path.exists(f"datasets/9x9-0{str(dataset_id)}/MC-FUNCAPPROX-0020-{str(n)}"):
  #       os.makedirs(f"datasets/9x9-0{str(dataset_id)}/MC-FUNCAPPROX-0020-{str(n)}")
  #     if not os.path.exists(f"datasets/9x9-0{str(dataset_id)}/TD-FUNCAPPROX-0020-{str(n)}"):
  #       os.makedirs(f"datasets/9x9-0{str(dataset_id)}/TD-FUNCAPPROX-0020-{str(n)}")

  #     pickle.dump(mcRolloutDataset, open(f"datasets/9x9-0{str(dataset_id)}/MC-ROLLOUT-0020-{str(n)}/dataset.p", "wb"))
  #     pickle.dump(mcFuncApproxDataset, open(f"datasets/9x9-0{str(dataset_id)}/MC-FUNCAPPROX-0020-{str(n)}/dataset.p", "wb"))
  #     pickle.dump(tdFuncApproxDataset, open(f"datasets/9x9-0{str(dataset_id)}/TD-FUNCAPPROX-0020-{str(n)}/dataset.p", "wb"))


  #     Things that a HER dataset could add
  #     - Pretend that plannedPath is shorter (either started later than it actually
  #       did or ended earlier than it actually did)

  #     Converts TD dataset to TD HER dataset
  #     tdherFuncApproxDataset = generateTdherDatasetFromTdDataset(f"datasets/9x9-0{str(dataset_id)}/TD-FUNCAPPROX-0020-{str(n)}/dataset.p")
  #     if not os.path.exists(f"datasets/9x9-0{str(dataset_id)}/TDHER-FUNCAPPROX-0020-{str(n)}"):
  #       os.makedirs(f"datasets/9x9-0{str(dataset_id)}/TDHER-FUNCAPPROX-0020-{str(n)}")
  #     pickle.dump(tdherFuncApproxDataset, open(f"datasets/9x9-0{str(dataset_id)}/TDHER-FUNCAPPROX-0020-{str(n)}/dataset.p", "wb"))

  # Converts TD dataset to unique HER dataset
  # n = 16384
  # uniqueHerFuncApproxDataset = generateUniqueHerDatasetFromTdDataset(f"datasets/9x9-04/TD-FUNCAPPROX-0020-{str(n)}/dataset.p")
  # if not os.path.exists(f"datasets/9x9-04/UNIQHER-FUNCAPPROX-0020-{str(n)}"):
  #   os.makedirs(f"datasets/9x9-04/UNIQHER-FUNCAPPROX-0020-{str(n)}")
  # pickle.dump(uniqueHerFuncApproxDataset, open(f"datasets/9x9-04/UNIQHER-FUNCAPPROX-0020-{str(n)}/dataset.p", "wb"))

  # generate_expected_rewards()

  solve_for_e_r()
