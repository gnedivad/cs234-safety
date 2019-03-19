import numpy as np
import random
from util import Grid1DMDP, Grid2DMDP, ValueIteration, IRL, print_values


def generate_demonstrations(mdp, vi, n=1000, traj_len_max=50):

  def sample(probs):
    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
      accum += prob
      if accum >= target: return i
    raise Exception("Invalid probs: %s" % probs)

  D = []
  for _ in range(n):
    state = mdp.randomStartState()
    traj_len = 0
    d = []
    while True:
      traj_len += 1
      d.append(state)
      if traj_len > traj_len_max:
        break

      action = vi.pi[state]
      transitions = mdp.succAndProbReward(state, action)
      i = sample([prob for newState, prob, reward in transitions])
      newState, prob, reward = transitions[i]
      state = newState
    D.append(d)

  return D


if __name__ == "__main__":
  # python irl.py
  mdp = Grid2DMDP()
  vi = ValueIteration()
  vi.solve(mdp)
  print_values(vi.V, mdp.rows, mdp.cols)
  D = generate_demonstrations(mdp, vi)
  irl = IRL()
  R_hat = irl.solve(D, mdp, np.eye(mdp.rows * mdp.cols))
  print_values(R_hat, mdp.rows, mdp.cols)

  counts = np.zeros((mdp.rows, mdp.cols))
  for d in D:
    for s in d:
      counts[s] += 1
  print(counts.astype(np.uint32))
