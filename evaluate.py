import argparse
import numpy as np
import os
import pickle
import random
import re

from configparser import ConfigParser
from train import (
  build_policy_evaluation_model_with_attention,
  load_mc_dataset,
  load_td_dataset,
  preprocess_mc_datum,)
from util import (
  Dijkstra, Grid2DMDP, FollowerPolicy, ValueIteration, print_values)


def evaluate_monte_carlo_rollout(dataset_folder,
                                 model_name,
                                 truth_root_dir,
                                 log_filename,
                                 epsilon=0.01):
  # def sample(probs):
  #   target = random.random()
  #   accum = 0
  #   for i, prob in enumerate(probs):
  #     accum += prob
  #     if accum >= target: return i
  #   raise Exception("Invalid probs: %s" % probs)

  with open(f"{truth_root_dir}/rewards.pkl", "rb") as rewards_file:
    rewards = pickle.load(rewards_file)
  with open(f"{truth_root_dir}/rhos.pkl", "rb") as rho_file:
    rhos = pickle.load(rho_file)

  # filenames = [
  #   os.path.join(f"{grids_root_dir}", filename)
  #     for filename in os.listdir(f"{grids_root_dir}")]
  # dijkstra = Dijkstra()
  # follower = FollowerPolicy()

  # # Trackers
  # counters = []
  # errors = []

  # for grid_filename in filenames:
  #   mdp = Grid2DMDP(grid_init="file", grid_filename=grid_filename)

  #   # Our planned path is the safest path, as determined by Dijkstra's
  #   # (uses caching to prevent unnecessary searches for seen MDP's)
  #   plannedPath = dijkstra.solve(mdp)
  #   startState = mdp.startState()
  #   goalState = plannedPath[-1]
  #   Y_true = rewards[goalState]
  #   pi = follower.solve(mdp, plannedPath)

  #   averageReward = 0.0  # average reward from start state
  #   counter = 0
  #   while True:
  #     # Performs roll-out of policy {pi}
  #     path = []  # actual path followed
  #     state = startState
  #     totalReward = 0
  #     while True:
  #       if state is not None: path.append(state)
  #       action = pi[state] if state is not None else None
  #       transitions = mdp.succAndProbReward(state, action)
  #       if len(transitions) == 0:
  #         break
  #       i = sample([prob for _, prob, _ in transitions])
  #       newState, prob, reward = transitions[i]
  #       totalReward += reward
  #       state = newState
  #     counter += 1
  #     totalReward *= mdp.discount() ** (len(path) - 1)
  #     newAverageReward = (averageReward * (counter - 1) + totalReward) / counter
  #     # if (abs(averageReward - newAverageReward) < epsilon
  #     #     and counter > 32):
  #     if counter > 625:
  #       break
  #     averageReward = newAverageReward

  #   counters.append(counter)
  #   errors.append(np.abs(Y_true[startState] - averageReward))
  #   print(counter,
  #         goalState,
  #         Y_true[startState],
  #         averageReward,
  #         np.abs(Y_true[startState] - averageReward))
  # print(sum(counters))
  # print(np.mean(errors))

  #############################################################################

  try:
    result = re.findall("(\w+)-ROLLOUT-0020-(\d+)", model_name)[0]
    algorithm, dataset_size = result
  except:
    print(f"Cannot extract information from model_name {model_name}.")
    import pdb; pdb.set_trace()

  for subdir in os.listdir(dataset_folder):
    try:
      mc_rollout_root_dir = f"{dataset_folder}/{subdir}/MC-ROLLOUT-0020-{dataset_size}"

      with open(f"{mc_rollout_root_dir}/dataset.p", "rb") as mc_rollout_file:
        mc_rollout = pickle.load(mc_rollout_file)

      errors = []
      startState = (8, 4)  # hardcoded for now
      goalStates = set([
        (7, 3), (4, 7), (6, 6), (5, 6), (2, 1), (1, 6), (3, 7), (2, 5), (1, 2),
        (5, 5), (6, 3), (1, 5), (3, 6), (2, 2), (4, 1), (6, 4), (2, 6), (4, 5),
        (7, 5), (2, 3), (4, 2), (6, 5), (3, 5), (2, 7), (5, 3), (4, 6), (5, 7),
        (3, 1), (7, 4), (1, 7), (5, 2), (8, 4)])  # 32 valid locations
      goalStatesToPred = {}  # mc_rollout maps the entire planned trajectory
                             # from start state (8, 4) to outcomes [1, 0, 0, 1 ...]
                             # whereas goalStatesToPred will map the goal state
                             # (the last location in the planned trajectory) to
                             # outcomes

      for k, v in mc_rollout.items():
        if k != 0:
          goalState = k[-1]
          Y_true = rewards[goalState][startState]
          Y_pred = np.mean(v)
          errors.append(np.abs(Y_true - Y_pred))
          goalStatesToPred[goalState] = v

      # for goalState in goalStates:
      #   if goalState not in mc_rollout:
      #     r, c = goalState
      #     neighbor_errors = []
      #     for dr in range(-1, 2):
      #       for dc in range(-1, 2):
      #         neighbor = (r + dr, c + dc)
      #         if neighbor in goalStatesToPred:
      #           neighbor_errors.extend(goalStatesToPred[neighbor])

      #     Y_true = rewards[goalState][startState]
      #     if len(neighbor_errors) == 0:
      #       Y_pred = 0.5  # not sure
      #     else:
      #       Y_pred = np.mean(neighbor_errors)

      #     errors.append(np.abs(Y_true - Y_pred))
    except:
      continue

  with open(log_filename, "a") as run_jobs_file:
    run_jobs_file.write(f"{model_name}, {np.mean(errors)}\n")


def evaluate_function_approximation(dataset_folder, model_name, truth_root_dir, log_filename):
  model = build_policy_evaluation_model_with_attention()
  model.load_weights(f"models/{model_name}.h5")

  print(f"Evaluating models/{model_name}.h5")

  with open(f"{truth_root_dir}/rewards.pkl", "rb") as rewards_file:
    rewards = pickle.load(rewards_file)
  with open(f"{truth_root_dir}/rhos.pkl", "rb") as rho_file:
    rhos = pickle.load(rho_file)

  testPlannedPaths = [
    [(8, 4), (7, 4), (6, 4), (5, 3), (4, 2), (3, 1), (2, 2), (1, 2)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6), (2, 6), (1, 5)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6), (2, 6), (1, 6)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6), (2, 6), (1, 7)],
    [(8, 4), (7, 4), (6, 4), (5, 3), (4, 2), (3, 1), (2, 1)],
    [(8, 4), (7, 4), (6, 4), (5, 3), (4, 2), (3, 1), (2, 2)],
    [(8, 4), (7, 4), (6, 4), (5, 3), (4, 2), (3, 1), (2, 2), (2, 3)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6), (2, 5)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6), (2, 6)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6), (2, 7)],
    [(8, 4), (7, 4), (6, 4), (5, 3), (4, 2), (3, 1)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 5)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 6)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6), (3, 7)],
    [(8, 4), (7, 4), (6, 3), (5, 2), (4, 1)],
    [(8, 4), (7, 4), (6, 4), (5, 3), (4, 2)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 5)],
    [(8, 4), (7, 4), (6, 4), (5, 5), (4, 6)],
    [(8, 4), (7, 4), (6, 5), (5, 6), (4, 7)],
    [(8, 4), (7, 4), (6, 3), (5, 2)],
    [(8, 4), (7, 4), (6, 4), (5, 3)],
    [(8, 4), (7, 4), (6, 4), (5, 5)],
    [(8, 4), (7, 4), (6, 5), (5, 6)],
    [(8, 4), (7, 4), (6, 5), (5, 6), (5, 7)],
    [(8, 4), (7, 4), (6, 3)],
    [(8, 4), (7, 4), (6, 4)],
    [(8, 4), (7, 4), (6, 5)],
    [(8, 4), (7, 4), (6, 5), (6, 6)],
    [(8, 4), (7, 3)],
    [(8, 4), (7, 4)],
    [(8, 4), (7, 5)],
    [(8, 4)]
  ]
  errors = []

  # Tests the test paths
  for testPlannedPath in testPlannedPaths:
    startState = testPlannedPath[0] 
    goalState = testPlannedPath[-1]
    Y_true = rewards[goalState]
    rho = rhos[goalState]
    Y_pred = np.zeros((9, 9))
    for r in range(9):
      for c in range(9):
        stateArr, plannedPathArr, _ = preprocess_mc_datum(((r, c), testPlannedPath), None)
        y_pred = model.predict([np.expand_dims(stateArr, axis=0),
                                np.expand_dims(plannedPathArr, axis=0)])
        Y_pred[r, c] = y_pred

    # Compares Y_pred to Y_true, weighted by rho
    # print(goalState, np.sum(np.square(Y_true - Y_pred) * rho / np.sum(rho)))

    # Compares Y_pred to Y_true for just startState
    errors.append(np.abs(Y_true - Y_pred)[startState])
    # print(goalState,
    #       Y_true[startState],
    #       Y_pred[startState],
    #       np.abs(Y_true - Y_pred)[startState])

  # Tests loss on the training set
  training_errors = []  # not exactly "training" since other trials might have
                        # unseen examples; these are mean squared errors
  dev_errors = []

  try:
    result = re.findall("(\w+)-FUNCAPPROX-0020-(\d+)-(\d+)", model_name)[0]
    algorithm, dataset_size, trial_id = result
  except:
    print(f"Cannot extract information from model_name {model_name}.")
    import pdb; pdb.set_trace()

  for subdir in os.listdir(dataset_folder):
    try:
      if algorithm == "MC":
        X_state, X_plannedPath, Y, X_state_val, X_plannedPath_val, Y_val =\
          load_mc_dataset(f"{dataset_folder}/{subdir}/MC-FUNCAPPROX-0020-{dataset_size}")
      elif algorithm == "TD":
        X_state, X_plannedPath, Y, X_newState, X_state_val, X_plannedPath_val, Y_val, X_newState_val =\
          load_td_dataset(f"{dataset_folder}/{subdir}/TD-FUNCAPPROX-0020-{dataset_size}")
        # A boolean numpy array with shape (batch_size,)
        nonTerminal = np.logical_not(
          np.any(np.any(np.isnan(X_newState), axis=2), axis=1))

        if np.any(nonTerminal):
          # Adds TD value to non-terminal states
          Y_nonTerminal = model.predict(
            [X_newState[nonTerminal], X_plannedPath[nonTerminal]])

          Y[nonTerminal] += (0.9 * Y_nonTerminal)
      else:
        import pdb; pdb.set_trace()
    except:
      continue  # encountered a .gitkeep

    Y_pred = model.predict([X_state, X_plannedPath])
    if trial_id in subdir:
      training_errors.extend(np.square(Y_pred - Y).squeeze())
    else:
      dev_errors.extend(np.square(Y_pred - Y).squeeze())

  with open(log_filename, "a") as run_jobs_file:
    run_jobs_file.write(f"{model_name}, {np.mean(training_errors)}, {np.mean(dev_errors)}, {np.mean(errors)}\n")


if __name__ == "__main__":
  # python evaluate.py --no-use-config --model_name=MC-ROLLOUT-0020-16 --log_filename=run_jobs.log
  # python evaluate.py --no-use-config --model_name=MC-FUNCAPPROX-0020-16-01-A --log_filename=run_jobs.log
  # python evaluate.py --no-use-config --model_name=TD-FUNCAPPROX-0020-16-01-A-Tbase2 --log_filename=run_jobs.log
  parser = argparse.ArgumentParser()
  parser.add_argument("--use_config", dest="use_config", action="store_true")
  parser.add_argument('--no-use-config', dest='use_config', action="store_false")
  parser.set_defaults(use_config=True)
  parser.add_argument("--config_filename", nargs="?", default="config.ini")
  # Unlike training where dataset_folder directly contains the dataset like
  # datasets/9x9/MC-FUNCAPPROX-0020-16, this one needs to contain all of the
  # trials of datasets
  parser.add_argument("--dataset_folder", default="datasets")
  parser.add_argument("--model_name", default="")
  parser.add_argument("--truth_root_dir", default="truth/9x9")
  parser.add_argument("--log_filename", default="run_jobs.log")

  parser.add_argument
  args = parser.parse_args()


  if args.use_config:
    cp = ConfigParser()
    cp.read("config.ini")

    dataset_folder = cp["EVAL"].get("DATASET_FOLDER")
    model_name = cp["EVAL"].get("MODEL_NAME")
    truth_root_dir = cp["EVAL"].get("TRUTH_ROOT_DIR")
    log_filename = cp["EVAL"].get("LOG_FILENAME")
  else:
    dataset_folder = args.dataset_folder
    model_name = args.model_name
    truth_root_dir = args.truth_root_dir
    log_filename = args.log_filename

  if "FUNCAPPROX" in model_name:
    evaluate_function_approximation(dataset_folder, model_name, truth_root_dir, log_filename)
  else:
    evaluate_monte_carlo_rollout(dataset_folder, model_name, truth_root_dir, log_filename)