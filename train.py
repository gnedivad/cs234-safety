import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from tqdm import tqdm
from util import Grid2DMDP
from configparser import ConfigParser

from keras.backend import squeeze, sum as ksum
from keras.initializers import Identity
from keras.layers import (
  Activation, Concatenate, Dense, Embedding, Input, Masking, Lambda, Multiply,
  Permute, RepeatVector)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

MAX_PATH_LENGTH = 8


def build_policy_evaluation_model_with_attention():

  class RemoveMask(Lambda):
    def __init__(self):
      super(RemoveMask, self).__init__((lambda x, mask: x))
      self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
      return None

  # attention layer
  # Kind of jank but couldn't find a way to convert a state of shape (?, 2)
  # to the appropriate variable length and zero-padded tiled states
  state = Input(shape=(MAX_PATH_LENGTH, 2,), name="state")
  plannedPath = Input(shape=(MAX_PATH_LENGTH, 2), name="planned_path")

  # (?, 33, 2)
  H = Lambda(lambda x: x, name="H")(state)

  # (?, 33, 2)
  U = Lambda(lambda x: x, name="U")(plannedPath)

  # Will this mask out the int representation of 0?
  H = Masking(mask_value=0.0)(H)
  U = Masking(mask_value=0.0)(U)

  # (?, 33, 2)
  H_circ_U = Multiply()([H, U])

  # (?, 33, 6)
  H_U_H_circ_U = Concatenate(axis=-1)([H, U, H_circ_U])

  S = Dense(1,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros")(H_U_H_circ_U)
  S = Lambda(lambda x: squeeze(x, axis=-1))(S)
  S = Masking(mask_value=0.0)(S)  # only considers relevant indices
  S = Activation("softmax", name="S")(S)
  # (?, 33)
  S = RemoveMask()(S)

  # (?, 2, 33)
  S_tile = RepeatVector(2)(S)
  # (?, 33, 2)
  S_tile = Permute((2, 1))(S_tile)
  S_circ_U = Multiply()([S_tile, U])

  # (?, 2)
  h = Lambda(lambda x: x[:,0,:])(H)
  # (?, 2)
  u_tilda = Lambda(lambda x: ksum(x, axis=1))(S_circ_U)
  # (?, 2)
  h_circ_u_tilda = Multiply()([h, u_tilda])
  # (?, 6)
  G = Concatenate(axis=-1, name="G")([h, u_tilda, h_circ_u_tilda])

  # modeling-layer
  # ARCHITECTURE A
  fc1 = Dense(64,
              activation="relu",
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros")(G)
  fc2 = Dense(128,
              activation="relu",
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros")(fc1)
  fc3 = Dense(256,
              activation="relu",
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros")(fc2)
  fc4 = Dense(128,
              activation="relu",
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros")(fc3)
  fc5 = Dense(64,
              activation="relu",
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros")(fc4)
  out = Dense(1,
              activation="sigmoid",
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros")(fc5)

  model = Model(inputs=[state, plannedPath], outputs=out)
  return model


def preprocess_mc_datum(X, y):
  """
  Inputs:
  - X: A Python tuple, where the first element of the tuple represents the
    state and second element of the tuple represents the planned path.
  - y: A Python float that represents the label.

  Outputs:
  - stateArr: A numpy array with shape (MAX_PATH_LENGTH, 2).
  - plannedPathArr: A numpy array with shape (MAX_PATH_LENGTH, 2).
  - y: A Python float that represents the label.
  """
  state, plannedPath = X

  stateArr = np.array(state, dtype=np.float32)
  stateArr = np.repeat(np.expand_dims(stateArr, axis=0),
                       len(plannedPath),
                       axis=0)
  stateArr = pad_sequences(
    stateArr.T,
    maxlen=MAX_PATH_LENGTH,
    padding="post").T

  plannedPathArr = np.array(plannedPath, dtype=np.float32)
  plannedPathArr = pad_sequences(
    plannedPathArr.T,
    maxlen=MAX_PATH_LENGTH,
    padding="post").T

  return stateArr.astype(np.float32), plannedPathArr.astype(np.float32), y


def preprocess_td_datum(X):
  """
  Inputs:
  - X: A Python tuple, where the first element of the tuple contains the state;
    the second element of the tuple contains the planned path; the third
    element of the tuple contains the immediate reward; and the last element of
    the tuple contains the new state.

  Outputs:
  - stateArr: A numpy array with shape (MAX_PATH_LENGTH, 2).
  - plannedPathArr: A numpy array with shape (MAX_PATH_LENGTH, 2).
  - r: A Python float that represents the immediate reward.
  - newStateArr: A numpy array with shape (MAX_PATH_LENGTH, 2).
  """
  state, plannedPath, r, newState = X

  stateArr = np.array(state, dtype=np.float32)
  stateArr = np.repeat(np.expand_dims(stateArr, axis=0),
                       len(plannedPath),
                       axis=0)
  stateArr = pad_sequences(
    stateArr.T,
    maxlen=MAX_PATH_LENGTH,
    padding="post").T

  plannedPathArr = np.array(plannedPath, dtype=np.float32)
  plannedPathArr = pad_sequences(
    plannedPathArr.T,
    maxlen=MAX_PATH_LENGTH,
    padding="post").T

  # if {state} is a terminal state, then {newStateArr} should be an array of
  # nan's; otherwise, proceed as normal
  if newState is None:
    newStateArr = np.array(
      [np.nan] * MAX_PATH_LENGTH * 2).reshape(MAX_PATH_LENGTH, 2)
  else:
    newStateArr = np.array(newState, dtype=np.float32)
    newStateArr = np.repeat(np.expand_dims(newStateArr, axis=0),
                            len(plannedPath),
                            axis=0)
    newStateArr = pad_sequences(
      newStateArr.T,
      maxlen=MAX_PATH_LENGTH,
      padding="post").T

  return (
    stateArr.astype(np.float32),
    plannedPathArr.astype(np.float32),
    r,
    newStateArr.astype(np.float32),)


def load_mc_dataset(dataset_folder):
  dataset_files = os.listdir(dataset_folder)
  read_train = False
  read_val = False

  if ("X_state.npy" in dataset_files
      and "X_plannedPath.npy" in dataset_files
      and "Y.npy" in dataset_files):
    X_state = np.load(f"{dataset_folder}/X_state.npy")
    X_plannedPath = np.load(f"{dataset_folder}/X_plannedPath.npy")
    Y = np.load(f"{dataset_folder}/Y.npy")
    read_train = True

  if ("X_state_val.npy" in dataset_files
      and "X_plannedPath_val.npy" in dataset_files
      and "Y_val.npy" in dataset_files):
    X_state_val = np.load(f"{dataset_folder}/X_state_val.npy")
    X_plannedPath_val = np.load(f"{dataset_folder}/X_plannedPath_val.npy")
    Y_val = np.load(f"{dataset_folder}/Y_val.npy")
    read_val = True

  if read_train and read_val:
    # Both training and validation sets were read from file
    return X_state, X_plannedPath, Y, X_state_val, X_plannedPath_val, Y_val 

  X_state = None
  X_plannedPath = None
  Y = None

  X_state_val = None
  X_plannedPath_val = None
  Y_val = None

  with open(f"{dataset_folder}/dataset.p", "rb") as f:
    data = pickle.load(f)

  for datum in data:
    stateArr, plannedPathArr, y = preprocess_mc_datum(*datum)

    # Decides whether to add to validation set
    r, c = stateArr[0]
    is_training = True  # r % 2 == c % 2

    if X_state is None and is_training:
      X_state = np.expand_dims(stateArr, axis=0)
      X_plannedPath = np.expand_dims(plannedPathArr, axis=0)
      Y = np.expand_dims(y, axis=0)
    elif is_training:
      X_state = np.vstack((X_state, [stateArr]))
      X_plannedPath = np.vstack((X_plannedPath, [plannedPathArr]))
      Y = np.vstack((Y, [y]))
    elif X_state_val is None:
      X_state_val = np.expand_dims(stateArr, axis=0)
      X_plannedPath_val = np.expand_dims(plannedPathArr, axis=0)
      Y_val = np.expand_dims(y, axis=0)
    else:
      X_state_val = np.vstack((X_state_val, [stateArr]))
      X_plannedPath_val = np.vstack((X_plannedPath_val, [plannedPathArr]))
      Y_val = np.vstack((Y_val, [y]))

  np.save(f"{dataset_folder}/X_state.npy", X_state)
  np.save(f"{dataset_folder}/X_plannedPath.npy", X_plannedPath)
  np.save(f"{dataset_folder}/Y.npy", Y)
  np.save(f"{dataset_folder}/X_state_val.npy", X_state_val)
  np.save(f"{dataset_folder}/X_plannedPath_val.npy", X_plannedPath_val)
  np.save(f"{dataset_folder}/Y_val.npy", Y_val)

  return X_state, X_plannedPath, Y, X_state_val, X_plannedPath_val, Y_val


def load_td_dataset(dataset_folder):
  dataset_files = os.listdir(dataset_folder)
  read_train = False
  read_val = False

  if ("X_state.npy" in dataset_files
      and "X_plannedPath.npy" in dataset_files
      and "R.npy" in dataset_files
      and "X_newState.npy" in dataset_files):
    X_state = np.load(f"{dataset_folder}/X_state.npy")
    X_plannedPath = np.load(f"{dataset_folder}/X_plannedPath.npy")
    R = np.load(f"{dataset_folder}/R.npy")
    X_newState = np.load(f"{dataset_folder}/X_newState.npy")
    read_train = True

  if ("X_state_val.npy" in dataset_files
      and "X_plannedPath_val.npy" in dataset_files
      and "R_val.npy" in dataset_files
      and "X_newState_val.npy" in dataset_files):
    X_state_val = np.load(f"{dataset_folder}/X_state_val.npy")
    X_plannedPath_val = np.load(f"{dataset_folder}/X_plannedPath_val.npy")
    R_val = np.load(f"{dataset_folder}/R_val.npy")
    X_newState_val = np.load(f"{dataset_folder}/X_newState_val.npy")
    read_val = True

  if read_train and read_val:
    # Both training and validation sets were read from file
    return (
      X_state,
      X_plannedPath,
      R,
      X_newState,
      X_state_val,
      X_plannedPath_val,
      R_val,
      X_newState_val,)

  X_state = None
  X_plannedPath = None
  R = None
  X_newState = None

  X_state_val = None
  X_plannedPath_val = None
  R_val = None
  X_newState_val = None

  with open(f"{dataset_folder}/dataset.p", "rb") as f:
    data = pickle.load(f)

  for datum in data:
    stateArr, plannedPathArr, r, newStateArr = preprocess_td_datum(datum)
    
    # Decides whether to add to validation set
    row, col = stateArr[0]
    is_training = True  # row % 2 == col % 2

    if X_state is None and is_training:
      X_state = np.expand_dims(stateArr, axis=0)
      X_plannedPath = np.expand_dims(plannedPathArr, axis=0)
      R = np.expand_dims(r, axis=0)
      X_newState = np.expand_dims(newStateArr, axis=0)
    elif is_training:
      X_state = np.vstack((X_state, [stateArr]))
      X_plannedPath = np.vstack((X_plannedPath, [plannedPathArr]))
      R = np.vstack((R, [r]))
      X_newState = np.vstack((X_newState, [newStateArr]))
    elif X_state_val is None:
      X_state_val = np.expand_dims(stateArr, axis=0)
      X_plannedPath_val = np.expand_dims(plannedPathArr, axis=0)
      R_val = np.expand_dims(r, axis=0)
      X_newState_val = np.expand_dims(newStateArr, axis=0)
    else:
      X_state_val = np.vstack((X_state_val, [stateArr]))
      X_plannedPath_val = np.vstack((X_plannedPath_val, [plannedPathArr]))
      R_val = np.vstack((R_val, [r]))
      X_newState_val = np.vstack((X_newState_val, [newStateArr]))

  np.save(f"{dataset_folder}/X_state.npy", X_state)
  np.save(f"{dataset_folder}/X_plannedPath.npy", X_plannedPath)
  np.save(f"{dataset_folder}/R.npy", R)
  np.save(f"{dataset_folder}/X_newState.npy", X_newState)
  np.save(f"{dataset_folder}/X_state_val.npy", X_state_val)
  np.save(f"{dataset_folder}/X_plannedPath_val.npy", X_plannedPath_val)
  np.save(f"{dataset_folder}/R_val.npy", R_val)
  np.save(f"{dataset_folder}/X_newState_val.npy", X_newState_val)

  return (
    X_state,
    X_plannedPath,
    R,
    X_newState,
    X_state_val,
    X_plannedPath_val,
    R_val,
    X_newState_val,)



def train():
  # python train.py --no-use-config --algorithm=td --model_name=TD-FUNCAPPROX-0020-16-01-A-Tbase4 --dataset_folder=datasets/9x9-01/TD-FUNCAPPROX-0020-16 --temperature_base=4
  parser = argparse.ArgumentParser()
  parser.add_argument("--use_config", dest="use_config", action="store_true")
  parser.add_argument('--no-use-config', dest='use_config', action="store_false")
  parser.set_defaults(use_config=True)
  parser.add_argument("--config_filename", nargs="?", default="config.ini")
  parser.add_argument("--algorithm", default="mc")
  parser.add_argument("--model_name", default="MC-FUNCAPPROX-0020-16-A")
  parser.add_argument("--dataset_folder", default="datasets/9x9/MC-FUNCAPPROX-0020-16")
  parser.add_argument("--temperature_base", default=2.718, type=float)
  parser.add_argument
  args = parser.parse_args()


  if args.use_config:
    cp = ConfigParser()
    cp.read("config.ini")

    algorithm = cp["TRAIN"].get("ALGORITHM")
    model_name = cp["TRAIN"].get("MODEL_NAME")
    dataset_folder = cp["TRAIN"].get("DATASET_FOLDER")

    # for TD
    temperature_base = cp["TRAIN"].getfloat("TEMPERATURE_BASE")
  else:
    algorithm = args.algorithm
    model_name = args.model_name
    dataset_folder = args.dataset_folder
    temperature_base = args.temperature_base

  model = build_policy_evaluation_model_with_attention()

  if algorithm == "mc":
    X_state, X_plannedPath, Y, X_state_val, X_plannedPath_val, Y_val =\
      load_mc_dataset(dataset_folder)

    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=0.005,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08,
                                 decay=0.001))

    model.fit(x=[X_state, X_plannedPath],
              y=Y,
              # Comment or uncomment depending on whether the dataset has
              # validation data
              # validation_data=([X_state_val[:1000], X_plannedPath_val[:1000]], Y_val[:1000]),
              batch_size=100,
              epochs=500,
              verbose=1)

###############################################################################
  elif algorithm == "td" or algorithm == "her":
    X_state, X_plannedPath, r, X_newState, X_state_val, X_plannedPath_val, r_val, X_newState_val =\
      load_td_dataset(dataset_folder)

    # FOR DEBUGGING ###########################################################
    # keep = []
    # for x_plannedPath in X_plannedPath:
    #   goal_state_idx = np.argmax(np.all(x_plannedPath == 0, axis=1)) - 1
    #   goal_state = x_plannedPath[goal_state_idx]
    #   if np.all(goal_state == [5, 2]):
    #     keep.append(True)
    #   else:
    #     keep.append(False)
    # X_state = X_state[keep]
    # X_plannedPath = X_plannedPath[keep]
    # r = r[keep]
    # X_newState = X_newState[keep]
    ###########################################################################

    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=0.001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08,
                                 decay=0.0))

    # Early on during training, we want to draw samples with states nearby to
    # goal states
    manhattan_dists = []
    for x_state, x_plannedPath in zip(X_state, X_plannedPath):
      goal_state_idx = np.argmax(np.all(x_plannedPath == 0, axis=1)) - 1
      goal_state = x_plannedPath[goal_state_idx]
      manhattan_dist = np.sum(np.abs(x_state[0] - goal_state))
      manhattan_dists.append(manhattan_dist)

    # num_batches = 50000
    batch_size = 100
    num_batches = max(int(X_state.shape[0] / batch_size * 500), 5000)  # 500 => equivalent to 500 epochs for MC
    gamma = Grid2DMDP().discount()
    for it in tqdm(range(num_batches)):
      temperature = max(np.log(it) / np.log(temperature_base), 1.0)
      numer = np.exp(-np.array(manhattan_dists) / temperature)
      denom = np.sum(numer)

      sample_indices = np.random.choice(np.arange(X_state.shape[0]),
                                        size=batch_size,
                                        replace=batch_size > X_state.shape[0],
                                        p=(numer / denom))
      X_state_batch = X_state[sample_indices]
      X_plannedPath_batch = X_plannedPath[sample_indices]
      Y_batch = r[sample_indices]
      X_newState_batch = X_newState[sample_indices]

      # A boolean numpy array with shape (batch_size,)
      nonTerminal = np.logical_not(
        np.any(np.any(np.isnan(X_newState_batch), axis=2), axis=1))

      if np.any(nonTerminal):
        # Adds TD value to non-terminal states
        Y_nonTerminal = model.predict(
          [X_newState_batch[nonTerminal], X_plannedPath_batch[nonTerminal]])

        Y_batch[nonTerminal] += (gamma * Y_nonTerminal)
      model.fit(x=[X_state_batch, X_plannedPath_batch],
                y=Y_batch,
                batch_size=batch_size,
                epochs=1,
                verbose=0)
###############################################################################

  # Visualizes attention
  # attn_model = Model(inputs=model.input, outputs=model.get_layer("S").output)
  # stateArr, plannedPathArr, _ = preprocess(((8, 4), [(8, 4), (7, 4), (6, 3)]), None)
  # attn = attn_model.predict([np.expand_dims(stateArr, axis=0), np.expand_dims(plannedPathArr, axis=0)])

  # model.save(f"models/{model_name}.h5")
  # model = load_model(f"models/{model_name}.h5")


  def plot_reward_for_planned_path(testPlannedPath):
    Y_pred = np.zeros((9, 9))

    for r in range(9):
      for c in range(9):
        stateArr, plannedPathArr, _ = preprocess_mc_datum(((r, c), testPlannedPath), None)
        y_pred = model.predict([np.expand_dims(stateArr, axis=0),
                                np.expand_dims(plannedPathArr, axis=0)])
        Y_pred[r, c] = y_pred
    plt.figure()
    plt.imshow(Y_pred, cmap="gray", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(testPlannedPath[-1])
    plt.text(3.86, 8.1, "S", color="white", fontsize=14)
    plt.plot([testPlannedPath[-1][1]], [testPlannedPath[-1][0]], marker="*", markersize=14)
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
    # plt.show()
    plt.savefig(f"results/{model_name}/{str(testPlannedPath[-1][0])},{str(testPlannedPath[-1][1])}.png")
    return Y_pred

  import os
  if not os.path.exists(f"results/{model_name}"):
    os.makedirs(f"results/{model_name}")
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

  for testPlannedPath in testPlannedPaths:
    plot_reward_for_planned_path(testPlannedPath)

  model.save_weights(f"models/{model_name}.h5")


if __name__ == "__main__":
  train()
