import numpy as np

class TicTacToe:
  def __init__(self):
    self.row_count = 3
    self.column_count = 3
    self.action_size = self.row_count * self.column_count
  
  # update the state played by player's action
  def get_next_state(self, state, action, player):
    row = action // self.column_count
    column = action % self.column_count
    state[row, column] = player
    return state

  # return the moves which == 0
  def get_valid_moves(self, state):
    return (state.reshape(-1)==0).astype(np.uint8)

  # check if anyone won after an action
  def check_win(self, state, action):
    if action == None:
      return False
    row = action // self.column_count
    column = action % self.column_count
    player = state[row, column]

    return (
        np.sum(state[row, :]) == player * self.column_count
        or np.sum(state[:, column]) == player * self.row_count
        or np.sum(np.diag(state)) == player * self.row_count
        or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
    )

  # check whether need to continue
  # terminated = True -> stop the game
  # terminated = False -> continue the game
  # value = 1 -> one player won
  # value = 0 -> no player won
  def get_value_and_terminated(self, state, action):
    if self.check_win(state, action):
      return 1, True
    if np.sum(self.get_valid_moves(state)) == 0:
      return 0, True
    return 0, False

  # switch player
  def get_opponent(self, player):
    return -player

  # switch player's value
  def get_opponent_value(self, value):
    return -value

  def change_perspective(self, state, player):
    return state * player