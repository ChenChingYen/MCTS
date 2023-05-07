import numpy as np
import math

class Node:
  def __init__(self, game, args, state, parent=None, action_taken=None):
    self.game = game
    self.args = args
    self.state = state
    self.parent = parent
    self.action_taken = action_taken

    self.children = []
    self.expandable_moves = game.get_valid_moves(state)

    self.visit_count = 0
    self.value_sum = 0

  def is_fully_expanded(self):
    return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

  # select the best child based on the UCB score
  def select(self):
    best_child = None
    best_ucb = -np.inf

    for child in self.children:
      ucb = self.get_ucb(child)
      if ucb > best_ucb:
        best_child = child
        best_ucb = ucb

    return best_child

  # calculate UCB score
  def get_ucb(self, child):
    q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
    return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

  # expansion
  def expand(self):
    # # randomly choose an action from the valid moves
    action = np.random.choice(np.where(self.expandable_moves == 1)[0])
    # # remove that action from valid moves
    self.expandable_moves[action] = 0
    # make a copy for child state to further expand 
    child_state = self.state.copy()
    # update it to next stage based on the action taken
    child_state = self.game.get_next_state(child_state, action, 1)
    child_state = self.game.change_perspective(child_state, player = -1)
    # create child node
    child = Node(self.game, self.args, child_state, self, action)
    self.children.append(child)
    return child

  # simulation
  def simulate(self):
    # is_terminal = True -> stop the game
    # is_terminal = False -> continue the game
    # value = 1 -> one player won
    # value = 0 -> no player won
    value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
    # switch the whole board from player 1 = 1 to player 1 = -1
    value = self.game.get_opponent_value(value)

    # if the game ended, return the last value (win or draw)
    if is_terminal:
      return value

    # else copy a new state
    rollout_state = self.state.copy()
    # now the computer is assigned as player 1
    rollout_player = 1

    # make random action till the leaf node
    while True:
      # check valid move for a child
      valid_moves = self.game.get_valid_moves(rollout_state)
      # randomly pick an action
      action = np.random.choice(np.where(valid_moves == 1)[0])
      # check the next state of the child if that random action taken
      rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
      # check the game is ended
      value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
      if is_terminal:
        # if ended, convert back the value to player, and return the either 1 (win) or 0 (lose)
        if rollout_player == -1:
          value = self.game.get_opponent_value(value)
        return value
      # else switch the player
      rollout_player = self.game.get_opponent(rollout_player)

  # propagate to the root node and sum up all the values passed by
  def backpropagate(self, value):
    # every nodes has its value_sum
    self.value_sum += value
    # when getting back propagated once, visit count is sum up
    self.visit_count += 1
    
    value = self.game.get_opponent_value(value)

    # if it is not root node, pass the value back to parent nodes to pile up
    if self.parent is not None:
      self.parent.backpropagate(value)