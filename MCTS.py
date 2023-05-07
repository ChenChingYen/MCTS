from Node import Node
import numpy as np

class MCTS:
  def __init__(self, game, args):
    self.game = game
    self.args = args

  def search(self, state):
    # take the current state as the root node
    root = Node(self.game, self.args, state)
    # define root
    # limit the search under 1000
    for search in range(self.args['num_searches']):
      node = root
      
      # selection
      while node.is_fully_expanded():
        node = node.select()

      value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
      value = self.game.get_opponent_value(value)
      
      # expansion
      if not is_terminal:
        node = node.expand()
        # simulation
        value = node.simulate()
      
      # backpropagation
      node.backpropagate(value)

    action_probs = np.zeros(self.game.action_size)

    for child in root.children:
      action_probs[child.action_taken] = child.visit_count

    action_probs /= np.sum(action_probs)
    return action_probs