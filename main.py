from flask import Flask, request, jsonify
from TicTacToe import TicTacToe
from MCTS import MCTS
import json
import numpy as np
from flask_cors import CORS, cross_origin

args = {
    'C': 1.41,
    'num_searches': 1000
}

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/mcts', methods=['GET'])
@cross_origin()
def predict():
    current_state = request.args.get('state')
    current_player = request.args.get('player')

    tictactoe = TicTacToe()

    mcts = MCTS(tictactoe, args)
    state = np.array(json.loads(current_state))
    player = int(current_player)
    
    if player == -1:
        neutral_state = tictactoe.change_perspective(state, player)
    else:
        neutral_state = state

    # pass current
    mcts_probs = mcts.search(neutral_state)
    # take the index of maximum probs from the mcts output
    action = int(np.argmax(mcts_probs))

    next_state = tictactoe.get_next_state(state, action, player)

    return jsonify({
        'next_state': next_state.tolist(),
        'action': action,
        'mcts_probs': mcts_probs.tolist()
        })

if __name__ == '__main__':
    app.run(debug=True)
