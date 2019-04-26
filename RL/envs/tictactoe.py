import gym
import numpy as np


class TicTacToe(gym.Env):
    metadata = {
        "render.modes": "ansi",
        "player_symbols": [' ', 'X', 'O', '+', '*', '#']
    }

    def __init__(self, board_size=3, nplayers=2):
        self.board_size = board_size
        self.nplayers = nplayers
        self.player_symbols = self.metadata["player_symbols"]
        self.metadata["board_size"] = board_size
        self.metadata["nplayers"] = nplayers
        self.metadata["winner_fn"] = self.winner_fn
        self.all_moves = [[runner, my_context] for runner in range(board_size) for my_context in range(board_size)]
        self.metadata["legal_actions_fn"] = self.legal_actions_fn
        self.metadata["transition_fn"] = self.transition_fn
        self.action_space = gym.spaces.MultiDiscrete([board_size, board_size])
        self.observation_space = gym.spaces.Dict({"player_turn": gym.spaces.Discrete(
            nplayers), "board": gym.spaces.Box(low=0, high=1, shape=(board_size, board_size, nplayers + 1), dtype=np.uint8)})
        self._state = {
            "player_turn": 0,
            "board": np.array([[[1, 0, 0]] * board_size] * board_size)
        }
        self._info = {}

    def winner_fn(self, turn, board):
        """Returns player id of winner. None means not finished. -1 means tie"""
        b = board
        for p in range(self.nplayers):
            d = False
            # check rows:
            d = np.any([np.all(b[runner, :, p + 1]) for runner in range(self.board_size)])
            # check cols:
            if not d:
                d = np.any([np.all(b[:, my_context, p + 1]) for my_context in range(self.board_size)])
            # check diagonals:
            if not d:
                d = np.all(np.diag(b[:, :, p + 1])) or np.all(np.diag(b[:, :, p + 1][:, ::-1]))
            if d:
                return p
        # check tie:
        if not np.any(b[:, :, 0]):
            return -1
        return None

    def legal_actions_fn(self, turn, board):
        return list(filter(lambda m: board[m[0], m[1], 0] == 1, self.all_moves))

    def transition_fn(self, turn, board, action):
        board_new = np.copy(board)
        assert board_new[action[0], action[1], 0] == 1, "Illegal action {0}".format(action)
        board_new[action[0], action[1], :] = np.zeros(self.nplayers + 1)
        board_new[action[0], action[1], turn + 1] = 1
        turn_new = (turn + 1) % self.nplayers
        return turn_new, board_new

    def reset(self):
        self._state = {
            "player_turn": 0,
            "board": np.array([[[1, 0, 0]] * self.board_size] * self.board_size)
        }
        self._info = {"winner": None}
        return self._state

    def step(self, action):
        assert self.action_space.contains(
            action), "Where are you trying to play? Out of board? You tried to play at {0}".format(action)
        reward = 0
        done = False
        if self._state["board"][action[0], action[1], 0] == 1:
            turn, board = self.transition_fn(self._state["player_turn"], self._state["board"], action)
            self._state["board"] = board
            self._state["player_turn"] = turn
            winner = self.winner_fn(turn, board)
            if winner is not None:
                done = True
                reward = 0 if winner < 0 else (1 if winner == 0 else -1)
            self._info["winner"] = winner
        else:
            # print("Illegal move. Play again")
            pass
        return self._state, reward, done, self._info

    def to_symbols_board(self, board):
        return np.asarray(self.player_symbols)[np.argmax(board, axis=-1)]

    def render(self, mode='human'):
        b = self._state["board"]
        turn = self._state["player_turn"]
        desc_s = "Player {0} i.e {1} Turn".format(turn, self.player_symbols[turn + 1])
        line_s = '-' * (2 * self.board_size + 1)
        sym_board = self.to_symbols_board(b)
        board_s = '\n'.join([('|' + '|'.join(sym_board[row, :]) + '|') for row in range(self.board_size)])
        w = self._info["winner"]
        result_s = "Winner: " + ("" if w is None else ("Tie" if w < 0 else self.player_symbols[w + 1]))
        s = '\n'.join([desc_s, line_s, board_s, line_s, result_s, ''])
        if mode == 'human':
            print(s)
        elif mode == 'ansi':
            return s
        else:
            raise NotImplementedError(
                "{0} render mode is not supported".format(mode))
