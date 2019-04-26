import RL
import numpy as np


class MinMaxAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, prune=True):
        super().__init__(context, name)
        self.prune = prune
        self.transition_fn = self.context.env.metadata["transition_fn"]
        self.legal_actions_fn = self.context.env.metadata["legal_actions_fn"]
        self.winner_fn = self.context.env.metadata["winner_fn"]
        self.nplayers = self.context.env.metadata["nplayers"]
        self.board_size = self.context.env.metadata["board_size"]
        assert self.nplayers == 2, "Only 2 player minmax is supported right now"

    def act(self):
        my_id = self.context.frame_obs["player_turn"]
        board = self.context.frame_obs["board"]
        legal_moves, pi, Qs = self.get_legal_policy_Qs(my_id, board, prune_fn=lambda v: self.prune and v[my_id] > 0.999)
        # Qs_my = np.asarray([Q[my_id] for Q in Qs])
        # V = np.sum(pi * Qs_my)
        a = np.random.choice(len(legal_moves), p=pi)
        # print("{0}: Value of this state: {1}".format(self.name, V))
        return np.array(legal_moves[a])

    def get_legal_policy_Qs(self, turn, board, depth=0, prune_fn=None):
        legal_moves = self.legal_actions_fn(turn, board)
        batch_size = len(legal_moves)
        Qs = self.Qs([turn] * batch_size, [board] * batch_size, legal_moves, depth=depth, prune_fn=prune_fn)
        if len(Qs) < len(legal_moves):
            # This must be due to pruning
            legal_moves = legal_moves[:len(Qs)]
        Qs_for_turn_player = np.asarray([Q[turn] for Q in Qs])
        exps = np.exp(Qs_for_turn_player / self.context.alpha)
        pi = exps / np.sum(exps)
        return legal_moves, pi, Qs

    def Qs(self, turns, boards, actions, depth=0, prune_fn=None):
        boards_new = []
        turns_new = []
        for turn, board, action in zip(turns, boards, actions):
            turn_new, board_new = self.transition_fn(turn, board, action)
            boards_new.append(board_new)
            turns_new.append(turn_new)
        return self.Vs(turns_new, boards_new, depth=depth + 1, prune_fn=prune_fn)

    def Vs(self, turns, boards, depth=0, prune_fn=None):
        Vs = []
        for turn, board in zip(turns, boards):
            winner = self.winner_fn(turn, board)
            if winner is not None:
                V = np.array([0, 0]) if winner < 0 else (np.array([1, -1]) if winner == 0 else np.array([-1, 1]))
            else:
                legal_moves, pi, Qs = self.get_legal_policy_Qs(turn, board, depth, prune_fn=lambda v: self.prune and v[turn] > 0.999)
                V = np.sum(np.expand_dims(pi, -1) * Qs, axis=0)
            Vs.append(V)
            if prune_fn is not None and prune_fn(V):
                break
        return np.asarray(Vs)
