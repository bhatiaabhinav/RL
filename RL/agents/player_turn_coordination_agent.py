import RL
from typing import List
import random


class PlayerTurnCoordinationAgent(RL.Agent):
    def __init__(self, context: RL.Context, name, list_player_agents: List[RL.Agent], shuffle=True):
        super().__init__(context, name)
        self.list_player_agents = list_player_agents
        self.shuffle = shuffle

    def set_enable_all(self, enabled):
        for p in self.list_player_agents:
            p.enabled = enabled

    def pre_episode(self):
        if self.shuffle:
            random.shuffle(self.list_player_agents)
        self.set_enable_all(False)
        turn = self.context.frame_obs["player_turn"]
        self.list_player_agents[turn].enabled = True

    def pre_act(self):
        assert "player_turn" in self.context.frame_obs
        self.set_enable_all(False)
        turn = self.context.frame_obs["player_turn"]
        self.list_player_agents[turn].enabled = True

    def post_act(self):
        for p in self.list_player_agents:
            p.enabled = True
