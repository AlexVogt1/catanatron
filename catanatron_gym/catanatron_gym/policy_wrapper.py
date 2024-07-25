import random
from pprint import pprint
from catanatron_experimental.play import play_batch
from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer, CityRandomPlayer, SettlementRandomPlayer, LongestRoadRandomPlayer, DevCardRandomPlayer, DoNothingRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from typing import Literal, List, Final

RandomPlayer_list = List[CityRandomPlayer(Color.WHITE),
                         SettlementRandomPlayer(Color.WHITE),
                         LongestRoadRandomPlayer(Color.WHITE),
                         DevCardRandomPlayer(Color.WHITE),
                         DoNothingRandomPlayer(Color.WHITE)]

class PolicyWrapper:
    def __init__(self,game:Game):
        self.play_style_list : Final[List[any]] = RandomPlayer_list
        self.game = game
    
    def get_policy_action(self,chosen_policy: int, valid_actions):
         policy:Player = self.play_style_list[chosen_policy]
         action = policy.decide(game=self.game, playable_actions=valid_actions)
         return action
    
    def get_num_policies(self)-> int:
        return len(self.play_style_list)