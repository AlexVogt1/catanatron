import random

from catanatron.models.player import Player
from catanatron.models.actions import ActionType
# from catanatron_experimental.cli.cli_players import register_player

WEIGHTS_BY_ACTION_TYPE = {
    ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.BUY_DEVELOPMENT_CARD: 100,
}

RANDOM_PLAYER = {
    ActionType.ACCEPT_TRADE:1,
    ActionType.BUILD_CITY:1,
    ActionType.BUILD_ROAD:1,
    ActionType.BUILD_SETTLEMENT:1,
    ActionType.BUY_DEVELOPMENT_CARD:1,
    ActionType.CANCEL_TRADE:1,
    ActionType.CONFIRM_TRADE:1,
    ActionType.DISCARD:1,
    ActionType.END_TURN:1,
    ActionType.MARITIME_TRADE:1,
    ActionType.MOVE_ROBBER:1,
    ActionType.OFFER_TRADE:1,
    ActionType.PLAY_KNIGHT_CARD:1,
    ActionType.PLAY_MONOPOLY:1,
    ActionType.PLAY_ROAD_BUILDING:1,
    ActionType.PLAY_YEAR_OF_PLENTY:1,
    ActionType.REJECT_TRADE:1,
    ActionType.ROLL:1,
}

WEIGHTS_FOR_CITY_SETTLEMENT_PLAYER = {
    # ActionType.BUILD_CITY: 10000,
    ActionType.BUILD_SETTLEMENT: 1000,
    ActionType.MARITIME_TRADE:0,
    # ActionType.BUILD_ROAD: 100,
    # ActionType.BUY_DEVELOPMENT_CARD: 10,
}

WEIGHTS_FOR_CITY_PLAYER = {
    ActionType.BUILD_CITY: 10000,
    # ActionType.BUILD_SETTLEMENT: 1000,
    # ActionType.BUILD_ROAD: 100,
    ActionType.BUY_DEVELOPMENT_CARD: 0,
    ActionType.MARITIME_TRADE:0
}

WEIGHTS_FOR_SETTLEMENT_PLAYER = {
    ActionType.BUILD_SETTLEMENT: 10000,
    # ActionType.BUILD_CITY: 1000,
    # ActionType.BUILD_ROAD: 100,
    ActionType.BUY_DEVELOPMENT_CARD: 0,
    ActionType.MARITIME_TRADE:0
}

WEIGHTS_FOR_LONGEST_ROAD_PLAYER ={
    ActionType.BUILD_ROAD: 10000,
    ActionType.PLAY_ROAD_BUILDING: 1000,
    # ActionType.BUY_DEVELOPMENT_CARD: 100,
    # ActionType.BUILD_SETTLEMENT: 1,
    ActionType.MARITIME_TRADE:0,
}

WEIGHTS_FOR_DEV_CARD_PLAYER = {
    ActionType.BUY_DEVELOPMENT_CARD: 100000,
    ActionType.PLAY_KNIGHT_CARD: 10000,
    # ActionType.PLAY_YEAR_OF_PLENTY:1000,
    # ActionType.PLAY_MONOPOLY: 100,
    ActionType.PLAY_ROAD_BUILDING:0,
    ActionType.MARITIME_TRADE:0,
}

WEIGHTS_FOR_DO_NOTHING_PLAYER = {
    # ActionType.BUILD_SETTLEMENT:0,
    ActionType.END_TURN: 10000,
    ActionType.MARITIME_TRADE:0,
}
class TestDoNothingRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to use dev Cards and largest army to win.
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_DO_NOTHING_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)

class WeightedRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution
    to actions that are likely better (cities > settlements > dev cards).
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_BY_ACTION_TYPE.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)
    
class CitySettlementRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to build Cities and Settlements to win.
    (Cities > Settelments > Road )
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_CITY_SETTLEMENT_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)
    
class CityRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to build Cities and Settlements to win.
    (Cities > Settelments > Road )
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_CITY_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)
    
class SettlementRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to build Cities and Settlements to win.
    (Cities > Settelments > Road )
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_SETTLEMENT_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)
    
class LongestRoadRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to use longest road to win.
    (Road > Road Dev Card > Dev Card)
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_LONGEST_ROAD_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)

class DevCardRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to use dev Cards and largest army to win.
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_DEV_CARD_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)

class DoNothingRandomPlayer(Player):
    """
    Player that decides at random, but skews distribution to roughly mimic a
    player that prefers to use dev Cards and largest army to win.
    """

    def decide(self, game, playable_actions):
        # print(playable_actions)
        bloated_actions = []
        for action in playable_actions:
            weight = WEIGHTS_FOR_DO_NOTHING_PLAYER.get(action.action_type, 1)
            bloated_actions.extend([action] * weight)

        return random.choice(bloated_actions)
    