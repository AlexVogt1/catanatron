import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import (
    WeightedRandomPlayer, 
    CityRandomPlayer, 
    SettlementRandomPlayer, 
    LongestRoadRandomPlayer, 
    DevCardRandomPlayer, 
    DoNothingRandomPlayer
)

from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
from catanatron_gym.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)

# policy switching imports
# from catanatron_gym.policy_wrapper import PolicyWrapper
from typing import Literal, List, Final

COLOR_TO_RICH_STYLE = {
    Color.RED: "red",
    Color.BLUE: "blue",
    Color.ORANGE: "yellow",
    Color.WHITE: "white",
}


def rich_player_name(player):
    style = COLOR_TO_RICH_STYLE[player.color]
    return f"[{style}]{player}[/{style}]"


def rich_color(color):
    if color is None:
        return ""
    style = COLOR_TO_RICH_STYLE[color]
    return f"[{style}]{color.value}[/{style}]"

BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]
ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    # TODO: One for each tile (and abuse 1v1 setting).
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    # 4:1 with bank
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 3:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    # 2:1 with port
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))  # type: ignore
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    (ActionType.END_TURN, None),
]
ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)
ACTION_TYPES = [i for i in ActionType]


def to_action_type_space(action):
    return ACTION_TYPES.index(action.action_type)


def normalize_action(action):
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return normalized


def to_action_space(action):
    """maps action to space_action equivalent integer"""
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    """maps action_int to catantron.models.actions.Action"""
    # Get "catan_action" based on space action.
    # i.e. Take first action in playable that matches ACTIONS_ARRAY blueprint
    (action_type, value) = ACTIONS_ARRAY[action_int]
    catan_action = None
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            catan_action = action
            break  # return the first one
    assert catan_action is not None
    return catan_action


FEATURES = get_feature_ordering(num_players=2)
NUM_FEATURES = len(FEATURES)

# Highest features is NUM_RESOURCES_IN_HAND which in theory is all resource cards
HIGH = 19 * 5

def victory_point_reward(game,p0_color):
    pass
def simple_reward(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100

RandomPlayer_list = [CityRandomPlayer(Color.BLUE),
                         SettlementRandomPlayer(Color.BLUE),
                         LongestRoadRandomPlayer(Color.BLUE),
                         DevCardRandomPlayer(Color.BLUE),
                         DoNothingRandomPlayer(Color.BLUE)]

class PolicyWrapper:
    def __init__(self,game:Game):
        self.play_style_list : Final[List[object]] = RandomPlayer_list
        self.game = game
    
    def get_policy_action(self,chosen_policy: int, valid_actions):
         policy:Player = self.play_style_list[chosen_policy]
        #  print(policy)
         action = policy.decide(game=self.game, playable_actions=valid_actions)
         print(action)
        #  return action
    
    def get_num_policies(self):
        return len(self.play_style_list)

class CatanatronSwitchEnv(gym.Env):

    observation_space = spaces.Box(low=0, high=HIGH, shape=(NUM_FEATURES,), dtype=float)
    
    # action_space = spaces.Discrete(wrapped_policies.get_num_policies())

    def __init__(self, config=None):
        self.config = config or dict()
        self.invalid_action_reward = self.config.get("invalid_action_reward", -1)
        self.reward_function = self.config.get("reward_function", simple_reward)
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)
        self.invalid_actions_count = 0
        self.max_invalid_actions = 1000

        # TODO: Make self.action_space smaller if possible (per map_type)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=float
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=float
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=float
            )

        self.reset()
        self.wrapped_policies = PolicyWrapper(self.game)
        self.action_space = spaces.Discrete(self.wrapped_policies.get_num_policies())

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))
    
    def get_playable_actions(self):
        """
        Turns List[int] -> Iterable[actions]
        """
        # valid_actions
        return list(map(from_action_space, self.get_valid_actions(), self.game.state.playable_actions))

    def step(self, action):
        """
        Input: 
            int: integer of which policy is used to choose the action from list of policies (i.e. action is index)
        """
        chosen_policy = action
        print(chosen_policy)
        #get valid catan actions
        print(rich_player_name(self.p0))
        valid_catan_actions = self.game.state.playable_actions
        #get chosen action from chosen policy given the valid actions
        policy_action = self.wrapped_policies.get_policy_action(chosen_policy=chosen_policy, valid_actions= valid_catan_actions)
        print(policy_action)
        try:
            catan_action = from_action_space(policy_action, self.game.state.playable_actions)
        except Exception as e:
            # print("exception")
            self.invalid_actions_count += 0.1

            observation = self._get_observation()
            winning_color = self.game.winning_color()
            done = (
                winning_color is not None
                or self.invalid_actions_count > self.max_invalid_actions
            )
            terminated = winning_color is not None
            truncated = (
                self.invalid_actions_count > self.max_invalid_actions
                or self.game.state.num_turns >= TURNS_LIMIT
            )
            info = dict(valid_actions=self.get_valid_actions())
            return observation, self.invalid_action_reward, terminated, truncated, info

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function(self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

        pass

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self.invalid_actions_count = 0

        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        return observation, info

    def _get_observation(self):
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, self.p0.color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()