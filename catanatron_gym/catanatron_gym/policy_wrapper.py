import random
from pprint import pprint
from catanatron_experimental.play import play_batch
from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer, CityRandomPlayer, SettlementRandomPlayer, LongestRoadRandomPlayer, DevCardRandomPlayer, DoNothingRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from typing import Literal, List, Final
from catanatron import Player, Color
from catanatron_experimental.cli.cli_players import register_player
from catanatron.game import Game, Action
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import *
from sb3_contrib.common.wrappers import ActionMasker
from typing import Iterable
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron_gym.rewards import reward_function, complex_reward

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
    
#----------------------Play-Style Player

def mask_function(env):
	mask = np.zeros(env.action_space.n)
	try:
		valid = env.unwrapped.get_valid_actions()
		for v in valid: mask[v] = 1
	except:
		return np.ones(len(mask))
	
	return mask

def direct_mask_function(valid_actions, size):
	mask = np.zeros(size)
	try:
		for v in valid_actions: mask[v] = 1
	except:
		return np.ones(len(mask))
	
	return mask

def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])

BEST_PATH = "logs/Catan_Baseline_rewards_Test/exp_001/best_model.zip"
LATEST_PATH = "logs/Catan_Baseline_rewards_Test/exp_001/latest_model_1000000_steps.zip"
@register_player("OWNREINFORCEMENT")
class OwnReinforcement(Player):
	i = 0
	def __init__(self, color, is_bot=True):
		self.init_model()
		super().__init__(color, is_bot)

	def init_model(self):
		self._mock_env = CatanatronEnv({
			"invalid_action_reward": -1,	
			"map_type": "BASE",
			"vps_to_win": 10,
			"enemies": [WeightedRandomPlayer(Color.RED)], # bot player is blue
			"reward_function": complex_reward,
			"representation": "vector"
		})

		# print(self._mock_env.observation_space)

		self._mock_env = ActionMasker(self._mock_env, mask_fn)
		# self._mock_env.observation_space = spaces.Box(0, 95, (614,), float)

		self._model = MaskablePPO.load(BEST_PATH, env=self._mock_env)
		# print(self._model.policy)
		print("Model initiated.")
		print("PATH: " +BEST_PATH)


	def decide(self, game: Game, playable_actions: Iterable[Action]):
		"""Should return one of the playable_actions.

		Args:
				game (Game): complete game state. read-only.
				playable_actions (Iterable[Action]): options to choose from
		Return:
				action (Action): Chosen element of playable_actions
		"""

		if len(playable_actions) == 1: return playable_actions[0]

		self._mock_env.unwrapped.players = self._mock_env.unwrapped.game.state.players
		self._mock_env.unwrapped.game = game

		env_action = self._model.predict(self._mock_env.unwrapped._get_observation(), deterministic=True,action_masks=mask_fn(self._mock_env))

		# print("action:")
		# print(playable_actions)
		# print([p.action_type for p in playable_actions])
		# print(self._mock_env.unwrapped.get_valid_actions())
		# print([from_action_space(v, playable_actions).action_type for v in self._mock_env.unwrapped.get_valid_actions()])
		# print(mask_function(self._mock_env))
		# print(env_action[0])
		# print(env_action[0], from_action_space(env_action[0], playable_actions))
		# print()
		return from_action_space(env_action[0], playable_actions)
	
		# ===== END YOUR CODE =====