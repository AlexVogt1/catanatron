import random
import gymnasium as gym
from pprint import pprint
from catanatron import Color,RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer, DevCardRandomPlayer, DoNothingRandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

from catanatron.players.search import VictoryPointPlayer
from catanatron_gym.envs.catanatron_env import from_action_space
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3 import PPO
from catanatron_gym.features import create_sample_vector, get_feature_ordering,create_sample
from catanatron_gym.envs.catanatron_env import from_action_space
# FEATURES = get_feature_ordering
from pprint import pprint
import random
import os
# os.environ["POSTGRES_USER"]="catanatron"
# os.environ["POSTGRES_PASSWORD"]="victorypoint"
# os.environ["POSTGRES_DB"]="catanatron_db"


best_path = "./logs/Catan_Switch_Exp/exp_014/best_model.zip"
latest_path = "./logs/Catan_Switch_Exp/exp_004/latest_model_1000000_steps.zip"


env = gym.make("catanatron_gym:catanatron-switch-v1",config = {"enemies": [AlphaBetaPlayer(Color.RED)]})
observation, info = env.reset()
# pprint(vars(env.unwrapped))
# print(env.p0)
# print(observation)
model = PPO.load(path=best_path,env=env)
print(env.p0.color)
actions_list=[]
# print(model.policy)
for _ in range(100):
    # print(env.game.state.player_state)
    # your agent here (this takes random actions)
    # action = random.choice([0,1,2,3,4])
    action = model.predict(observation=observation,deterministic=True)
    # print(action[0].dtype)
    observation, reward, terminated, truncated, info = env.step(action[0])
    done = terminated or truncated
    env.render()
    if done:
        # pprint(vars(env))
        # print(done)
        # print(env.game.winning_color())
        # print(env.game.state.player_state)
        # print(info)
        actions_list.append(env.game.state.actions)
        observation, info = env.reset()
# print(info)
env.close()