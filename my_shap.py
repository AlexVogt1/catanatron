import random
import gymnasium as gym
from pprint import pprint
from catanatron import Color,RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer, DevCardRandomPlayer, DoNothingRandomPlayer,CityRandomPlayer,SettlementRandomPlayer,LongestRoadRandomPlayer
from catanatron_gym.rewards import reward_function
from catanatron_gym.features import create_sample_vector, create_sample
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
import ast

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
import pandas as pd
from shaprpy import explain
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
import matplotlib.pyplot as plt
# !pip install seaborn
import seaborn as sns
import torch
import torch.nn as nn
# from torch import float32
import shap
# from torch.autograd import Variable
import os
import re
import argparse
# from rpy2.robjects.vectors import StrVector, ListVector
# from rpy2.rinterface import NULL, NA

class sb3Wrapper(nn.Module):
    def __init__(self, model):
        super(sb3Wrapper,self).__init__()
        self.extractor = model.policy.mlp_extractor
        self.policy_net = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net
        print(self.action_net)

    def forward(self,x):
        x = self.policy_net(x)
        x = self.action_net(x)
        return torch.tensor(torch.argmax(x,keepdim=True),dtype=torch.float64)

def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])

def run_rl_agent(env_config, agent_path= "../logs/Catan_Baseline_rewards_Test/exp_001/latest_model_1000000_steps.zip"):
    # inits for data collection
    try:
        env = gym.make("catanatron_gym:catanatron-v1", env_config)  
        print("made env with config")
    except:
        env = gym.make("catanatron_gym:catanatron-v1")  
        print("made env without config")
    env = ActionMasker(env=env,action_mask_fn=mask_fn)
    observation, info = env.reset()

    model = MaskablePPO.load(path= agent_path,env=env)
    for _ in range(1000):
        action = model.predict(observation=observation,action_masks=mask_fn(env),deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
        if done:
            print(env.game.winning_color())
            observation, info = env.reset()
    # print(info)
    env.close()

def run_switch_agent(env_config, agent_path="./logs/Catan_Switch_Exp/exp_014/best_model.zip"):
    best_path = "../logs/Catan_Switch_Exp/exp_014/best_model.zip"
    latest_path = "../logs/Catan_Switch_Exp/exp_004/latest_model_1000000_steps.zip"
    obs_list = []
    info_list =[]
    action_list =[]
    record_list =[]
    data = []
    features = None
    try:
        env = gym.make("catanatron_gym:catanatron-switch-v1",config = env_config)
        print("made env using config")
    except:
        env = gym.make("catanatron_gym:catanatron-switch-v1")
        print('made env without config')

    observation, info = env.reset()
    model = PPO.load(path=agent_path,env=env)
    print(env.unwrapped.game)
    print(env.p0.color)
    # print(model.policy)
    for ep in range(10):
        for step in range(1000):
            action = model.predict(observation=observation,deterministic=True)
            # obs_list.append(observation)
            # action_list.append(action)
            record = create_sample(env.game,env.p0.color)
            features =sorted(record.keys()) # get list of feature names in same order as observation
            data.append([ep,step,observation,action[0]])
            observation, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            if done:
                # pprint(vars(env))
                # print(done)
                # print(env.game.winning_color())
                # print(env.game.state.player_state)
                # print(info)
                info_list.append(info)
                observation, info = env.reset()
                break
    # print(info)
    env.close()
    # print(record_list)
    return data, info_list, features

def get_group(features, group):
     # group = {
    #     'a':['BANK_BRICK', 'BANK_DEV_CARDS', 'BANK_ORE', 'BANK_SHEEP', 'BANK_WHEAT', 'BANK_WOOD'],
    #     'b':['EDGE(0, 1)_P0_ROAD', 'EDGE(0, 1)_P1_ROAD', 'EDGE(0, 20)_P0_ROAD', 'EDGE(0, 20)_P1_ROAD', 'EDGE(0, 5)_P0_ROAD', 'EDGE(0, 5)_P1_ROAD', 'EDGE(1, 2)_P0_ROAD', 'EDGE(1, 2)_P1_ROAD', 'EDGE(1, 6)_P0_ROAD', 'EDGE(1, 6)_P1_ROAD', 'EDGE(10, 11)_P0_ROAD', 'EDGE(10, 11)_P1_ROAD', 'EDGE(10, 29)_P0_ROAD', 'EDGE(10, 29)_P1_ROAD', 'EDGE(11, 12)_P0_ROAD', 'EDGE(11, 12)_P1_ROAD', 'EDGE(11, 32)_P0_ROAD', 'EDGE(11, 32)_P1_ROAD', 'EDGE(12, 13)_P0_ROAD', 'EDGE(12, 13)_P1_ROAD', 'EDGE(13, 14)_P0_ROAD', 'EDGE(13, 14)_P1_ROAD', 'EDGE(13, 34)_P0_ROAD', 'EDGE(13, 34)_P1_ROAD', 'EDGE(14, 15)_P0_ROAD', 'EDGE(14, 15)_P1_ROAD', 'EDGE(14, 37)_P0_ROAD', 'EDGE(14, 37)_P1_ROAD', 'EDGE(15, 17)_P0_ROAD', 'EDGE(15, 17)_P1_ROAD', 'EDGE(16, 18)_P0_ROAD', 'EDGE(16, 18)_P1_ROAD', 'EDGE(16, 21)_P0_ROAD', 'EDGE(16, 21)_P1_ROAD', 'EDGE(17, 18)_P0_ROAD', 'EDGE(17, 18)_P1_ROAD', 'EDGE(17, 39)_P0_ROAD', 'EDGE(17, 39)_P1_ROAD', 'EDGE(18, 40)_P0_ROAD', 'EDGE(18, 40)_P1_ROAD', 'EDGE(19, 20)_P0_ROAD', 'EDGE(19, 20)_P1_ROAD', 'EDGE(19, 21)_P0_ROAD', 'EDGE(19, 21)_P1_ROAD', 'EDGE(19, 46)_P0_ROAD', 'EDGE(19, 46)_P1_ROAD', 'EDGE(2, 3)_P0_ROAD', 'EDGE(2, 3)_P1_ROAD', 'EDGE(2, 9)_P0_ROAD', 'EDGE(2, 9)_P1_ROAD', 'EDGE(20, 22)_P0_ROAD', 'EDGE(20, 22)_P1_ROAD', 'EDGE(21, 43)_P0_ROAD', 'EDGE(21, 43)_P1_ROAD', 'EDGE(22, 23)_P0_ROAD', 'EDGE(22, 23)_P1_ROAD', 'EDGE(22, 49)_P0_ROAD', 'EDGE(22, 49)_P1_ROAD', 'EDGE(23, 52)_P0_ROAD', 'EDGE(23, 52)_P1_ROAD', 'EDGE(24, 25)_P0_ROAD', 'EDGE(24, 25)_P1_ROAD', 'EDGE(24, 53)_P0_ROAD', 'EDGE(24, 53)_P1_ROAD', 'EDGE(25, 26)_P0_ROAD', 'EDGE(25, 26)_P1_ROAD', 'EDGE(26, 27)_P0_ROAD', 'EDGE(26, 27)_P1_ROAD', 'EDGE(27, 28)_P0_ROAD', 'EDGE(27, 28)_P1_ROAD', 'EDGE(28, 29)_P0_ROAD', 'EDGE(28, 29)_P1_ROAD', 'EDGE(29, 30)_P0_ROAD', 'EDGE(29, 30)_P1_ROAD', 'EDGE(3, 12)_P0_ROAD', 'EDGE(3, 12)_P1_ROAD', 'EDGE(3, 4)_P0_ROAD', 'EDGE(3, 4)_P1_ROAD', 'EDGE(30, 31)_P0_ROAD', 'EDGE(30, 31)_P1_ROAD', 'EDGE(31, 32)_P0_ROAD', 'EDGE(31, 32)_P1_ROAD', 'EDGE(32, 33)_P0_ROAD', 'EDGE(32, 33)_P1_ROAD', 'EDGE(33, 34)_P0_ROAD', 'EDGE(33, 34)_P1_ROAD', 'EDGE(34, 35)_P0_ROAD', 'EDGE(34, 35)_P1_ROAD', 'EDGE(35, 36)_P0_ROAD', 'EDGE(35, 36)_P1_ROAD', 'EDGE(36, 37)_P0_ROAD', 'EDGE(36, 37)_P1_ROAD', 'EDGE(37, 38)_P0_ROAD', 'EDGE(37, 38)_P1_ROAD', 'EDGE(38, 39)_P0_ROAD', 'EDGE(38, 39)_P1_ROAD', 'EDGE(39, 41)_P0_ROAD', 'EDGE(39, 41)_P1_ROAD', 'EDGE(4, 15)_P0_ROAD', 'EDGE(4, 15)_P1_ROAD', 'EDGE(4, 5)_P0_ROAD', 'EDGE(4, 5)_P1_ROAD', 'EDGE(40, 42)_P0_ROAD', 'EDGE(40, 42)_P1_ROAD', 'EDGE(40, 44)_P0_ROAD', 'EDGE(40, 44)_P1_ROAD', 'EDGE(41, 42)_P0_ROAD', 'EDGE(41, 42)_P1_ROAD', 'EDGE(43, 44)_P0_ROAD', 'EDGE(43, 44)_P1_ROAD', 'EDGE(43, 47)_P0_ROAD', 'EDGE(43, 47)_P1_ROAD', 'EDGE(45, 46)_P0_ROAD', 'EDGE(45, 46)_P1_ROAD', 'EDGE(45, 47)_P0_ROAD', 'EDGE(45, 47)_P1_ROAD', 'EDGE(46, 48)_P0_ROAD', 'EDGE(46, 48)_P1_ROAD', 'EDGE(48, 49)_P0_ROAD', 'EDGE(48, 49)_P1_ROAD', 'EDGE(49, 50)_P0_ROAD', 'EDGE(49, 50)_P1_ROAD', 'EDGE(5, 16)_P0_ROAD', 'EDGE(5, 16)_P1_ROAD', 'EDGE(50, 51)_P0_ROAD', 'EDGE(50, 51)_P1_ROAD', 'EDGE(51, 52)_P0_ROAD', 'EDGE(51, 52)_P1_ROAD', 'EDGE(52, 53)_P0_ROAD', 'EDGE(52, 53)_P1_ROAD', 'EDGE(6, 23)_P0_ROAD', 'EDGE(6, 23)_P1_ROAD', 'EDGE(6, 7)_P0_ROAD', 'EDGE(6, 7)_P1_ROAD', 'EDGE(7, 24)_P0_ROAD', 'EDGE(7, 24)_P1_ROAD', 'EDGE(7, 8)_P0_ROAD', 'EDGE(7, 8)_P1_ROAD', 'EDGE(8, 27)_P0_ROAD', 'EDGE(8, 27)_P1_ROAD', 'EDGE(8, 9)_P0_ROAD', 'EDGE(8, 9)_P1_ROAD', 'EDGE(9, 10)_P0_ROAD', 'EDGE(9, 10)_P1_ROAD'],
    #     'c':['TILE0_HAS_ROBBER', 'TILE0_IS_BRICK', 'TILE0_IS_DESERT', 'TILE0_IS_ORE', 'TILE0_IS_SHEEP', 'TILE0_IS_WHEAT', 'TILE0_IS_WOOD', 'TILE0_PROBA', 'TILE10_HAS_ROBBER', 'TILE10_IS_BRICK', 'TILE10_IS_DESERT', 'TILE10_IS_ORE', 'TILE10_IS_SHEEP', 'TILE10_IS_WHEAT', 'TILE10_IS_WOOD', 'TILE10_PROBA', 'TILE11_HAS_ROBBER', 'TILE11_IS_BRICK', 'TILE11_IS_DESERT', 'TILE11_IS_ORE', 'TILE11_IS_SHEEP', 'TILE11_IS_WHEAT', 'TILE11_IS_WOOD', 'TILE11_PROBA', 'TILE12_HAS_ROBBER', 'TILE12_IS_BRICK', 'TILE12_IS_DESERT', 'TILE12_IS_ORE', 'TILE12_IS_SHEEP', 'TILE12_IS_WHEAT', 'TILE12_IS_WOOD', 'TILE12_PROBA', 'TILE13_HAS_ROBBER', 'TILE13_IS_BRICK', 'TILE13_IS_DESERT', 'TILE13_IS_ORE', 'TILE13_IS_SHEEP', 'TILE13_IS_WHEAT', 'TILE13_IS_WOOD', 'TILE13_PROBA', 'TILE14_HAS_ROBBER', 'TILE14_IS_BRICK', 'TILE14_IS_DESERT', 'TILE14_IS_ORE', 'TILE14_IS_SHEEP', 'TILE14_IS_WHEAT', 'TILE14_IS_WOOD', 'TILE14_PROBA', 'TILE15_HAS_ROBBER', 'TILE15_IS_BRICK', 'TILE15_IS_DESERT', 'TILE15_IS_ORE', 'TILE15_IS_SHEEP', 'TILE15_IS_WHEAT', 'TILE15_IS_WOOD', 'TILE15_PROBA', 'TILE16_HAS_ROBBER', 'TILE16_IS_BRICK', 'TILE16_IS_DESERT', 'TILE16_IS_ORE', 'TILE16_IS_SHEEP', 'TILE16_IS_WHEAT', 'TILE16_IS_WOOD', 'TILE16_PROBA', 'TILE17_HAS_ROBBER', 'TILE17_IS_BRICK', 'TILE17_IS_DESERT', 'TILE17_IS_ORE', 'TILE17_IS_SHEEP', 'TILE17_IS_WHEAT', 'TILE17_IS_WOOD', 'TILE17_PROBA', 'TILE18_HAS_ROBBER', 'TILE18_IS_BRICK', 'TILE18_IS_DESERT', 'TILE18_IS_ORE', 'TILE18_IS_SHEEP', 'TILE18_IS_WHEAT', 'TILE18_IS_WOOD', 'TILE18_PROBA', 'TILE1_HAS_ROBBER', 'TILE1_IS_BRICK', 'TILE1_IS_DESERT', 'TILE1_IS_ORE', 'TILE1_IS_SHEEP', 'TILE1_IS_WHEAT', 'TILE1_IS_WOOD', 'TILE1_PROBA', 'TILE2_HAS_ROBBER', 'TILE2_IS_BRICK', 'TILE2_IS_DESERT', 'TILE2_IS_ORE', 'TILE2_IS_SHEEP', 'TILE2_IS_WHEAT', 'TILE2_IS_WOOD', 'TILE2_PROBA', 'TILE3_HAS_ROBBER', 'TILE3_IS_BRICK', 'TILE3_IS_DESERT', 'TILE3_IS_ORE', 'TILE3_IS_SHEEP', 'TILE3_IS_WHEAT', 'TILE3_IS_WOOD', 'TILE3_PROBA', 'TILE4_HAS_ROBBER', 'TILE4_IS_BRICK', 'TILE4_IS_DESERT', 'TILE4_IS_ORE', 'TILE4_IS_SHEEP', 'TILE4_IS_WHEAT', 'TILE4_IS_WOOD', 'TILE4_PROBA', 'TILE5_HAS_ROBBER', 'TILE5_IS_BRICK', 'TILE5_IS_DESERT', 'TILE5_IS_ORE', 'TILE5_IS_SHEEP', 'TILE5_IS_WHEAT', 'TILE5_IS_WOOD', 'TILE5_PROBA', 'TILE6_HAS_ROBBER', 'TILE6_IS_BRICK', 'TILE6_IS_DESERT', 'TILE6_IS_ORE', 'TILE6_IS_SHEEP', 'TILE6_IS_WHEAT', 'TILE6_IS_WOOD', 'TILE6_PROBA', 'TILE7_HAS_ROBBER', 'TILE7_IS_BRICK', 'TILE7_IS_DESERT', 'TILE7_IS_ORE', 'TILE7_IS_SHEEP', 'TILE7_IS_WHEAT', 'TILE7_IS_WOOD', 'TILE7_PROBA', 'TILE8_HAS_ROBBER', 'TILE8_IS_BRICK', 'TILE8_IS_DESERT', 'TILE8_IS_ORE', 'TILE8_IS_SHEEP', 'TILE8_IS_WHEAT', 'TILE8_IS_WOOD', 'TILE8_PROBA', 'TILE9_HAS_ROBBER', 'TILE9_IS_BRICK', 'TILE9_IS_DESERT', 'TILE9_IS_ORE', 'TILE9_IS_SHEEP', 'TILE9_IS_WHEAT', 'TILE9_IS_WOOD', 'TILE9_PROBA'],
    #     'd':['NODE0_P0_CITY', 'NODE0_P0_SETTLEMENT', 'NODE0_P1_CITY', 'NODE0_P1_SETTLEMENT', 'NODE10_P0_CITY', 'NODE10_P0_SETTLEMENT', 'NODE10_P1_CITY', 'NODE10_P1_SETTLEMENT', 'NODE11_P0_CITY', 'NODE11_P0_SETTLEMENT', 'NODE11_P1_CITY', 'NODE11_P1_SETTLEMENT', 'NODE12_P0_CITY', 'NODE12_P0_SETTLEMENT', 'NODE12_P1_CITY', 'NODE12_P1_SETTLEMENT', 'NODE13_P0_CITY', 'NODE13_P0_SETTLEMENT', 'NODE13_P1_CITY', 'NODE13_P1_SETTLEMENT', 'NODE14_P0_CITY', 'NODE14_P0_SETTLEMENT', 'NODE14_P1_CITY', 'NODE14_P1_SETTLEMENT', 'NODE15_P0_CITY', 'NODE15_P0_SETTLEMENT', 'NODE15_P1_CITY', 'NODE15_P1_SETTLEMENT', 'NODE16_P0_CITY', 'NODE16_P0_SETTLEMENT', 'NODE16_P1_CITY', 'NODE16_P1_SETTLEMENT', 'NODE17_P0_CITY', 'NODE17_P0_SETTLEMENT', 'NODE17_P1_CITY', 'NODE17_P1_SETTLEMENT', 'NODE18_P0_CITY', 'NODE18_P0_SETTLEMENT', 'NODE18_P1_CITY', 'NODE18_P1_SETTLEMENT', 'NODE19_P0_CITY', 'NODE19_P0_SETTLEMENT', 'NODE19_P1_CITY', 'NODE19_P1_SETTLEMENT', 'NODE1_P0_CITY', 'NODE1_P0_SETTLEMENT', 'NODE1_P1_CITY', 'NODE1_P1_SETTLEMENT', 'NODE20_P0_CITY', 'NODE20_P0_SETTLEMENT', 'NODE20_P1_CITY', 'NODE20_P1_SETTLEMENT', 'NODE21_P0_CITY', 'NODE21_P0_SETTLEMENT', 'NODE21_P1_CITY', 'NODE21_P1_SETTLEMENT', 'NODE22_P0_CITY', 'NODE22_P0_SETTLEMENT', 'NODE22_P1_CITY', 'NODE22_P1_SETTLEMENT', 'NODE23_P0_CITY', 'NODE23_P0_SETTLEMENT', 'NODE23_P1_CITY', 'NODE23_P1_SETTLEMENT', 'NODE24_P0_CITY', 'NODE24_P0_SETTLEMENT', 'NODE24_P1_CITY', 'NODE24_P1_SETTLEMENT', 'NODE25_P0_CITY', 'NODE25_P0_SETTLEMENT', 'NODE25_P1_CITY', 'NODE25_P1_SETTLEMENT', 'NODE26_P0_CITY', 'NODE26_P0_SETTLEMENT', 'NODE26_P1_CITY', 'NODE26_P1_SETTLEMENT', 'NODE27_P0_CITY', 'NODE27_P0_SETTLEMENT', 'NODE27_P1_CITY', 'NODE27_P1_SETTLEMENT', 'NODE28_P0_CITY', 'NODE28_P0_SETTLEMENT', 'NODE28_P1_CITY', 'NODE28_P1_SETTLEMENT', 'NODE29_P0_CITY', 'NODE29_P0_SETTLEMENT', 'NODE29_P1_CITY', 'NODE29_P1_SETTLEMENT', 'NODE2_P0_CITY', 'NODE2_P0_SETTLEMENT', 'NODE2_P1_CITY', 'NODE2_P1_SETTLEMENT', 'NODE30_P0_CITY', 'NODE30_P0_SETTLEMENT', 'NODE30_P1_CITY', 'NODE30_P1_SETTLEMENT', 'NODE31_P0_CITY', 'NODE31_P0_SETTLEMENT', 'NODE31_P1_CITY', 'NODE31_P1_SETTLEMENT', 'NODE32_P0_CITY', 'NODE32_P0_SETTLEMENT', 'NODE32_P1_CITY', 'NODE32_P1_SETTLEMENT', 'NODE33_P0_CITY', 'NODE33_P0_SETTLEMENT', 'NODE33_P1_CITY', 'NODE33_P1_SETTLEMENT', 'NODE34_P0_CITY', 'NODE34_P0_SETTLEMENT', 'NODE34_P1_CITY', 'NODE34_P1_SETTLEMENT', 'NODE35_P0_CITY', 'NODE35_P0_SETTLEMENT', 'NODE35_P1_CITY', 'NODE35_P1_SETTLEMENT', 'NODE36_P0_CITY', 'NODE36_P0_SETTLEMENT', 'NODE36_P1_CITY', 'NODE36_P1_SETTLEMENT', 'NODE37_P0_CITY', 'NODE37_P0_SETTLEMENT', 'NODE37_P1_CITY', 'NODE37_P1_SETTLEMENT', 'NODE38_P0_CITY', 'NODE38_P0_SETTLEMENT', 'NODE38_P1_CITY', 'NODE38_P1_SETTLEMENT', 'NODE39_P0_CITY', 'NODE39_P0_SETTLEMENT', 'NODE39_P1_CITY', 'NODE39_P1_SETTLEMENT', 'NODE3_P0_CITY', 'NODE3_P0_SETTLEMENT', 'NODE3_P1_CITY', 'NODE3_P1_SETTLEMENT', 'NODE40_P0_CITY', 'NODE40_P0_SETTLEMENT', 'NODE40_P1_CITY', 'NODE40_P1_SETTLEMENT', 'NODE41_P0_CITY', 'NODE41_P0_SETTLEMENT', 'NODE41_P1_CITY', 'NODE41_P1_SETTLEMENT', 'NODE42_P0_CITY', 'NODE42_P0_SETTLEMENT', 'NODE42_P1_CITY', 'NODE42_P1_SETTLEMENT', 'NODE43_P0_CITY', 'NODE43_P0_SETTLEMENT', 'NODE43_P1_CITY', 'NODE43_P1_SETTLEMENT', 'NODE44_P0_CITY', 'NODE44_P0_SETTLEMENT', 'NODE44_P1_CITY', 'NODE44_P1_SETTLEMENT', 'NODE45_P0_CITY', 'NODE45_P0_SETTLEMENT', 'NODE45_P1_CITY', 'NODE45_P1_SETTLEMENT', 'NODE46_P0_CITY', 'NODE46_P0_SETTLEMENT', 'NODE46_P1_CITY', 'NODE46_P1_SETTLEMENT', 'NODE47_P0_CITY', 'NODE47_P0_SETTLEMENT', 'NODE47_P1_CITY', 'NODE47_P1_SETTLEMENT', 'NODE48_P0_CITY', 'NODE48_P0_SETTLEMENT', 'NODE48_P1_CITY', 'NODE48_P1_SETTLEMENT', 'NODE49_P0_CITY', 'NODE49_P0_SETTLEMENT', 'NODE49_P1_CITY', 'NODE49_P1_SETTLEMENT', 'NODE4_P0_CITY', 'NODE4_P0_SETTLEMENT', 'NODE4_P1_CITY', 'NODE4_P1_SETTLEMENT', 'NODE50_P0_CITY', 'NODE50_P0_SETTLEMENT', 'NODE50_P1_CITY', 'NODE50_P1_SETTLEMENT', 'NODE51_P0_CITY', 'NODE51_P0_SETTLEMENT', 'NODE51_P1_CITY', 'NODE51_P1_SETTLEMENT', 'NODE52_P0_CITY', 'NODE52_P0_SETTLEMENT', 'NODE52_P1_CITY', 'NODE52_P1_SETTLEMENT', 'NODE53_P0_CITY', 'NODE53_P0_SETTLEMENT', 'NODE53_P1_CITY', 'NODE53_P1_SETTLEMENT', 'NODE5_P0_CITY', 'NODE5_P0_SETTLEMENT', 'NODE5_P1_CITY', 'NODE5_P1_SETTLEMENT', 'NODE6_P0_CITY', 'NODE6_P0_SETTLEMENT', 'NODE6_P1_CITY', 'NODE6_P1_SETTLEMENT', 'NODE7_P0_CITY', 'NODE7_P0_SETTLEMENT', 'NODE7_P1_CITY', 'NODE7_P1_SETTLEMENT', 'NODE8_P0_CITY', 'NODE8_P0_SETTLEMENT', 'NODE8_P1_CITY', 'NODE8_P1_SETTLEMENT', 'NODE9_P0_CITY', 'NODE9_P0_SETTLEMENT', 'NODE9_P1_CITY', 'NODE9_P1_SETTLEMENT'],
    #     'e':['PORT0_IS_BRICK', 'PORT0_IS_ORE', 'PORT0_IS_SHEEP', 'PORT0_IS_THREE_TO_ONE', 'PORT0_IS_WHEAT', 'PORT0_IS_WOOD', 'PORT1_IS_BRICK', 'PORT1_IS_ORE', 'PORT1_IS_SHEEP', 'PORT1_IS_THREE_TO_ONE', 'PORT1_IS_WHEAT', 'PORT1_IS_WOOD', 'PORT2_IS_BRICK', 'PORT2_IS_ORE', 'PORT2_IS_SHEEP', 'PORT2_IS_THREE_TO_ONE', 'PORT2_IS_WHEAT', 'PORT2_IS_WOOD', 'PORT3_IS_BRICK', 'PORT3_IS_ORE', 'PORT3_IS_SHEEP', 'PORT3_IS_THREE_TO_ONE', 'PORT3_IS_WHEAT', 'PORT3_IS_WOOD', 'PORT4_IS_BRICK', 'PORT4_IS_ORE', 'PORT4_IS_SHEEP', 'PORT4_IS_THREE_TO_ONE', 'PORT4_IS_WHEAT', 'PORT4_IS_WOOD', 'PORT5_IS_BRICK', 'PORT5_IS_ORE', 'PORT5_IS_SHEEP', 'PORT5_IS_THREE_TO_ONE', 'PORT5_IS_WHEAT', 'PORT5_IS_WOOD', 'PORT6_IS_BRICK', 'PORT6_IS_ORE', 'PORT6_IS_SHEEP', 'PORT6_IS_THREE_TO_ONE', 'PORT6_IS_WHEAT', 'PORT6_IS_WOOD', 'PORT7_IS_BRICK', 'PORT7_IS_ORE', 'PORT7_IS_SHEEP', 'PORT7_IS_THREE_TO_ONE', 'PORT7_IS_WHEAT', 'PORT7_IS_WOOD', 'PORT8_IS_BRICK', 'PORT8_IS_ORE', 'PORT8_IS_SHEEP', 'PORT8_IS_THREE_TO_ONE', 'PORT8_IS_WHEAT', 'PORT8_IS_WOOD'],
    #     'f':['IS_DISCARDING', 'IS_MOVING_ROBBER'],
    #     'g':['P0_HAS_ROLLED', 'P1_HAS_ROLLED'],
    #     'h':['P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'],
    #     'i':['P0_ACTUAL_VPS', 'P0_PUBLIC_VPS'],
    #     'j':['P0_BRICK_IN_HAND', 'P0_KNIGHT_IN_HAND', 'P0_MONOPOLY_IN_HAND', 'P0_NUM_DEVS_IN_HAND', 'P0_NUM_RESOURCES_IN_HAND', 'P0_ORE_IN_HAND', 'P0_ROAD_BUILDING_IN_HAND', 'P0_SHEEP_IN_HAND', 'P0_VICTORY_POINT_IN_HAND', 'P0_WHEAT_IN_HAND', 'P0_WOOD_IN_HAND', 'P0_YEAR_OF_PLENTY_IN_HAND'],
    #     'k':['P1_NUM_DEVS_IN_HAND', 'P1_NUM_RESOURCES_IN_HAND'],
    #     'l':['P0_HAS_ARMY'],
    #     'm':['P1_HAS_ARMY'],
    #     'n':['P0_HAS_ROAD'],
    #     'o':['P1_HAS_ROAD'],
    #     'p':['P0_CITIES_LEFT', 'P0_ROADS_LEFT', 'P0_SETTLEMENTS_LEFT'],
    #     'q':['P1_CITIES_LEFT', 'P1_ROADS_LEFT', 'P1_SETTLEMENTS_LEFT'],
    #     'r':['P0_KNIGHT_PLAYED', 'P0_MONOPOLY_PLAYED', 'P0_ROAD_BUILDING_PLAYED', 'P0_YEAR_OF_PLENTY_PLAYED'],
    #     's':['P1_KNIGHT_PLAYED', 'P1_MONOPOLY_PLAYED', 'P1_ROAD_BUILDING_PLAYED', 'P1_YEAR_OF_PLENTY_PLAYED'],
    #     't':['P1_PUBLIC_VPS'],
    #     'u':['P0_LONGEST_ROAD_LENGTH'],
    #     'v':['P1_LONGEST_ROAD_LENGTH'],
    # }
    bank_features = re.compile(".*BANK")
    edge_features = re.compile(".*EDGE")
    P0_roads = re.compile(r'EDGE\([0-9]+,\s+[0-9]+\)_P0_ROAD')
    P1_roads = re.compile(r'EDGE\([0-9]+,\s+[0-9]+\)_P1_ROAD')
    tile_features = re.compile(".*TILE")
    node_features = re.compile(".*NODE")
    P0_settlements = re.compile(r'NODE[A-Za-z0-9_]+_P0_SETTLEMENT')
    P1_settlements = re.compile(r'NODE[A-Za-z0-9_]+_P1_SETTLEMENT')
    P0_cities = re.compile(r'NODE[A-Za-z0-9_]+_P0_CITY')
    P1_cities = re.compile(r'NODE[A-Za-z0-9_]+_P1_CITY')
    port_features = re.compile(".*PORT")
    rolled_robber_features = re.compile("(.*IS_DISCARDING)|(.*IS_MOVING_ROBBER)")
    rolled_feature = re.compile(".*HAS_ROLLED")
    p0_played_dev_in_turn = re.compile(r'P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN')
    p0_vps = re.compile(".*P0_ACTUAL_VPS|.*P0_PUBLIC_VPS")
    p0_hand = re.compile(r'P0_[A-Za-z0-9_]+_IN_HAND')
    p1_hand = re.compile(r'P1_[A-Za-z0-9_]+_IN_HAND')
    p0_has_army = re.compile(r'P0_HAS_ARMY')
    p0_has_road = re.compile(r'P0_HAS_ROAD')
    p1_has_army = re.compile(r'P1_HAS_ARMY')
    p1_has_road = re.compile(r'P1_HAS_ROAD')
    p0_pieces= re.compile(r'P0_[A-Za-z0-9_]+_LEFT')
    p1_pieces= re.compile(r'P1_[A-Za-z0-9_]+_LEFT')
    p0_devs_played = re.compile(r'P0_[A-Za-z0-9_]+_PLAYED$')
    p1_devs_played = re.compile(r'P1_[A-Za-z0-9_]+_PLAYED$')
    p1_public_vps = re.compile(r'P1_PUBLIC_VPS')
    p0_longest_road_length = re.compile(r'P0_LONGEST_ROAD_LENGTH')
    p1_longest_road_length = re.compile(r'P1_LONGEST_ROAD_LENGTH')


    bank_features =list(filter(bank_features.match, features))
    edge_features =list(filter(edge_features.match, features))
    p0_road_features = list(filter(P0_roads.match, features))
    p1_road_features = list(filter(P1_roads.match, features))
    tile_features =list(filter(tile_features.match, features))
    node_features=list(filter(node_features.match, features))
    P0_settlement_features= list(filter(P0_settlements.match, features))
    P1_settlement_features= list(filter(P1_settlements.match, features))
    P0_city_features= list(filter(P0_cities.match, features))
    P1_city_features= list(filter(P1_cities.match, features))
    port_features=list(filter(port_features.match, features))
    rolled_robber_features=list(filter(rolled_robber_features.match, features))
    rolled_feature=list(filter(rolled_feature.match, features))
    p0_played_dev_in_turn =list(filter(p0_played_dev_in_turn.match, features))
    p0_vps = list(filter(p0_vps.match, features))
    p0_hand = list(filter(p0_hand.match, features))
    p1_hand =list(filter(p1_hand.match, features))
    p0_has_army =list(filter(p0_has_army.match, features))
    p1_has_army=list(filter(p1_has_army.match, features))
    p0_has_road=list(filter(p0_has_road.match, features))
    p1_has_road=list(filter(p1_has_road.match, features))
    p0_pieces =list(filter(p0_pieces.match, features))
    p1_pieces=list(filter(p1_pieces.match, features))
    p0_devs_played =list(filter(p0_devs_played.match, features))
    p1_devs_played =list(filter(p1_devs_played.match, features))
    p1_public_vps=list(filter(p1_public_vps.match, features))
    p0_longest_road_length=list(filter(p0_longest_road_length.match, features))
    p1_longest_road_length=list(filter(p1_longest_road_length.match, features))


    # as a group
    if group == "group_1":
        group_1 = {
            "bank": bank_features,
            "edge_features": edge_features,
            "tile_features": tile_features,
            "node_features": node_features,
            "port_features":port_features,
            "rolled_robber_features": rolled_robber_features, # ['IS_DISCARDING', 'IS_MOVING_ROBBER'],
            "rolled_feature": rolled_feature, # ['P0_HAS_ROLLED', 'P1_HAS_ROLLED'],
            "p0_played_dev_in_turn": p0_played_dev_in_turn, # ['P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'],
            "p0_vps": p0_vps, # ['P0_ACTUAL_VPS', 'P0_PUBLIC_VPS'],
            "p1_public_vps": p1_public_vps, # ['P0_PUBLIC_VPS'],
            "p0_hand": p0_hand, # ['P0_BRICK_IN_HAND', 'P0_KNIGHT_IN_HAND', 'P0_MONOPOLY_IN_HAND', 'P0_NUM_DEVS_IN_HAND', 'P0_NUM_RESOURCES_IN_HAND', 'P0_ORE_IN_HAND', 'P0_ROAD_BUILDING_IN_HAND', 'P0_SHEEP_IN_HAND', 'P0_VICTORY_POINT_IN_HAND', 'P0_WHEAT_IN_HAND', 'P0_WOOD_IN_HAND', 'P0_YEAR_OF_PLENTY_IN_HAND'],
            "p1_hand": p1_hand, # ['P1_NUM_DEVS_IN_HAND', 'P1_NUM_RESOURCES_IN_HAND'],
            "p0_has_army": p0_has_army, #['P0_HAS_ARMY'],
            "p0_has_longest_road": p0_has_road,# ['P0_HAS_ROAD'],
            "p0_longest_road_length": p0_longest_road_length,# ['P0_LONGEST_ROAD_LENGTH'],
            "p1_has_army":p1_has_army, # ['P1_HAS_ARMY'],
            "p1_has_longest_road": p1_has_road,# ['P1_HAS_ROAD'],
            "p1_longest_road_length":p1_longest_road_length,# ['P1_LONGEST_ROAD_LENGTH'],
            "p0_pieces": p0_pieces, # ['P0_CITIES_LEFT', 'P0_ROADS_LEFT', 'P0_SETTLEMENTS_LEFT'],
            "p1_pieces":p1_pieces, #['P1_CITIES_LEFT', 'P1_ROADS_LEFT', 'P1_SETTLEMENTS_LEFT'],
            "p0_devs_played":p0_devs_played, # ['P0_KNIGHT_PLAYED', 'P0_MONOPOLY_PLAYED', 'P0_ROAD_BUILDING_PLAYED', 'P0_YEAR_OF_PLENTY_PLAYED'],
            "p1_devs_played": p1_devs_played, # ['P1_KNIGHT_PLAYED', 'P1_MONOPOLY_PLAYED', 'P1_ROAD_BUILDING_PLAYED', 'P1_YEAR_OF_PLENTY_PLAYED'],
        }
        return group_1
    elif group == "group_2":
        group_2 ={
            "board_features": bank_features + tile_features + port_features, 
            "graph_features": node_features + edge_features,
            "turn_features": rolled_feature + rolled_robber_features+p0_played_dev_in_turn,
            "p0_vps": p0_vps + p0_has_army + p0_has_road, #TODO Get the VP_in hand
            "p1_vps": p1_public_vps + p1_has_army + p1_has_road,
            "p0_hand_and_pieces": p0_hand + p0_pieces + p0_longest_road_length,
            "p1_hand_and_pieces": p1_hand + p1_pieces +p1_longest_road_length,
            "p0_dev_cards_played": p0_devs_played,
            "p1_dev_cards_played": p1_devs_played,
        }
        return group_2
    elif group == "group_3":
        group_3 ={
            'board_features': bank_features + tile_features + port_features, 
            'graph_features': node_features + edge_features,
            'turn_features': rolled_feature + rolled_robber_features+p0_played_dev_in_turn,
            'p0_vps': p0_vps + p0_has_army + p0_has_road+ p0_hand + p0_pieces + p0_longest_road_length + p0_devs_played,
            'p1_vps': p1_public_vps + p1_has_army + p1_has_road + p1_hand + p1_pieces +p1_longest_road_length + p1_devs_played
        }
        return group_3
    elif group == "small_strat":
        small_strat_group ={
            "bank": bank_features,
            "p0_roads": p0_road_features + p0_longest_road_length + p0_has_road, 
            "p1_roads": p1_road_features + p1_longest_road_length + p1_has_road,
            "tile_features": tile_features + port_features,
            "p0_settlements": P0_settlement_features,
            "p1_settlements": P1_settlement_features,
            "p0_cities": P0_city_features,
            "p1_cities": P1_city_features,
            "rolled_robber_features": rolled_robber_features,
            "rolled_feature": rolled_feature,
            "p0_vps": p0_vps+p0_has_army,
            "p1_public_vps": p1_public_vps + p1_has_army,
            "p0_pieces_and_hand": p0_pieces+p0_hand,
            "p1_pieces_and_hand":p1_pieces+p1_hand,
            "p0_devs_played":p0_devs_played+p0_played_dev_in_turn,
            "p1_devs_played": p1_devs_played,
        }
        return small_strat_group
    else:
        return None

def prep_data(data:pd.DataFrame,features):
    data = data.copy()
    df = data[['obs','action']]
    print(data[data.ep == 3])
    data_list =df['obs'].apply(lambda x: x.strip("[]").split(", ")).tolist()
    # data_list =df['obs'].apply(lambda x: ast.literal_eval(x))
    # print("data list",data_list)
    # print(len(data_list[0]))
    dfx = pd.DataFrame(data_list,columns=features ,index = data.index)
    dfx[features] = dfx[features].astype("float")
    # dfx = 
    print(dfx)
    #find len of final ep
    #TODO find length of winning episode to use as the test
    # len(df[df.ep ==df.ep.max()])
    # old split
    dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(dfx[features],df['action'],shuffle=False, test_size=len(data[data.ep == 8])) #ep=2 is game 6
    # suffle the training data
    
    # new_split to include test in training
    dfx_train = dfx[features]
    dfy_train = df['action']
    print("dfx_train\n",dfx_train)
    test_indexes =data[data.ep == 3].index
    dfx_test = dfx.loc[test_indexes]
    dfy_test = df.loc[test_indexes]
    dfy_test = dfy_test['action']
    print("dfx_test\n",dfx_test)
    print("dfy_test\n",dfy_test)
    dfx_train, dfy_train = shuffle(dfx_train,dfy_train, random_state=42)
    #TODO: add dfx_test
    return dfx_train, dfx_test, dfy_train, dfy_test

@torch.no_grad()
def shapley(env_config= None, agent_path = "./logs/Catan_Switch_Exp/exp_014/best_model.zip"):
    data, info_list, catan_features = run_switch_agent(env_config=env_config, agent_path=agent_path)
    features = ["ep","step",'obs', 'action']
    df = pd.DataFrame(data=data, columns=features)

    #format data
    # dfx_train, dfx_test, dfy_train, dfy_test = prep_data(df,catan_features)

    #load_model
    
    model = PPO.load(path =agent_path, device='cuda')
    print(type(model))
    # model = sb3Wrapper(model)
    print(type(model))
    print(model)
    # print(model.action_net)
    # print(dfy_train.mean())
    # model.predict()[0]
    # explainer = shap.shap.DeepExplainer(model,dfx_train,framework='pytorch')
    state_log = np.array(df['obs'].values.tolist())
    tensor_data = torch.from_numpy(state_log).to('cuda')
    print(tensor_data)
    print(tensor_data.shape)
    # f = lambda x: model.forward(Variable(torch.from_numpy(x),requires_grad=False).to(float32).cuda()).detach().cpu().numpy()
    
    f = lambda x: model.predict(x,deterministic=True)[0]
    print(f(state_log[0]))
    explainer = shap.shap.KernelExplainer(f, state_log)
    #TODO get shapvals for single episode and look into sampling and saving models
    shap_vals= explainer.shap_values(state_log)
    # state_log = np.array(df['obs'].values.tolist())


    # tensor_data = torch.FloatTensor(state_log).to('cuda')
    # print(tensor_data[0])
    # print(model.forward(tensor_data[0]).dtype)
    # print(df['action'][0])
    # explainer = shap.DeepExplainer(model,tensor_data)
    # shap_values = explainer.shap_values(data,check_additivity=True)

    return shap_vals
def get_games(games_dir = './good_games/'):
    # def combine_csvs_into_dataframe(directory="./good_games/"):
    # Initialize an empty list to store individual DataFrames
    data_frames = []
    
    # Traverse directory, including all subdirectories
    for root, _, files in os.walk(games_dir):
        for file in files:
            # Check if file name matches the naming scheme "episode_{ep}_data.csv"
            if file.startswith("episode_") and file.endswith("_data.csv"):
                try:
                    # Attempt to extract the episode number from the file name
                    ep = int(file.split("_")[1])
                    if 0 <= ep <= 9:
                        # Read the CSV file into a DataFrame
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        # Add a new column to indicate the episode number
                        df["episode"] = ep
                        # Append the DataFrame to the list
                        data_frames.append(df)
                except ValueError:
                    print(f"Skipping file {file} as it doesn't match the episode naming convention.")
                    
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    return combined_df

@torch.no_grad()
def groupshap(env_config= None, approach:str ="independence", grouping =None, agent_path = "./logs/Catan_Switch_Exp/exp_019/latest_model_1000000_steps.zip", nsamples=None,nbatches=None):
    print("using grad", torch.is_grad_enabled())
    data, info_list, catan_features = run_switch_agent(env_config=env_config, agent_path=agent_path)
    features = ["ep","step",'obs', 'action']
    # df = pd.DataFrame(data=data, columns=features)
    
    # print("first df",df)
    group = get_group(catan_features, grouping)
    # print(group)
    df = get_games("./good_games/")
    print("good games",df)
    print(ast.literal_eval(df['obs'][0]))
    #TODO: finish setting up the use of good games
    # return
    #format data
    dfx_train, dfx_test, dfy_train, dfy_test = prep_data(df,catan_features)
    # print(dfx_train.dtypes)
    # print(dfx_train.columns.values.tolist())
    # print(len(catan_features))
    # r_group = NULL if group is None else ListVector({key: StrVector(value) for key, value in group.items()})
    # print(r_group)

    # group = {
    #     'a':['BANK_BRICK', 'BANK_DEV_CARDS', 'BANK_ORE', 'BANK_SHEEP', 'BANK_WHEAT', 'BANK_WOOD'],
    #     'b':['EDGE(0, 1)_P0_ROAD', 'EDGE(0, 1)_P1_ROAD', 'EDGE(0, 20)_P0_ROAD', 'EDGE(0, 20)_P1_ROAD', 'EDGE(0, 5)_P0_ROAD', 'EDGE(0, 5)_P1_ROAD', 'EDGE(1, 2)_P0_ROAD', 'EDGE(1, 2)_P1_ROAD', 'EDGE(1, 6)_P0_ROAD', 'EDGE(1, 6)_P1_ROAD', 'EDGE(10, 11)_P0_ROAD', 'EDGE(10, 11)_P1_ROAD', 'EDGE(10, 29)_P0_ROAD', 'EDGE(10, 29)_P1_ROAD', 'EDGE(11, 12)_P0_ROAD', 'EDGE(11, 12)_P1_ROAD', 'EDGE(11, 32)_P0_ROAD', 'EDGE(11, 32)_P1_ROAD', 'EDGE(12, 13)_P0_ROAD', 'EDGE(12, 13)_P1_ROAD', 'EDGE(13, 14)_P0_ROAD', 'EDGE(13, 14)_P1_ROAD', 'EDGE(13, 34)_P0_ROAD', 'EDGE(13, 34)_P1_ROAD', 'EDGE(14, 15)_P0_ROAD', 'EDGE(14, 15)_P1_ROAD', 'EDGE(14, 37)_P0_ROAD', 'EDGE(14, 37)_P1_ROAD', 'EDGE(15, 17)_P0_ROAD', 'EDGE(15, 17)_P1_ROAD', 'EDGE(16, 18)_P0_ROAD', 'EDGE(16, 18)_P1_ROAD', 'EDGE(16, 21)_P0_ROAD', 'EDGE(16, 21)_P1_ROAD', 'EDGE(17, 18)_P0_ROAD', 'EDGE(17, 18)_P1_ROAD', 'EDGE(17, 39)_P0_ROAD', 'EDGE(17, 39)_P1_ROAD', 'EDGE(18, 40)_P0_ROAD', 'EDGE(18, 40)_P1_ROAD', 'EDGE(19, 20)_P0_ROAD', 'EDGE(19, 20)_P1_ROAD', 'EDGE(19, 21)_P0_ROAD', 'EDGE(19, 21)_P1_ROAD', 'EDGE(19, 46)_P0_ROAD', 'EDGE(19, 46)_P1_ROAD', 'EDGE(2, 3)_P0_ROAD', 'EDGE(2, 3)_P1_ROAD', 'EDGE(2, 9)_P0_ROAD', 'EDGE(2, 9)_P1_ROAD', 'EDGE(20, 22)_P0_ROAD', 'EDGE(20, 22)_P1_ROAD', 'EDGE(21, 43)_P0_ROAD', 'EDGE(21, 43)_P1_ROAD', 'EDGE(22, 23)_P0_ROAD', 'EDGE(22, 23)_P1_ROAD', 'EDGE(22, 49)_P0_ROAD', 'EDGE(22, 49)_P1_ROAD', 'EDGE(23, 52)_P0_ROAD', 'EDGE(23, 52)_P1_ROAD', 'EDGE(24, 25)_P0_ROAD', 'EDGE(24, 25)_P1_ROAD', 'EDGE(24, 53)_P0_ROAD', 'EDGE(24, 53)_P1_ROAD', 'EDGE(25, 26)_P0_ROAD', 'EDGE(25, 26)_P1_ROAD', 'EDGE(26, 27)_P0_ROAD', 'EDGE(26, 27)_P1_ROAD', 'EDGE(27, 28)_P0_ROAD', 'EDGE(27, 28)_P1_ROAD', 'EDGE(28, 29)_P0_ROAD', 'EDGE(28, 29)_P1_ROAD', 'EDGE(29, 30)_P0_ROAD', 'EDGE(29, 30)_P1_ROAD', 'EDGE(3, 12)_P0_ROAD', 'EDGE(3, 12)_P1_ROAD', 'EDGE(3, 4)_P0_ROAD', 'EDGE(3, 4)_P1_ROAD', 'EDGE(30, 31)_P0_ROAD', 'EDGE(30, 31)_P1_ROAD', 'EDGE(31, 32)_P0_ROAD', 'EDGE(31, 32)_P1_ROAD', 'EDGE(32, 33)_P0_ROAD', 'EDGE(32, 33)_P1_ROAD', 'EDGE(33, 34)_P0_ROAD', 'EDGE(33, 34)_P1_ROAD', 'EDGE(34, 35)_P0_ROAD', 'EDGE(34, 35)_P1_ROAD', 'EDGE(35, 36)_P0_ROAD', 'EDGE(35, 36)_P1_ROAD', 'EDGE(36, 37)_P0_ROAD', 'EDGE(36, 37)_P1_ROAD', 'EDGE(37, 38)_P0_ROAD', 'EDGE(37, 38)_P1_ROAD', 'EDGE(38, 39)_P0_ROAD', 'EDGE(38, 39)_P1_ROAD', 'EDGE(39, 41)_P0_ROAD', 'EDGE(39, 41)_P1_ROAD', 'EDGE(4, 15)_P0_ROAD', 'EDGE(4, 15)_P1_ROAD', 'EDGE(4, 5)_P0_ROAD', 'EDGE(4, 5)_P1_ROAD', 'EDGE(40, 42)_P0_ROAD', 'EDGE(40, 42)_P1_ROAD', 'EDGE(40, 44)_P0_ROAD', 'EDGE(40, 44)_P1_ROAD', 'EDGE(41, 42)_P0_ROAD', 'EDGE(41, 42)_P1_ROAD', 'EDGE(43, 44)_P0_ROAD', 'EDGE(43, 44)_P1_ROAD', 'EDGE(43, 47)_P0_ROAD', 'EDGE(43, 47)_P1_ROAD', 'EDGE(45, 46)_P0_ROAD', 'EDGE(45, 46)_P1_ROAD', 'EDGE(45, 47)_P0_ROAD', 'EDGE(45, 47)_P1_ROAD', 'EDGE(46, 48)_P0_ROAD', 'EDGE(46, 48)_P1_ROAD', 'EDGE(48, 49)_P0_ROAD', 'EDGE(48, 49)_P1_ROAD', 'EDGE(49, 50)_P0_ROAD', 'EDGE(49, 50)_P1_ROAD', 'EDGE(5, 16)_P0_ROAD', 'EDGE(5, 16)_P1_ROAD', 'EDGE(50, 51)_P0_ROAD', 'EDGE(50, 51)_P1_ROAD', 'EDGE(51, 52)_P0_ROAD', 'EDGE(51, 52)_P1_ROAD', 'EDGE(52, 53)_P0_ROAD', 'EDGE(52, 53)_P1_ROAD', 'EDGE(6, 23)_P0_ROAD', 'EDGE(6, 23)_P1_ROAD', 'EDGE(6, 7)_P0_ROAD', 'EDGE(6, 7)_P1_ROAD', 'EDGE(7, 24)_P0_ROAD', 'EDGE(7, 24)_P1_ROAD', 'EDGE(7, 8)_P0_ROAD', 'EDGE(7, 8)_P1_ROAD', 'EDGE(8, 27)_P0_ROAD', 'EDGE(8, 27)_P1_ROAD', 'EDGE(8, 9)_P0_ROAD', 'EDGE(8, 9)_P1_ROAD', 'EDGE(9, 10)_P0_ROAD', 'EDGE(9, 10)_P1_ROAD'],
    #     'c':['TILE0_HAS_ROBBER', 'TILE0_IS_BRICK', 'TILE0_IS_DESERT', 'TILE0_IS_ORE', 'TILE0_IS_SHEEP', 'TILE0_IS_WHEAT', 'TILE0_IS_WOOD', 'TILE0_PROBA', 'TILE10_HAS_ROBBER', 'TILE10_IS_BRICK', 'TILE10_IS_DESERT', 'TILE10_IS_ORE', 'TILE10_IS_SHEEP', 'TILE10_IS_WHEAT', 'TILE10_IS_WOOD', 'TILE10_PROBA', 'TILE11_HAS_ROBBER', 'TILE11_IS_BRICK', 'TILE11_IS_DESERT', 'TILE11_IS_ORE', 'TILE11_IS_SHEEP', 'TILE11_IS_WHEAT', 'TILE11_IS_WOOD', 'TILE11_PROBA', 'TILE12_HAS_ROBBER', 'TILE12_IS_BRICK', 'TILE12_IS_DESERT', 'TILE12_IS_ORE', 'TILE12_IS_SHEEP', 'TILE12_IS_WHEAT', 'TILE12_IS_WOOD', 'TILE12_PROBA', 'TILE13_HAS_ROBBER', 'TILE13_IS_BRICK', 'TILE13_IS_DESERT', 'TILE13_IS_ORE', 'TILE13_IS_SHEEP', 'TILE13_IS_WHEAT', 'TILE13_IS_WOOD', 'TILE13_PROBA', 'TILE14_HAS_ROBBER', 'TILE14_IS_BRICK', 'TILE14_IS_DESERT', 'TILE14_IS_ORE', 'TILE14_IS_SHEEP', 'TILE14_IS_WHEAT', 'TILE14_IS_WOOD', 'TILE14_PROBA', 'TILE15_HAS_ROBBER', 'TILE15_IS_BRICK', 'TILE15_IS_DESERT', 'TILE15_IS_ORE', 'TILE15_IS_SHEEP', 'TILE15_IS_WHEAT', 'TILE15_IS_WOOD', 'TILE15_PROBA', 'TILE16_HAS_ROBBER', 'TILE16_IS_BRICK', 'TILE16_IS_DESERT', 'TILE16_IS_ORE', 'TILE16_IS_SHEEP', 'TILE16_IS_WHEAT', 'TILE16_IS_WOOD', 'TILE16_PROBA', 'TILE17_HAS_ROBBER', 'TILE17_IS_BRICK', 'TILE17_IS_DESERT', 'TILE17_IS_ORE', 'TILE17_IS_SHEEP', 'TILE17_IS_WHEAT', 'TILE17_IS_WOOD', 'TILE17_PROBA', 'TILE18_HAS_ROBBER', 'TILE18_IS_BRICK', 'TILE18_IS_DESERT', 'TILE18_IS_ORE', 'TILE18_IS_SHEEP', 'TILE18_IS_WHEAT', 'TILE18_IS_WOOD', 'TILE18_PROBA', 'TILE1_HAS_ROBBER', 'TILE1_IS_BRICK', 'TILE1_IS_DESERT', 'TILE1_IS_ORE', 'TILE1_IS_SHEEP', 'TILE1_IS_WHEAT', 'TILE1_IS_WOOD', 'TILE1_PROBA', 'TILE2_HAS_ROBBER', 'TILE2_IS_BRICK', 'TILE2_IS_DESERT', 'TILE2_IS_ORE', 'TILE2_IS_SHEEP', 'TILE2_IS_WHEAT', 'TILE2_IS_WOOD', 'TILE2_PROBA', 'TILE3_HAS_ROBBER', 'TILE3_IS_BRICK', 'TILE3_IS_DESERT', 'TILE3_IS_ORE', 'TILE3_IS_SHEEP', 'TILE3_IS_WHEAT', 'TILE3_IS_WOOD', 'TILE3_PROBA', 'TILE4_HAS_ROBBER', 'TILE4_IS_BRICK', 'TILE4_IS_DESERT', 'TILE4_IS_ORE', 'TILE4_IS_SHEEP', 'TILE4_IS_WHEAT', 'TILE4_IS_WOOD', 'TILE4_PROBA', 'TILE5_HAS_ROBBER', 'TILE5_IS_BRICK', 'TILE5_IS_DESERT', 'TILE5_IS_ORE', 'TILE5_IS_SHEEP', 'TILE5_IS_WHEAT', 'TILE5_IS_WOOD', 'TILE5_PROBA', 'TILE6_HAS_ROBBER', 'TILE6_IS_BRICK', 'TILE6_IS_DESERT', 'TILE6_IS_ORE', 'TILE6_IS_SHEEP', 'TILE6_IS_WHEAT', 'TILE6_IS_WOOD', 'TILE6_PROBA', 'TILE7_HAS_ROBBER', 'TILE7_IS_BRICK', 'TILE7_IS_DESERT', 'TILE7_IS_ORE', 'TILE7_IS_SHEEP', 'TILE7_IS_WHEAT', 'TILE7_IS_WOOD', 'TILE7_PROBA', 'TILE8_HAS_ROBBER', 'TILE8_IS_BRICK', 'TILE8_IS_DESERT', 'TILE8_IS_ORE', 'TILE8_IS_SHEEP', 'TILE8_IS_WHEAT', 'TILE8_IS_WOOD', 'TILE8_PROBA', 'TILE9_HAS_ROBBER', 'TILE9_IS_BRICK', 'TILE9_IS_DESERT', 'TILE9_IS_ORE', 'TILE9_IS_SHEEP', 'TILE9_IS_WHEAT', 'TILE9_IS_WOOD', 'TILE9_PROBA'],
    #     'd':['NODE0_P0_CITY', 'NODE0_P0_SETTLEMENT', 'NODE0_P1_CITY', 'NODE0_P1_SETTLEMENT', 'NODE10_P0_CITY', 'NODE10_P0_SETTLEMENT', 'NODE10_P1_CITY', 'NODE10_P1_SETTLEMENT', 'NODE11_P0_CITY', 'NODE11_P0_SETTLEMENT', 'NODE11_P1_CITY', 'NODE11_P1_SETTLEMENT', 'NODE12_P0_CITY', 'NODE12_P0_SETTLEMENT', 'NODE12_P1_CITY', 'NODE12_P1_SETTLEMENT', 'NODE13_P0_CITY', 'NODE13_P0_SETTLEMENT', 'NODE13_P1_CITY', 'NODE13_P1_SETTLEMENT', 'NODE14_P0_CITY', 'NODE14_P0_SETTLEMENT', 'NODE14_P1_CITY', 'NODE14_P1_SETTLEMENT', 'NODE15_P0_CITY', 'NODE15_P0_SETTLEMENT', 'NODE15_P1_CITY', 'NODE15_P1_SETTLEMENT', 'NODE16_P0_CITY', 'NODE16_P0_SETTLEMENT', 'NODE16_P1_CITY', 'NODE16_P1_SETTLEMENT', 'NODE17_P0_CITY', 'NODE17_P0_SETTLEMENT', 'NODE17_P1_CITY', 'NODE17_P1_SETTLEMENT', 'NODE18_P0_CITY', 'NODE18_P0_SETTLEMENT', 'NODE18_P1_CITY', 'NODE18_P1_SETTLEMENT', 'NODE19_P0_CITY', 'NODE19_P0_SETTLEMENT', 'NODE19_P1_CITY', 'NODE19_P1_SETTLEMENT', 'NODE1_P0_CITY', 'NODE1_P0_SETTLEMENT', 'NODE1_P1_CITY', 'NODE1_P1_SETTLEMENT', 'NODE20_P0_CITY', 'NODE20_P0_SETTLEMENT', 'NODE20_P1_CITY', 'NODE20_P1_SETTLEMENT', 'NODE21_P0_CITY', 'NODE21_P0_SETTLEMENT', 'NODE21_P1_CITY', 'NODE21_P1_SETTLEMENT', 'NODE22_P0_CITY', 'NODE22_P0_SETTLEMENT', 'NODE22_P1_CITY', 'NODE22_P1_SETTLEMENT', 'NODE23_P0_CITY', 'NODE23_P0_SETTLEMENT', 'NODE23_P1_CITY', 'NODE23_P1_SETTLEMENT', 'NODE24_P0_CITY', 'NODE24_P0_SETTLEMENT', 'NODE24_P1_CITY', 'NODE24_P1_SETTLEMENT', 'NODE25_P0_CITY', 'NODE25_P0_SETTLEMENT', 'NODE25_P1_CITY', 'NODE25_P1_SETTLEMENT', 'NODE26_P0_CITY', 'NODE26_P0_SETTLEMENT', 'NODE26_P1_CITY', 'NODE26_P1_SETTLEMENT', 'NODE27_P0_CITY', 'NODE27_P0_SETTLEMENT', 'NODE27_P1_CITY', 'NODE27_P1_SETTLEMENT', 'NODE28_P0_CITY', 'NODE28_P0_SETTLEMENT', 'NODE28_P1_CITY', 'NODE28_P1_SETTLEMENT', 'NODE29_P0_CITY', 'NODE29_P0_SETTLEMENT', 'NODE29_P1_CITY', 'NODE29_P1_SETTLEMENT', 'NODE2_P0_CITY', 'NODE2_P0_SETTLEMENT', 'NODE2_P1_CITY', 'NODE2_P1_SETTLEMENT', 'NODE30_P0_CITY', 'NODE30_P0_SETTLEMENT', 'NODE30_P1_CITY', 'NODE30_P1_SETTLEMENT', 'NODE31_P0_CITY', 'NODE31_P0_SETTLEMENT', 'NODE31_P1_CITY', 'NODE31_P1_SETTLEMENT', 'NODE32_P0_CITY', 'NODE32_P0_SETTLEMENT', 'NODE32_P1_CITY', 'NODE32_P1_SETTLEMENT', 'NODE33_P0_CITY', 'NODE33_P0_SETTLEMENT', 'NODE33_P1_CITY', 'NODE33_P1_SETTLEMENT', 'NODE34_P0_CITY', 'NODE34_P0_SETTLEMENT', 'NODE34_P1_CITY', 'NODE34_P1_SETTLEMENT', 'NODE35_P0_CITY', 'NODE35_P0_SETTLEMENT', 'NODE35_P1_CITY', 'NODE35_P1_SETTLEMENT', 'NODE36_P0_CITY', 'NODE36_P0_SETTLEMENT', 'NODE36_P1_CITY', 'NODE36_P1_SETTLEMENT', 'NODE37_P0_CITY', 'NODE37_P0_SETTLEMENT', 'NODE37_P1_CITY', 'NODE37_P1_SETTLEMENT', 'NODE38_P0_CITY', 'NODE38_P0_SETTLEMENT', 'NODE38_P1_CITY', 'NODE38_P1_SETTLEMENT', 'NODE39_P0_CITY', 'NODE39_P0_SETTLEMENT', 'NODE39_P1_CITY', 'NODE39_P1_SETTLEMENT', 'NODE3_P0_CITY', 'NODE3_P0_SETTLEMENT', 'NODE3_P1_CITY', 'NODE3_P1_SETTLEMENT', 'NODE40_P0_CITY', 'NODE40_P0_SETTLEMENT', 'NODE40_P1_CITY', 'NODE40_P1_SETTLEMENT', 'NODE41_P0_CITY', 'NODE41_P0_SETTLEMENT', 'NODE41_P1_CITY', 'NODE41_P1_SETTLEMENT', 'NODE42_P0_CITY', 'NODE42_P0_SETTLEMENT', 'NODE42_P1_CITY', 'NODE42_P1_SETTLEMENT', 'NODE43_P0_CITY', 'NODE43_P0_SETTLEMENT', 'NODE43_P1_CITY', 'NODE43_P1_SETTLEMENT', 'NODE44_P0_CITY', 'NODE44_P0_SETTLEMENT', 'NODE44_P1_CITY', 'NODE44_P1_SETTLEMENT', 'NODE45_P0_CITY', 'NODE45_P0_SETTLEMENT', 'NODE45_P1_CITY', 'NODE45_P1_SETTLEMENT', 'NODE46_P0_CITY', 'NODE46_P0_SETTLEMENT', 'NODE46_P1_CITY', 'NODE46_P1_SETTLEMENT', 'NODE47_P0_CITY', 'NODE47_P0_SETTLEMENT', 'NODE47_P1_CITY', 'NODE47_P1_SETTLEMENT', 'NODE48_P0_CITY', 'NODE48_P0_SETTLEMENT', 'NODE48_P1_CITY', 'NODE48_P1_SETTLEMENT', 'NODE49_P0_CITY', 'NODE49_P0_SETTLEMENT', 'NODE49_P1_CITY', 'NODE49_P1_SETTLEMENT', 'NODE4_P0_CITY', 'NODE4_P0_SETTLEMENT', 'NODE4_P1_CITY', 'NODE4_P1_SETTLEMENT', 'NODE50_P0_CITY', 'NODE50_P0_SETTLEMENT', 'NODE50_P1_CITY', 'NODE50_P1_SETTLEMENT', 'NODE51_P0_CITY', 'NODE51_P0_SETTLEMENT', 'NODE51_P1_CITY', 'NODE51_P1_SETTLEMENT', 'NODE52_P0_CITY', 'NODE52_P0_SETTLEMENT', 'NODE52_P1_CITY', 'NODE52_P1_SETTLEMENT', 'NODE53_P0_CITY', 'NODE53_P0_SETTLEMENT', 'NODE53_P1_CITY', 'NODE53_P1_SETTLEMENT', 'NODE5_P0_CITY', 'NODE5_P0_SETTLEMENT', 'NODE5_P1_CITY', 'NODE5_P1_SETTLEMENT', 'NODE6_P0_CITY', 'NODE6_P0_SETTLEMENT', 'NODE6_P1_CITY', 'NODE6_P1_SETTLEMENT', 'NODE7_P0_CITY', 'NODE7_P0_SETTLEMENT', 'NODE7_P1_CITY', 'NODE7_P1_SETTLEMENT', 'NODE8_P0_CITY', 'NODE8_P0_SETTLEMENT', 'NODE8_P1_CITY', 'NODE8_P1_SETTLEMENT', 'NODE9_P0_CITY', 'NODE9_P0_SETTLEMENT', 'NODE9_P1_CITY', 'NODE9_P1_SETTLEMENT'],
    #     'e':['PORT0_IS_BRICK', 'PORT0_IS_ORE', 'PORT0_IS_SHEEP', 'PORT0_IS_THREE_TO_ONE', 'PORT0_IS_WHEAT', 'PORT0_IS_WOOD', 'PORT1_IS_BRICK', 'PORT1_IS_ORE', 'PORT1_IS_SHEEP', 'PORT1_IS_THREE_TO_ONE', 'PORT1_IS_WHEAT', 'PORT1_IS_WOOD', 'PORT2_IS_BRICK', 'PORT2_IS_ORE', 'PORT2_IS_SHEEP', 'PORT2_IS_THREE_TO_ONE', 'PORT2_IS_WHEAT', 'PORT2_IS_WOOD', 'PORT3_IS_BRICK', 'PORT3_IS_ORE', 'PORT3_IS_SHEEP', 'PORT3_IS_THREE_TO_ONE', 'PORT3_IS_WHEAT', 'PORT3_IS_WOOD', 'PORT4_IS_BRICK', 'PORT4_IS_ORE', 'PORT4_IS_SHEEP', 'PORT4_IS_THREE_TO_ONE', 'PORT4_IS_WHEAT', 'PORT4_IS_WOOD', 'PORT5_IS_BRICK', 'PORT5_IS_ORE', 'PORT5_IS_SHEEP', 'PORT5_IS_THREE_TO_ONE', 'PORT5_IS_WHEAT', 'PORT5_IS_WOOD', 'PORT6_IS_BRICK', 'PORT6_IS_ORE', 'PORT6_IS_SHEEP', 'PORT6_IS_THREE_TO_ONE', 'PORT6_IS_WHEAT', 'PORT6_IS_WOOD', 'PORT7_IS_BRICK', 'PORT7_IS_ORE', 'PORT7_IS_SHEEP', 'PORT7_IS_THREE_TO_ONE', 'PORT7_IS_WHEAT', 'PORT7_IS_WOOD', 'PORT8_IS_BRICK', 'PORT8_IS_ORE', 'PORT8_IS_SHEEP', 'PORT8_IS_THREE_TO_ONE', 'PORT8_IS_WHEAT', 'PORT8_IS_WOOD'],
    #     'f':['IS_DISCARDING', 'IS_MOVING_ROBBER'],
    #     'g':['P0_HAS_ROLLED', 'P1_HAS_ROLLED'],
    #     'h':['P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'],
    #     'i':['P0_ACTUAL_VPS', 'P0_PUBLIC_VPS'],
    #     'j':['P0_BRICK_IN_HAND', 'P0_KNIGHT_IN_HAND', 'P0_MONOPOLY_IN_HAND', 'P0_NUM_DEVS_IN_HAND', 'P0_NUM_RESOURCES_IN_HAND', 'P0_ORE_IN_HAND', 'P0_ROAD_BUILDING_IN_HAND', 'P0_SHEEP_IN_HAND', 'P0_VICTORY_POINT_IN_HAND', 'P0_WHEAT_IN_HAND', 'P0_WOOD_IN_HAND', 'P0_YEAR_OF_PLENTY_IN_HAND'],
    #     'k':['P1_NUM_DEVS_IN_HAND', 'P1_NUM_RESOURCES_IN_HAND'],
    #     'l':['P0_HAS_ARMY'],
    #     'm':['P1_HAS_ARMY'],
    #     'n':['P0_HAS_ROAD'],
    #     'o':['P1_HAS_ROAD'],
    #     'p':['P0_CITIES_LEFT', 'P0_ROADS_LEFT', 'P0_SETTLEMENTS_LEFT'],
    #     'q':['P1_CITIES_LEFT', 'P1_ROADS_LEFT', 'P1_SETTLEMENTS_LEFT'],
    #     'r':['P0_KNIGHT_PLAYED', 'P0_MONOPOLY_PLAYED', 'P0_ROAD_BUILDING_PLAYED', 'P0_YEAR_OF_PLENTY_PLAYED'],
    #     's':['P1_KNIGHT_PLAYED', 'P1_MONOPOLY_PLAYED', 'P1_ROAD_BUILDING_PLAYED', 'P1_YEAR_OF_PLENTY_PLAYED'],
    #     't':['P1_PUBLIC_VPS'],
    #     'u':['P0_LONGEST_ROAD_LENGTH'],
    #     'v':['P1_LONGEST_ROAD_LENGTH'],
    # }
    #load_model
    model = PPO.load(path =agent_path, device='cuda')
    print(dfy_test.mean())
    print(dfy_train.mean())
    print("using grad", torch.is_grad_enabled())
    # model.predict()[0]

    df_shapley, pred_explain, internal, timing ,MSEv= explain(
        model = model,
        x_train = dfx_train,
        x_explain = dfx_test,
        approach = approach,
        # n_combinations=2**614,
        n_samples=nsamples,
        n_batches=nbatches,
        predict_model= lambda m,x: m.predict(x,deterministic =True)[0],
        prediction_zero = dfy_test.mean().item(),
        group=group,
        verbose=1,
    )
    return df_shapley, pred_explain, internal, timing, MSEv

def groupshap_new(env_config= None, approach:str ="independence", grouping =None, agent_path = "./logs/Catan_Switch_Exp/exp_019/latest_model_1000000_steps.zip", nsamples=None,nbatches=None):
    print("using grad", torch.is_grad_enabled())
    data, info_list, catan_features = run_switch_agent(env_config=env_config, agent_path=agent_path)
    features = ["ep","step",'obs', 'action']
    # df = pd.DataFrame(data=data, columns=features)
    
    # print("first df",df)
    group = get_group(catan_features, grouping)
    # print(group)
    df = get_games("./good_games/")
    print("good games",df)
    print(ast.literal_eval(df['obs'][0]))
    #TODO: finish setting up the use of good games
    # return
    #format data
    dfx_train, dfx_test, dfy_train, dfy_test = prep_data(df,catan_features)
    # print(dfx_train.dtypes)
    # print(dfx_train.columns.values.tolist())
    # print(len(catan_features))
    # r_group = NULL if group is None else ListVector({key: StrVector(value) for key, value in group.items()})
    # print(r_group)

    # group = {
    #     'a':['BANK_BRICK', 'BANK_DEV_CARDS', 'BANK_ORE', 'BANK_SHEEP', 'BANK_WHEAT', 'BANK_WOOD'],
    #     'b':['EDGE(0, 1)_P0_ROAD', 'EDGE(0, 1)_P1_ROAD', 'EDGE(0, 20)_P0_ROAD', 'EDGE(0, 20)_P1_ROAD', 'EDGE(0, 5)_P0_ROAD', 'EDGE(0, 5)_P1_ROAD', 'EDGE(1, 2)_P0_ROAD', 'EDGE(1, 2)_P1_ROAD', 'EDGE(1, 6)_P0_ROAD', 'EDGE(1, 6)_P1_ROAD', 'EDGE(10, 11)_P0_ROAD', 'EDGE(10, 11)_P1_ROAD', 'EDGE(10, 29)_P0_ROAD', 'EDGE(10, 29)_P1_ROAD', 'EDGE(11, 12)_P0_ROAD', 'EDGE(11, 12)_P1_ROAD', 'EDGE(11, 32)_P0_ROAD', 'EDGE(11, 32)_P1_ROAD', 'EDGE(12, 13)_P0_ROAD', 'EDGE(12, 13)_P1_ROAD', 'EDGE(13, 14)_P0_ROAD', 'EDGE(13, 14)_P1_ROAD', 'EDGE(13, 34)_P0_ROAD', 'EDGE(13, 34)_P1_ROAD', 'EDGE(14, 15)_P0_ROAD', 'EDGE(14, 15)_P1_ROAD', 'EDGE(14, 37)_P0_ROAD', 'EDGE(14, 37)_P1_ROAD', 'EDGE(15, 17)_P0_ROAD', 'EDGE(15, 17)_P1_ROAD', 'EDGE(16, 18)_P0_ROAD', 'EDGE(16, 18)_P1_ROAD', 'EDGE(16, 21)_P0_ROAD', 'EDGE(16, 21)_P1_ROAD', 'EDGE(17, 18)_P0_ROAD', 'EDGE(17, 18)_P1_ROAD', 'EDGE(17, 39)_P0_ROAD', 'EDGE(17, 39)_P1_ROAD', 'EDGE(18, 40)_P0_ROAD', 'EDGE(18, 40)_P1_ROAD', 'EDGE(19, 20)_P0_ROAD', 'EDGE(19, 20)_P1_ROAD', 'EDGE(19, 21)_P0_ROAD', 'EDGE(19, 21)_P1_ROAD', 'EDGE(19, 46)_P0_ROAD', 'EDGE(19, 46)_P1_ROAD', 'EDGE(2, 3)_P0_ROAD', 'EDGE(2, 3)_P1_ROAD', 'EDGE(2, 9)_P0_ROAD', 'EDGE(2, 9)_P1_ROAD', 'EDGE(20, 22)_P0_ROAD', 'EDGE(20, 22)_P1_ROAD', 'EDGE(21, 43)_P0_ROAD', 'EDGE(21, 43)_P1_ROAD', 'EDGE(22, 23)_P0_ROAD', 'EDGE(22, 23)_P1_ROAD', 'EDGE(22, 49)_P0_ROAD', 'EDGE(22, 49)_P1_ROAD', 'EDGE(23, 52)_P0_ROAD', 'EDGE(23, 52)_P1_ROAD', 'EDGE(24, 25)_P0_ROAD', 'EDGE(24, 25)_P1_ROAD', 'EDGE(24, 53)_P0_ROAD', 'EDGE(24, 53)_P1_ROAD', 'EDGE(25, 26)_P0_ROAD', 'EDGE(25, 26)_P1_ROAD', 'EDGE(26, 27)_P0_ROAD', 'EDGE(26, 27)_P1_ROAD', 'EDGE(27, 28)_P0_ROAD', 'EDGE(27, 28)_P1_ROAD', 'EDGE(28, 29)_P0_ROAD', 'EDGE(28, 29)_P1_ROAD', 'EDGE(29, 30)_P0_ROAD', 'EDGE(29, 30)_P1_ROAD', 'EDGE(3, 12)_P0_ROAD', 'EDGE(3, 12)_P1_ROAD', 'EDGE(3, 4)_P0_ROAD', 'EDGE(3, 4)_P1_ROAD', 'EDGE(30, 31)_P0_ROAD', 'EDGE(30, 31)_P1_ROAD', 'EDGE(31, 32)_P0_ROAD', 'EDGE(31, 32)_P1_ROAD', 'EDGE(32, 33)_P0_ROAD', 'EDGE(32, 33)_P1_ROAD', 'EDGE(33, 34)_P0_ROAD', 'EDGE(33, 34)_P1_ROAD', 'EDGE(34, 35)_P0_ROAD', 'EDGE(34, 35)_P1_ROAD', 'EDGE(35, 36)_P0_ROAD', 'EDGE(35, 36)_P1_ROAD', 'EDGE(36, 37)_P0_ROAD', 'EDGE(36, 37)_P1_ROAD', 'EDGE(37, 38)_P0_ROAD', 'EDGE(37, 38)_P1_ROAD', 'EDGE(38, 39)_P0_ROAD', 'EDGE(38, 39)_P1_ROAD', 'EDGE(39, 41)_P0_ROAD', 'EDGE(39, 41)_P1_ROAD', 'EDGE(4, 15)_P0_ROAD', 'EDGE(4, 15)_P1_ROAD', 'EDGE(4, 5)_P0_ROAD', 'EDGE(4, 5)_P1_ROAD', 'EDGE(40, 42)_P0_ROAD', 'EDGE(40, 42)_P1_ROAD', 'EDGE(40, 44)_P0_ROAD', 'EDGE(40, 44)_P1_ROAD', 'EDGE(41, 42)_P0_ROAD', 'EDGE(41, 42)_P1_ROAD', 'EDGE(43, 44)_P0_ROAD', 'EDGE(43, 44)_P1_ROAD', 'EDGE(43, 47)_P0_ROAD', 'EDGE(43, 47)_P1_ROAD', 'EDGE(45, 46)_P0_ROAD', 'EDGE(45, 46)_P1_ROAD', 'EDGE(45, 47)_P0_ROAD', 'EDGE(45, 47)_P1_ROAD', 'EDGE(46, 48)_P0_ROAD', 'EDGE(46, 48)_P1_ROAD', 'EDGE(48, 49)_P0_ROAD', 'EDGE(48, 49)_P1_ROAD', 'EDGE(49, 50)_P0_ROAD', 'EDGE(49, 50)_P1_ROAD', 'EDGE(5, 16)_P0_ROAD', 'EDGE(5, 16)_P1_ROAD', 'EDGE(50, 51)_P0_ROAD', 'EDGE(50, 51)_P1_ROAD', 'EDGE(51, 52)_P0_ROAD', 'EDGE(51, 52)_P1_ROAD', 'EDGE(52, 53)_P0_ROAD', 'EDGE(52, 53)_P1_ROAD', 'EDGE(6, 23)_P0_ROAD', 'EDGE(6, 23)_P1_ROAD', 'EDGE(6, 7)_P0_ROAD', 'EDGE(6, 7)_P1_ROAD', 'EDGE(7, 24)_P0_ROAD', 'EDGE(7, 24)_P1_ROAD', 'EDGE(7, 8)_P0_ROAD', 'EDGE(7, 8)_P1_ROAD', 'EDGE(8, 27)_P0_ROAD', 'EDGE(8, 27)_P1_ROAD', 'EDGE(8, 9)_P0_ROAD', 'EDGE(8, 9)_P1_ROAD', 'EDGE(9, 10)_P0_ROAD', 'EDGE(9, 10)_P1_ROAD'],
    #     'c':['TILE0_HAS_ROBBER', 'TILE0_IS_BRICK', 'TILE0_IS_DESERT', 'TILE0_IS_ORE', 'TILE0_IS_SHEEP', 'TILE0_IS_WHEAT', 'TILE0_IS_WOOD', 'TILE0_PROBA', 'TILE10_HAS_ROBBER', 'TILE10_IS_BRICK', 'TILE10_IS_DESERT', 'TILE10_IS_ORE', 'TILE10_IS_SHEEP', 'TILE10_IS_WHEAT', 'TILE10_IS_WOOD', 'TILE10_PROBA', 'TILE11_HAS_ROBBER', 'TILE11_IS_BRICK', 'TILE11_IS_DESERT', 'TILE11_IS_ORE', 'TILE11_IS_SHEEP', 'TILE11_IS_WHEAT', 'TILE11_IS_WOOD', 'TILE11_PROBA', 'TILE12_HAS_ROBBER', 'TILE12_IS_BRICK', 'TILE12_IS_DESERT', 'TILE12_IS_ORE', 'TILE12_IS_SHEEP', 'TILE12_IS_WHEAT', 'TILE12_IS_WOOD', 'TILE12_PROBA', 'TILE13_HAS_ROBBER', 'TILE13_IS_BRICK', 'TILE13_IS_DESERT', 'TILE13_IS_ORE', 'TILE13_IS_SHEEP', 'TILE13_IS_WHEAT', 'TILE13_IS_WOOD', 'TILE13_PROBA', 'TILE14_HAS_ROBBER', 'TILE14_IS_BRICK', 'TILE14_IS_DESERT', 'TILE14_IS_ORE', 'TILE14_IS_SHEEP', 'TILE14_IS_WHEAT', 'TILE14_IS_WOOD', 'TILE14_PROBA', 'TILE15_HAS_ROBBER', 'TILE15_IS_BRICK', 'TILE15_IS_DESERT', 'TILE15_IS_ORE', 'TILE15_IS_SHEEP', 'TILE15_IS_WHEAT', 'TILE15_IS_WOOD', 'TILE15_PROBA', 'TILE16_HAS_ROBBER', 'TILE16_IS_BRICK', 'TILE16_IS_DESERT', 'TILE16_IS_ORE', 'TILE16_IS_SHEEP', 'TILE16_IS_WHEAT', 'TILE16_IS_WOOD', 'TILE16_PROBA', 'TILE17_HAS_ROBBER', 'TILE17_IS_BRICK', 'TILE17_IS_DESERT', 'TILE17_IS_ORE', 'TILE17_IS_SHEEP', 'TILE17_IS_WHEAT', 'TILE17_IS_WOOD', 'TILE17_PROBA', 'TILE18_HAS_ROBBER', 'TILE18_IS_BRICK', 'TILE18_IS_DESERT', 'TILE18_IS_ORE', 'TILE18_IS_SHEEP', 'TILE18_IS_WHEAT', 'TILE18_IS_WOOD', 'TILE18_PROBA', 'TILE1_HAS_ROBBER', 'TILE1_IS_BRICK', 'TILE1_IS_DESERT', 'TILE1_IS_ORE', 'TILE1_IS_SHEEP', 'TILE1_IS_WHEAT', 'TILE1_IS_WOOD', 'TILE1_PROBA', 'TILE2_HAS_ROBBER', 'TILE2_IS_BRICK', 'TILE2_IS_DESERT', 'TILE2_IS_ORE', 'TILE2_IS_SHEEP', 'TILE2_IS_WHEAT', 'TILE2_IS_WOOD', 'TILE2_PROBA', 'TILE3_HAS_ROBBER', 'TILE3_IS_BRICK', 'TILE3_IS_DESERT', 'TILE3_IS_ORE', 'TILE3_IS_SHEEP', 'TILE3_IS_WHEAT', 'TILE3_IS_WOOD', 'TILE3_PROBA', 'TILE4_HAS_ROBBER', 'TILE4_IS_BRICK', 'TILE4_IS_DESERT', 'TILE4_IS_ORE', 'TILE4_IS_SHEEP', 'TILE4_IS_WHEAT', 'TILE4_IS_WOOD', 'TILE4_PROBA', 'TILE5_HAS_ROBBER', 'TILE5_IS_BRICK', 'TILE5_IS_DESERT', 'TILE5_IS_ORE', 'TILE5_IS_SHEEP', 'TILE5_IS_WHEAT', 'TILE5_IS_WOOD', 'TILE5_PROBA', 'TILE6_HAS_ROBBER', 'TILE6_IS_BRICK', 'TILE6_IS_DESERT', 'TILE6_IS_ORE', 'TILE6_IS_SHEEP', 'TILE6_IS_WHEAT', 'TILE6_IS_WOOD', 'TILE6_PROBA', 'TILE7_HAS_ROBBER', 'TILE7_IS_BRICK', 'TILE7_IS_DESERT', 'TILE7_IS_ORE', 'TILE7_IS_SHEEP', 'TILE7_IS_WHEAT', 'TILE7_IS_WOOD', 'TILE7_PROBA', 'TILE8_HAS_ROBBER', 'TILE8_IS_BRICK', 'TILE8_IS_DESERT', 'TILE8_IS_ORE', 'TILE8_IS_SHEEP', 'TILE8_IS_WHEAT', 'TILE8_IS_WOOD', 'TILE8_PROBA', 'TILE9_HAS_ROBBER', 'TILE9_IS_BRICK', 'TILE9_IS_DESERT', 'TILE9_IS_ORE', 'TILE9_IS_SHEEP', 'TILE9_IS_WHEAT', 'TILE9_IS_WOOD', 'TILE9_PROBA'],
    #     'd':['NODE0_P0_CITY', 'NODE0_P0_SETTLEMENT', 'NODE0_P1_CITY', 'NODE0_P1_SETTLEMENT', 'NODE10_P0_CITY', 'NODE10_P0_SETTLEMENT', 'NODE10_P1_CITY', 'NODE10_P1_SETTLEMENT', 'NODE11_P0_CITY', 'NODE11_P0_SETTLEMENT', 'NODE11_P1_CITY', 'NODE11_P1_SETTLEMENT', 'NODE12_P0_CITY', 'NODE12_P0_SETTLEMENT', 'NODE12_P1_CITY', 'NODE12_P1_SETTLEMENT', 'NODE13_P0_CITY', 'NODE13_P0_SETTLEMENT', 'NODE13_P1_CITY', 'NODE13_P1_SETTLEMENT', 'NODE14_P0_CITY', 'NODE14_P0_SETTLEMENT', 'NODE14_P1_CITY', 'NODE14_P1_SETTLEMENT', 'NODE15_P0_CITY', 'NODE15_P0_SETTLEMENT', 'NODE15_P1_CITY', 'NODE15_P1_SETTLEMENT', 'NODE16_P0_CITY', 'NODE16_P0_SETTLEMENT', 'NODE16_P1_CITY', 'NODE16_P1_SETTLEMENT', 'NODE17_P0_CITY', 'NODE17_P0_SETTLEMENT', 'NODE17_P1_CITY', 'NODE17_P1_SETTLEMENT', 'NODE18_P0_CITY', 'NODE18_P0_SETTLEMENT', 'NODE18_P1_CITY', 'NODE18_P1_SETTLEMENT', 'NODE19_P0_CITY', 'NODE19_P0_SETTLEMENT', 'NODE19_P1_CITY', 'NODE19_P1_SETTLEMENT', 'NODE1_P0_CITY', 'NODE1_P0_SETTLEMENT', 'NODE1_P1_CITY', 'NODE1_P1_SETTLEMENT', 'NODE20_P0_CITY', 'NODE20_P0_SETTLEMENT', 'NODE20_P1_CITY', 'NODE20_P1_SETTLEMENT', 'NODE21_P0_CITY', 'NODE21_P0_SETTLEMENT', 'NODE21_P1_CITY', 'NODE21_P1_SETTLEMENT', 'NODE22_P0_CITY', 'NODE22_P0_SETTLEMENT', 'NODE22_P1_CITY', 'NODE22_P1_SETTLEMENT', 'NODE23_P0_CITY', 'NODE23_P0_SETTLEMENT', 'NODE23_P1_CITY', 'NODE23_P1_SETTLEMENT', 'NODE24_P0_CITY', 'NODE24_P0_SETTLEMENT', 'NODE24_P1_CITY', 'NODE24_P1_SETTLEMENT', 'NODE25_P0_CITY', 'NODE25_P0_SETTLEMENT', 'NODE25_P1_CITY', 'NODE25_P1_SETTLEMENT', 'NODE26_P0_CITY', 'NODE26_P0_SETTLEMENT', 'NODE26_P1_CITY', 'NODE26_P1_SETTLEMENT', 'NODE27_P0_CITY', 'NODE27_P0_SETTLEMENT', 'NODE27_P1_CITY', 'NODE27_P1_SETTLEMENT', 'NODE28_P0_CITY', 'NODE28_P0_SETTLEMENT', 'NODE28_P1_CITY', 'NODE28_P1_SETTLEMENT', 'NODE29_P0_CITY', 'NODE29_P0_SETTLEMENT', 'NODE29_P1_CITY', 'NODE29_P1_SETTLEMENT', 'NODE2_P0_CITY', 'NODE2_P0_SETTLEMENT', 'NODE2_P1_CITY', 'NODE2_P1_SETTLEMENT', 'NODE30_P0_CITY', 'NODE30_P0_SETTLEMENT', 'NODE30_P1_CITY', 'NODE30_P1_SETTLEMENT', 'NODE31_P0_CITY', 'NODE31_P0_SETTLEMENT', 'NODE31_P1_CITY', 'NODE31_P1_SETTLEMENT', 'NODE32_P0_CITY', 'NODE32_P0_SETTLEMENT', 'NODE32_P1_CITY', 'NODE32_P1_SETTLEMENT', 'NODE33_P0_CITY', 'NODE33_P0_SETTLEMENT', 'NODE33_P1_CITY', 'NODE33_P1_SETTLEMENT', 'NODE34_P0_CITY', 'NODE34_P0_SETTLEMENT', 'NODE34_P1_CITY', 'NODE34_P1_SETTLEMENT', 'NODE35_P0_CITY', 'NODE35_P0_SETTLEMENT', 'NODE35_P1_CITY', 'NODE35_P1_SETTLEMENT', 'NODE36_P0_CITY', 'NODE36_P0_SETTLEMENT', 'NODE36_P1_CITY', 'NODE36_P1_SETTLEMENT', 'NODE37_P0_CITY', 'NODE37_P0_SETTLEMENT', 'NODE37_P1_CITY', 'NODE37_P1_SETTLEMENT', 'NODE38_P0_CITY', 'NODE38_P0_SETTLEMENT', 'NODE38_P1_CITY', 'NODE38_P1_SETTLEMENT', 'NODE39_P0_CITY', 'NODE39_P0_SETTLEMENT', 'NODE39_P1_CITY', 'NODE39_P1_SETTLEMENT', 'NODE3_P0_CITY', 'NODE3_P0_SETTLEMENT', 'NODE3_P1_CITY', 'NODE3_P1_SETTLEMENT', 'NODE40_P0_CITY', 'NODE40_P0_SETTLEMENT', 'NODE40_P1_CITY', 'NODE40_P1_SETTLEMENT', 'NODE41_P0_CITY', 'NODE41_P0_SETTLEMENT', 'NODE41_P1_CITY', 'NODE41_P1_SETTLEMENT', 'NODE42_P0_CITY', 'NODE42_P0_SETTLEMENT', 'NODE42_P1_CITY', 'NODE42_P1_SETTLEMENT', 'NODE43_P0_CITY', 'NODE43_P0_SETTLEMENT', 'NODE43_P1_CITY', 'NODE43_P1_SETTLEMENT', 'NODE44_P0_CITY', 'NODE44_P0_SETTLEMENT', 'NODE44_P1_CITY', 'NODE44_P1_SETTLEMENT', 'NODE45_P0_CITY', 'NODE45_P0_SETTLEMENT', 'NODE45_P1_CITY', 'NODE45_P1_SETTLEMENT', 'NODE46_P0_CITY', 'NODE46_P0_SETTLEMENT', 'NODE46_P1_CITY', 'NODE46_P1_SETTLEMENT', 'NODE47_P0_CITY', 'NODE47_P0_SETTLEMENT', 'NODE47_P1_CITY', 'NODE47_P1_SETTLEMENT', 'NODE48_P0_CITY', 'NODE48_P0_SETTLEMENT', 'NODE48_P1_CITY', 'NODE48_P1_SETTLEMENT', 'NODE49_P0_CITY', 'NODE49_P0_SETTLEMENT', 'NODE49_P1_CITY', 'NODE49_P1_SETTLEMENT', 'NODE4_P0_CITY', 'NODE4_P0_SETTLEMENT', 'NODE4_P1_CITY', 'NODE4_P1_SETTLEMENT', 'NODE50_P0_CITY', 'NODE50_P0_SETTLEMENT', 'NODE50_P1_CITY', 'NODE50_P1_SETTLEMENT', 'NODE51_P0_CITY', 'NODE51_P0_SETTLEMENT', 'NODE51_P1_CITY', 'NODE51_P1_SETTLEMENT', 'NODE52_P0_CITY', 'NODE52_P0_SETTLEMENT', 'NODE52_P1_CITY', 'NODE52_P1_SETTLEMENT', 'NODE53_P0_CITY', 'NODE53_P0_SETTLEMENT', 'NODE53_P1_CITY', 'NODE53_P1_SETTLEMENT', 'NODE5_P0_CITY', 'NODE5_P0_SETTLEMENT', 'NODE5_P1_CITY', 'NODE5_P1_SETTLEMENT', 'NODE6_P0_CITY', 'NODE6_P0_SETTLEMENT', 'NODE6_P1_CITY', 'NODE6_P1_SETTLEMENT', 'NODE7_P0_CITY', 'NODE7_P0_SETTLEMENT', 'NODE7_P1_CITY', 'NODE7_P1_SETTLEMENT', 'NODE8_P0_CITY', 'NODE8_P0_SETTLEMENT', 'NODE8_P1_CITY', 'NODE8_P1_SETTLEMENT', 'NODE9_P0_CITY', 'NODE9_P0_SETTLEMENT', 'NODE9_P1_CITY', 'NODE9_P1_SETTLEMENT'],
    #     'e':['PORT0_IS_BRICK', 'PORT0_IS_ORE', 'PORT0_IS_SHEEP', 'PORT0_IS_THREE_TO_ONE', 'PORT0_IS_WHEAT', 'PORT0_IS_WOOD', 'PORT1_IS_BRICK', 'PORT1_IS_ORE', 'PORT1_IS_SHEEP', 'PORT1_IS_THREE_TO_ONE', 'PORT1_IS_WHEAT', 'PORT1_IS_WOOD', 'PORT2_IS_BRICK', 'PORT2_IS_ORE', 'PORT2_IS_SHEEP', 'PORT2_IS_THREE_TO_ONE', 'PORT2_IS_WHEAT', 'PORT2_IS_WOOD', 'PORT3_IS_BRICK', 'PORT3_IS_ORE', 'PORT3_IS_SHEEP', 'PORT3_IS_THREE_TO_ONE', 'PORT3_IS_WHEAT', 'PORT3_IS_WOOD', 'PORT4_IS_BRICK', 'PORT4_IS_ORE', 'PORT4_IS_SHEEP', 'PORT4_IS_THREE_TO_ONE', 'PORT4_IS_WHEAT', 'PORT4_IS_WOOD', 'PORT5_IS_BRICK', 'PORT5_IS_ORE', 'PORT5_IS_SHEEP', 'PORT5_IS_THREE_TO_ONE', 'PORT5_IS_WHEAT', 'PORT5_IS_WOOD', 'PORT6_IS_BRICK', 'PORT6_IS_ORE', 'PORT6_IS_SHEEP', 'PORT6_IS_THREE_TO_ONE', 'PORT6_IS_WHEAT', 'PORT6_IS_WOOD', 'PORT7_IS_BRICK', 'PORT7_IS_ORE', 'PORT7_IS_SHEEP', 'PORT7_IS_THREE_TO_ONE', 'PORT7_IS_WHEAT', 'PORT7_IS_WOOD', 'PORT8_IS_BRICK', 'PORT8_IS_ORE', 'PORT8_IS_SHEEP', 'PORT8_IS_THREE_TO_ONE', 'PORT8_IS_WHEAT', 'PORT8_IS_WOOD'],
    #     'f':['IS_DISCARDING', 'IS_MOVING_ROBBER'],
    #     'g':['P0_HAS_ROLLED', 'P1_HAS_ROLLED'],
    #     'h':['P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'],
    #     'i':['P0_ACTUAL_VPS', 'P0_PUBLIC_VPS'],
    #     'j':['P0_BRICK_IN_HAND', 'P0_KNIGHT_IN_HAND', 'P0_MONOPOLY_IN_HAND', 'P0_NUM_DEVS_IN_HAND', 'P0_NUM_RESOURCES_IN_HAND', 'P0_ORE_IN_HAND', 'P0_ROAD_BUILDING_IN_HAND', 'P0_SHEEP_IN_HAND', 'P0_VICTORY_POINT_IN_HAND', 'P0_WHEAT_IN_HAND', 'P0_WOOD_IN_HAND', 'P0_YEAR_OF_PLENTY_IN_HAND'],
    #     'k':['P1_NUM_DEVS_IN_HAND', 'P1_NUM_RESOURCES_IN_HAND'],
    #     'l':['P0_HAS_ARMY'],
    #     'm':['P1_HAS_ARMY'],
    #     'n':['P0_HAS_ROAD'],
    #     'o':['P1_HAS_ROAD'],
    #     'p':['P0_CITIES_LEFT', 'P0_ROADS_LEFT', 'P0_SETTLEMENTS_LEFT'],
    #     'q':['P1_CITIES_LEFT', 'P1_ROADS_LEFT', 'P1_SETTLEMENTS_LEFT'],
    #     'r':['P0_KNIGHT_PLAYED', 'P0_MONOPOLY_PLAYED', 'P0_ROAD_BUILDING_PLAYED', 'P0_YEAR_OF_PLENTY_PLAYED'],
    #     's':['P1_KNIGHT_PLAYED', 'P1_MONOPOLY_PLAYED', 'P1_ROAD_BUILDING_PLAYED', 'P1_YEAR_OF_PLENTY_PLAYED'],
    #     't':['P1_PUBLIC_VPS'],
    #     'u':['P0_LONGEST_ROAD_LENGTH'],
    #     'v':['P1_LONGEST_ROAD_LENGTH'],
    # }
    #load_model
    model = PPO.load(path =agent_path, device='cuda')
    print(dfy_test.mean())
    print(dfy_train.mean())
    print("using grad", torch.is_grad_enabled())
    # model.predict()[0]

    df_shapley, pred_explain, internal, timing ,MSEv= explain(
        model = model,
        x_train = dfx_train,
        x_explain = dfx_test,
        approach = approach,
        # n_combinations=2**614,
        n_samples=nsamples,
        n_batches=nbatches,
        predict_model= lambda m,x: m.predict(x,deterministic =True)[0],
        prediction_zero = dfy_test.mean().item(),
        group=group,
        verbose=1,
    )
    return df_shapley, pred_explain, internal, timing, MSEv

def test_shap():
    from sklearn.ensemble import RandomForestRegressor
    from shaprpy import explain
    from shaprpy.datasets import load_california_housing

    dfx_train, dfx_test, dfy_train, dfy_test = load_california_housing()

    ## Fit model
    model = RandomForestRegressor()
    model.fit(dfx_train, dfy_train.values.flatten())

    ## Shapr
    df_shapley, pred_explain, internal, timing ,MSEv= explain(
        model = model,
        x_train = dfx_train,
        x_explain = dfx_test,
        approach = 'empirical',
        prediction_zero = dfy_train.mean().item(),
        verbose= 1
    )
    print(df_shapley)

def check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        # If it doesn't exist, create it
        os.makedirs(dir_path)

def parse_args():
    parser = argparse.ArgumentParser("Catanatron Switch Training")
    parser.add_argument("--json_path", type=str, default="./experiments_cluster/test_exp.json", help="directory for expeiment parameters")
    
    return parser.parse_args()

if __name__ == '__main__':
    # get experiment parameters
    args = parse_args()
    # Load params from JSON file
    with open(f"{args.json_path}.json", 'r') as f:
        config = json.load(f)

    group = config["group"]
    n_samples = config["n_samples"]
    n_batches = config["n_batches"]
    results_dir = f"./shapley_data/{str(group)}/sample_{str(n_samples)}"
    print(results_dir)
    #create results dir if it does not exist
    check_dir_exists(results_dir)
    config={"enemies": [AlphaBetaPlayer(Color.RED,depth=1)]}
    print("using grad", torch.is_grad_enabled())
    #normal shap
    # shap_vals = shapley(config)
    # np.save('./shapley_data/shapley_vals.npy', shap_vals)
    #group_shapley for group 3 (the smallest group) 
    df_shapley, pred_explain, internal, timing, MSEv = groupshap(env_config=config, 
                                                                 approach= 'empirical', 
                                                                 grouping = group, 
                                                                 nsamples = n_samples,
                                                                 nbatches = n_batches)
    df_shapley.to_csv(path_or_buf=f'{results_dir}/shapley_values_{str(group)}.csv')
    print("pred_explain is numpy with shape:",pred_explain.shape)
    np.save(file =f'{results_dir}/pred_explain_{str(group)}.npy',arr=pred_explain)
    # print("internal is dict\n",internal)
    # with open("./shapley_data/internal_group_3.json", "w") as file:
    #     json.dump(internal,file,indent=4)   
    print("timing is dict:\n",timing)
    with open(f"{results_dir}/timing_{str(group)}.json", "w") as file:
        json.dump(timing,fp=file,indent=4)
    print("msev is dict:\n",MSEv)
    with open(f"{results_dir}/MSEv_{str(group)}.json", "w") as file:
        json.dump({'MSEv':MSEv["MSEv"].to_dict(orient='records'),"MSEv_explicand":MSEv["MSEv_explicand"].to_dict(orient='records')},fp =file, indent=4)
    