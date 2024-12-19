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

from catanatron.players.search import VictoryPointPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_gym.envs.catanatron_env import from_action_space
from catanatron.json import GameEncoder
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from catanatron_gym.features import create_sample_vector, get_feature_ordering,create_sample
from catanatron_gym.envs.catanatron_env import from_action_space
# FEATURES = get_feature_ordering
from pprint import pprint
import random
import pandas as pd

from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

from selenium import webdriver
from catanatron_server.utils import open_link,ensure_link
import webbrowser
import time
import os

def render(game,ep,step):
    ep_dir = f"./games/episode_{ep}/states"
    geckodriver_path = "/snap/bin/geckodriver"
    driver_service = webdriver.FirefoxService(executable_path=geckodriver_path)

    driver = webdriver.Firefox(service=driver_service)
    link = ensure_link(game)
    driver.get(link)
    time.sleep(1)
    # driver.get_screenshot_as_png('image.png')
    driver.save_screenshot(f'./games/episode_{ep}/states/step_{step}.png')
    # print(driver.get_screenshot_as_png()) 
    driver.close()
    # TODO: save using screenshot as png then convert to pil

def run_switch_agent(env_config, agent_path="./logs/Catan_Switch_Exp/exp_019/latest_model_10000000_steps"):
    # "./logs/Catan_Switch_Exp/exp_019/latest_model_10000000_steps"
    best_path = "../logs/Catan_Switch_Exp/exp_014/best_model.zip"
    latest_path = "../logs/Catan_Switch_Exp/exp_004/latest_model_1000000_steps.zip"
    features = None
    try:
        env = gym.make("catanatron_gym:catanatron-switch-v1", config=env_config)
        print("Made env using config")
    except:
        env = gym.make("catanatron_gym:catanatron-switch-v1")
        print('Made env without config')

    observation, info = env.reset()
    model = PPO.load(path=agent_path, env=env)

    for ep in range(5):
        ep_state_dir = f"./games/episode_{ep}/states"
        os.makedirs(ep_state_dir, exist_ok=True)
        episode_data = []  # Store data for current episode
        episode_info = []  # Store info for current episode

        for step in range(1000):
            action = model.predict(observation=observation, deterministic=True)
            record = create_sample(env.game, env.p0.color)
            features = sorted(record.keys())  # Get list of feature names in same order as observation
            episode_data.append([ep, step, observation.tolist(), action[0]])
            # print(observation)
            observation, reward, terminated, truncated, info = env.step(action[0])
            render(env.game,ep=ep,step=step)
            done = terminated or truncated
            # print(env.p0.color)
            if done:
                # env.unwrapped.game.state()
                game_json = GameEncoder().default(env.game)

                episode_info.append(info) 
                save_episode_data(ep, episode_data, episode_info, features,env)
                episode_info.append(info)  # Save info at the end of each episode
                # print(env.p0.color)
                render(env.game,ep=ep,step=step)
                observation, info = env.reset()
                break

        # Save the episode data and info
        # save_episode_data(ep, episode_data, episode_info, features,env)

    env.close()

def save_episode_data(ep, data, info, features,env):
    # Create directory for the episode
    ep_dir = f"./games/episode_{ep}"
    os.makedirs(ep_dir, exist_ok=True)
    ep_state_dir = f"./games/episode_{ep}/states"
    os.makedirs(ep_state_dir, exist_ok=True)

    # Convert data to DataFrame and save as CSV
    df = pd.DataFrame(data, columns=["ep","step",'obs', 'action'])
    print(df.dtypes)
    df.to_csv(f"{ep_dir}/episode_{ep}_data.csv", index=False)

    # Save info as JSON
    with open(f"{ep_dir}/episode_{ep}_info.json", 'w') as f:
        json.dump(info, f, indent=4)

    game_json = GameEncoder().default(env.game)
    # pprint(game_json)
    print(info)
    # my_json= json.loads(game_json)
    with open(f"{ep_dir}/episode_{ep}_game.json",'w') as f:
        f.write(json.dumps(game_json, cls=GameEncoder,indent=4))
        # json.dump(game_json, f, indent=4,cls=GameEncoder)
    with open(f"{ep_dir}/episode_{ep}_game.txt",'w') as f:
        f.write(str(game_json))
        # json.dump(game_json, f, indent=4,cls=GameEncoder)
    # # Optionally save feature list
    # with open(f"{ep_dir}/episode_{ep}_features.json", 'w') as f:
    #     json.dump(features, f, indent=4)

if __name__ == "__main__":
    config={
        "enemies": [VictoryPointPlayer(Color.RED)],
        # "enemies": [ValueFunctionPlayer(Color.RED)],
        # "enemies": [AlphaBetaPlayer(Color.RED,depth=1)],
        "map_type":"TOURNAMENT"}
    run_switch_agent(config)