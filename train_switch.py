import gymnasium as gym
import numpy as np
import json
import wandb
import argparse
import torch as th
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from pathlib import Path
import os
from catanatron_gym.rewards import complex_reward, reward_function, settlement_reward,city_reward,longest_road_reward,dev_card_rewards

def parse_args():
    parser = argparse.ArgumentParser("Catanatron Switch Training")
    parser.add_argument("--json_path", type=str, default="./experiment_json/switch_env_test.json", help="directory for expeiment parameters")
    

    return parser.parse_args()

def mask_fn(env) -> np.ndarray:
    valid_actions = env.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])


def main(config,log_dir):
    catan_env_config={}
    # Extract environment parameters and hyperparameters
    env_config = config['environment_parameters']
    ppo_params = config['ppo_hyperparameters']

    # Make Model dir
    model_path = os.path.join(log_dir, config["experiment_info"]["experiment_id"])
    #best model path
    catan_env_config["map_type"] = env_config["map_type"]
    # process json env config for gym env config
    if env_config["enemies"] == "weighted_random":
        catan_env_config["enemies"] = [WeightedRandomPlayer(Color.RED)]
    elif env_config["enemies"] == "victory_point":
        catan_env_config["enemies"] = [VictoryPointPlayer(Color.RED)]
    elif env_config["enemies"] == "alphabeta":
        env_config["enemies"] = [AlphaBetaPlayer(Color.RED)]
    elif env_config["enemies"] == "value":
        env_config["enemies"] = [ValueFunctionPlayer(Color.RED)]
    else:
        catan_env_config["enemies"] = [RandomPlayer(Color.RED)]
    
    if env_config["reward_function"]== "simple":
        env_config["reward_function"] = None
    elif env_config["reward_function"] == "complex":
        catan_env_config["reward_function"] = complex_reward
    elif env_config["reward_function"]== "other_complex":
        catan_env_config["reward_function"] = reward_function
    elif env_config["reward_function"]== "settlement":
        catan_env_config["reward_function"] = settlement_reward
    elif env_config["reward_function"]== "city":
        catan_env_config["reward_function"] = city_reward
    elif env_config["reward_function"]== "longest_road":
        catan_env_config["reward_function"] = longest_road_reward
    elif env_config["reward_function"]== "dev_card":
        catan_env_config["reward_function"] = dev_card_rewards

    if env_config["representation"]== "vector":
        catan_env_config["representation"] = 'vector'
    #TODO add seed
    # catan_env_config['seed'] = env_config["seed"]
    ppo_params["policy_kwargs"]["activation_fn"] = th.nn.Mish
    print(ppo_params)
    # Init Environment and Model
    env = gym.make("catanatron_gym:catanatron-switch-v1",config = catan_env_config)
    # env = make_vec_env("catanatron_gym:catanatron-switch-v1",config = catan_env_config)
    print(env.unwrapped.game)
    # env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    print(env.observation_space)
    #Setup Model
    model = PPO(policy='MlpPolicy', 
                        env= env,
                        learning_rate=ppo_params["learning_rate"],
                        batch_size=ppo_params['batch_size'], 
                        verbose=1, 
                        device=ppo_params['device'], 
                        tensorboard_log=log_dir,
                        policy_kwargs=ppo_params["policy_kwargs"],
                        # ent_coef=ppo_params["ent_coef"],
                        # seed=env_config["seed"]
                        )

    # Setup Callbacks
    eval_callback = EvalCallback(env, best_model_save_path=model_path, log_path=model_path, eval_freq=1000,deterministic=True,verbose=1,render=False) 
    checkpoint_callback= CheckpointCallback(save_freq=10000, save_path=model_path,name_prefix='latest_model',verbose=2)
    callback = CallbackList([checkpoint_callback,eval_callback])
    wandb_callback = WandbCallback(verbose=1, model_save_path=log_dir, model_save_freq=5000)  

    #TODO setup loading from checkpoints (maybe include param in json to say if must load from checkpoint or overwrite it)
    # Train
    model.learn(total_timesteps=ppo_params['total_timesteps'],callback=callback)

if __name__ == '__main__':
    args = parse_args()
    # Load hyperparameters from JSON file
    #TODO: make json readable from args
    with open(f"{args.json_path}.json", 'r') as f:
        config = json.load(f)

    # Extract environment parameters and hyperparameters
    exp_config = config['experiment_info']
    env_config = config['environment_parameters']
    ppo_hyperparameters = config['ppo_hyperparameters']

    #log directory stuff
    log_dir =f"./logs/{exp_config['experiment_name']}"
    #make log dir if not there
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=exp_config['experiment_name'],
        config=config,
        sync_tensorboard=True,  # Automatically sync tensorboard metrics
        name=f"{config['experiment_info']['experiment_id']}",
        notes=config['experiment_info']['description'],
        tags=[config['experiment_info']['date']]
    )

    main(config=config, log_dir=log_dir)
    wandb.finish()