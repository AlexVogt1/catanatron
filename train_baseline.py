import gymnasium as gym
import numpy as np
import json
import wandb
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.models.player import Color, Player, RandomPlayer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from pathlib import Path
import os

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

    # process json env config for gym env config
    if env_config["enemies"] == "weighted_random":
        env_config["enemies"] = [WeightedRandomPlayer(Color.RED)]
    elif env_config["enemies"] == "victory_point":
        env_config["enemies"] = [VictoryPointPlayer(Color.RED)]
    else:
        env_config["enemies"] = None
    
    if env_config["reward_function"]== "simple":
        env_config["reward_function"] = None

    if env_config["representation"]== "vector":
        catan_env_config["representation"] = 'vector'
    # Init Environment and Model
    env = gym.make("catanatron_gym:catanatron-v1",config = catan_env_config)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    print(env.observation_space)
    #Setup Model
    model = MaskablePPO(policy=MaskableActorCriticPolicy, env= env,batch_size=ppo_params['batch_size'], verbose=1, device=ppo_params['device'], tensorboard_log=log_dir)

    # Setup Callbacks
    eval_callback = EvalCallback(env, best_model_save_path=model_path, log_path=log_dir, eval_freq=1000,deterministic=True,verbose=1,render=False) 
    checkpoint_callback= CheckpointCallback(save_freq=10000, save_path=model_path,name_prefix='latest_model',verbose=2)
    callback = CallbackList([checkpoint_callback,eval_callback])
    wandb_callback = WandbCallback(verbose=1, model_save_path=log_dir, model_save_freq=5000)  

    #TODO setup loading from checkpoints (maybe include param in json to say if must load from checkpoint or overwrite it)
    # Train
    model.learn(total_timesteps=ppo_params['total_timesteps'],callback=callback)

if __name__ == '__main__':

    # Load hyperparameters from JSON file
    #TODO: make json readable from args
    with open('experiment_json/baseline.json', 'r') as f:
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