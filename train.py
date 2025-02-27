import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from datetime import datetime as dt
from gym_simplegrid.envs.simple_grid import SimpleGridEnv
from gym_simplegrid.grid_converter import GridConverter
import time
from stable_baselines3.common.monitor import Monitor
from typing import Callable
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == '__main__':
    
    size = 40
    obstacles_range = 3.5

    BASE_DIR = "train_logs"
    os.makedirs(BASE_DIR, exist_ok=True)
    existing_folders = [folder for folder in os.listdir(BASE_DIR) if folder.startswith("model_") and folder[6:].isdigit()]
    next_model_num = max([int(folder[6:]) for folder in existing_folders], default=-1) + 1
    
    FOLDER_NAME = f"model_{next_model_num}"
    LOG_DIR = os.path.join(BASE_DIR, FOLDER_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)

    grid_converter = GridConverter(3, 2, size)
    map_grid = grid_converter.create_grid(max_obstacles=grid_converter.grid_size**2//obstacles_range)


    env = Monitor(
        gym.make(
            'SimpleGrid-v0',
            obstacle_map=map_grid,
            render_mode=None,
            obstacles_range=obstacles_range
        ),
        filename=os.path.join(LOG_DIR, "train")
    )


    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(1e-4), 
        buffer_size=100000, 
        learning_starts=10000, 
        batch_size=256,
        tau=1.0,  
        gamma=0.99,
        train_freq=4,  
        gradient_steps=1,
        target_update_interval=1000,  
        exploration_fraction=0.2,  
        exploration_initial_eps=1.0,  
        exploration_final_eps=0.05, 
        verbose=1,
        tensorboard_log=f"{LOG_DIR}/tensorboard_logs",
        device="cpu"
    )


    # Create evaluation environment
    eval_env = Monitor(
        gym.make(
            'SimpleGrid-v0',
            obstacle_map=map_grid,
            render_mode=None,
            obstacles_range=obstacles_range
        ),
        filename=os.path.join(LOG_DIR, "eval")
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=10000,
        n_eval_episodes=100,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=LOG_DIR,
        name_prefix="dqn_simplegrid"
    )

    time_steps = 4_000_000

    model.learn(
        total_timesteps=time_steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=100, 
        return_episode_rewards=True
    )

    print("Results after training:")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode lengths: {episode_lengths}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save(f"{LOG_DIR}/dqn_simplegrid_final")
    env.close()