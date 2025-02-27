import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from gym_simplegrid.envs.simple_grid import SimpleGridEnv
from gym_simplegrid.grid_converter import GridConverter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import imageio

from gymnasium.envs.registration import register

register(
    id='SimpleGrid-v0',
    entry_point='gym_simplegrid.envs.simple_grid:SimpleGridEnv',
)


if __name__ == '__main__':
    # Set up the environment with the same configuration as training
    field_width = 3
    field_length = 2
    grid_size = 18
    grid_converter = GridConverter(field_length, field_width, grid_size)
    map_grid = grid_converter.create_grid(max_obstacles=grid_converter.grid_size**2//4)
    
    # Create environment with rendering
    env = gym.make(
        'SimpleGrid-v0',
        obstacle_map=map_grid,
        render_mode=None  # Enable rendering for visualization
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    
    # Specify the path to your trained model
    BASE_DIR = "train_logs"
    MODEL_NUM = 1  # Change this to the model number you want to evaluate
    MODEL_PATH = os.path.join(BASE_DIR, f"model_{MODEL_NUM}", "dqn_simplegrid_final.zip")
    
    model = DQN.load(MODEL_PATH, env=env, device="cpu")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = Monitor(env, allow_early_resets=True)
    episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)

    print("Results after training:")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode lengths: {episode_lengths}")


    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    frames = []
    obs, _ = env.reset()
    done = False

    env.close()