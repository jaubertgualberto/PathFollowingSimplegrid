from __future__ import annotations
import logging
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from typing import Tuple
from gym_simplegrid.grid_converter import GridConverter
from gym_simplegrid.d_star_lite import DStarLite

class SimpleGridEnv(Env):
 
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 'render_fps': 8}
    FREE: int = 0
    OBSTACLE: int = 1
    MOVES: dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }

    def __init__(self,     
        obstacle_map: str | list[str],
        obstacles_range: int = 4,
        render_mode: str | None = None,
    ):
        
        self.num_steps = 0
        self.grid_size = obstacle_map.shape[0]
        self.obstacles_range = obstacles_range

        # Env confinguration
        self.obstacles = obstacle_map
        print(self.obstacles)
        print("Obstacles shape: ", self.obstacles.shape)
        
        self.nrow, self.ncol = self.obstacles.shape

        self.action_space = spaces.Discrete(len(self.MOVES))

        self.observation_space = spaces.Box(
            low=np.full(65, -30, dtype=np.float32),   # Lowest possible values
            high=np.full(65, 30, dtype=np.float32),   # Highest possible values
            dtype=np.float64
        )

        # Rendering configuration
        self.fig = None
        self.n_total_iter = 0
        self.threshold = 50000

        self.agent_action = 0
        self.last_action = 2
        self.last_location = None

        # self.options = options

        self.render_mode = render_mode
        self.fps = self.metadata['render_fps']

    def reset(
            self, 
            seed: int | None = None,
        ) -> tuple:

        # Set seed
        super().reset(seed=seed)


        # Change grid at every 50k iterations
        if self.n_total_iter > self.threshold and self.n_total_iter != 0:
            self.create_new_grid()
            print("Grid Changed.") 
            self.threshold+=30000

        self.start_xy, self.goal_xy  = self.get_random_start_goal()


        # If there's no possible path to goal, create a new grid and compute new start and goal positions
        self.path = self.get_path()
        if self.path is None:
            self.create_new_grid()
            self.start_xy, self.goal_xy  = self.get_random_start_goal()

        self.current_path_index = 0


        self.curr_distance = self.get_distance(self.goal_xy, self.start_xy)
        self.last_distance = self.curr_distance

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward, self.done = self.get_reward_and_done(*self.agent_xy)
        # self.done = self.on_goal()

        self.agent_action = 0
        self.last_action = 2
        self.last_location = self.agent_xy

        self.n_iter = 0

        # Check integrity
        self.integrity_checks()


        #if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.get_info()
    
    def create_new_grid(self):
        grid_converter = GridConverter(3, 2, self.grid_size)
        self.obstacles = grid_converter.create_grid(max_obstacles=grid_converter.grid_size**2//self.obstacles_range)
    

        self.path = self.get_path()
        if self.path is None:
            self.create_new_grid()


    def get_random_start_goal(self):
        goal_cell = np.random.randint(0, self.nrow-1), np.random.randint(0, self.ncol-1)
        start_cell = np.random.randint(0, self.nrow-1), np.random.randint(0, self.ncol-1)
        
        while not (self.is_free(*goal_cell) and self.is_free(*start_cell) and goal_cell != start_cell):
            goal_cell = np.random.randint(0, self.nrow), np.random.randint(0, self.ncol)
            start_cell = np.random.randint(0, self.nrow), np.random.randint(0, self.ncol)

        return start_cell, goal_cell

    def get_path(self):
        dstar = DStarLite(grid=self.obstacles, start=self.start_xy, goal=self.goal_xy)
        dstar.compute_shortest_path()
        dstar_path = dstar.reconstruct_path()
    
        if self.goal_xy not in dstar_path:
            return None

        return dstar_path


    def step(self, action: int):
        """
        Take a step in the environment.
        """
        #assert action in self.action_space
        self.agent_action = action


        row, col = self.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward and done
        self.reward, self.done = self.get_reward_and_done(*self.agent_xy)
        
        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.agent_xy = (target_row, target_col)

        self.n_iter += 1
        self.n_total_iter += 1

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.reward, self.done, False, self.get_info()
    
    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert self.obstacles[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with a wall."
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        try:
            return self.obstacles[row, col] == self.FREE
        except:
            return False
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward_and_done(self, x: int, y: int) -> float:
        reward = 0.0
        
        # Out-of-bounds handling
        if not self.is_in_bounds(x, y):
            reward += -5.  
            return reward, True
            
        # Wall collision
        elif not self.is_free(x, y):
            # reward += -5.0  
            return reward, True
        
        # Goal reached
        elif (x, y) == self.goal_xy:
            reward = 100.0  
            return reward, True
        
        else:
            # Path following logic
            if self.current_path_index < len(self.path):
                # Exact path following
                if (x, y) == self.path[self.current_path_index]:
                    reward += 10.0  # Large reward for exact path match
                    self.current_path_index += 1
                
                # Proximity to path rewards/penalties
                else:
                    min_distance = float('inf')
                    closest_index = self.current_path_index
                    
                    # Find closest point on remaining path
                    for i in range(self.current_path_index, len(self.path)):
                        dist = self.get_distance(self.path[i], (x, y))
                        if dist < min_distance:
                            min_distance = dist
                            closest_index = i
                    
                    # Adaptive reward based on path proximity
                    if min_distance < 1:
                        reward += 5.0 * (1 - min_distance)  # Near-path reward
                    elif min_distance < 3:
                        reward += -min_distance  # Mild penalty for deviation
                    else:
                        reward += -2.0 * min_distance  # Strong penalty for large deviation
                    
                    # Update path index if progressing
                    if closest_index > self.current_path_index:
                        self.current_path_index = closest_index
            
            # Minimal efficiency penalty
            reward += -0.01

            return reward, False
        

    def get_reward_and_done_pathfind(self, x: int, y: int) -> float:
        reward = 0.0

        # penalty for out-of-bounds
        if not self.is_in_bounds(x, y):
            reward += -0.2  
            return reward, True
        elif not self.is_free(x, y):
            return reward, True
        elif (x, y) == self.goal_xy:
            reward = 50.0  # Goal reward

            return reward, True
        else:
            # Distance-based reward
            self.curr_distance = self.get_distance(self.goal_xy, (x, y))
            distance_reward = (self.last_distance - self.curr_distance)
            reward += distance_reward * 0.2 
            
            # Step penalty
            reward += -0.001

            # Penalize moving away
            if distance_reward < 0:
                reward += distance_reward * 0.2
            
            self.last_distance = self.curr_distance

            # Penalize staying in place
            if (self.last_location == (x, y)):
                reward += -0.05

            
            self.last_location = (x, y)
            self.last_action = self.agent_action


        return reward, False
    
    def get_distance(self, target: Tuple[int, int], current: Tuple[int, int]) -> float:
        """
        Get the distance between two points.
        """
        # return np.sqrt((target[0] - current[0])**2 + (target[1] - current[1])**2)
        # return Manhattan distance
        return abs(target[0] - current[0]) + abs(target[1] - current[1])

    def get_obs(self) -> np.ndarray:
        agent_x, agent_y = self.agent_xy
        goal_x, goal_y = self.goal_xy
        
        # Path representation
        max_path_length = 25  # Limit to prevent observation space explosion
        padded_path = np.zeros((max_path_length, 2), dtype=np.float32)
        
        # Fill path with relative coordinates from current agent position
        for i in range(min(max_path_length, len(self.path) - self.current_path_index)):
            path_x, path_y = self.path[self.current_path_index + i]
            padded_path[i] = [
                path_x - agent_x,  # Relative x
                path_y - agent_y   # Relative y
            ]

        # Existing components
        relative_x = goal_x - agent_x
        relative_y = goal_y - agent_y
        distance_to_goal = self.get_distance(self.goal_xy, (agent_x, agent_y))
        
        # Wall proximity
        wall_left = float(not self.is_free(agent_x - 1, agent_y))
        wall_right = float(not self.is_free(agent_x + 1, agent_y))
        wall_up = float(not self.is_free(agent_x, agent_y - 1))
        wall_down = float(not self.is_free(agent_x, agent_y + 1))
        
        # Next waypoint relative position
        next_waypoint_x, next_waypoint_y = 0.0, 0.0
        if self.current_path_index < len(self.path):
            next_waypoint_x, next_waypoint_y = self.path[self.current_path_index]
            next_waypoint_x -= agent_x
            next_waypoint_y -= agent_y
        
        # Directional progress indicators
        progress_x = float(np.sign(relative_x))
        progress_y = float(np.sign(relative_y))
        
        # Combine all components
        obs = np.concatenate([
            [agent_x, agent_y],                  # Agent's absolute position
            [goal_x, goal_y],                    # Goal's absolute position
            [relative_x, relative_y],            # Relative position to goal
            [distance_to_goal],                  # Distance to goal
            [wall_left, wall_right, wall_up, wall_down],  # Wall proximity
            [next_waypoint_x, next_waypoint_y],  # Next waypoint relative position
            [progress_x, progress_y],            # Directional progress
            padded_path.flatten()                # Flattened path representation
        ])
        
        return obs

    
    
    def get_info(self) -> dict:
        return {
            'agent_xy': self.agent_xy,
            'n_iter': self.n_iter,
        }

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return None
        
        elif self.render_mode == "ansi":
            s = f"{self.n_iter},{self.agent_xy[0]},{self.agent_xy[1]},{self.reward},{self.done},{self.agent_action}\n"
            #print(s)
            return s

        elif self.render_mode == "rgb_array":
            self.render_frame()
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
        elif self.render_mode == "human":
            self.render_frame()
            plt.pause(1/self.fps)
            return None
        
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self):
        """
        Render the environment frame, ensuring all elements are updated, including the A* path.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.fig.canvas.mpl_connect('close_event', self.close)
        
        self.ax.clear()  # Clear previous frame

        # Draw grid
        for x in range(self.nrow + 1):
            self.ax.plot([0, self.ncol], [x, x], "k-", lw=1)
        for y in range(self.ncol + 1):
            self.ax.plot([y, y], [0, self.nrow], "k-", lw=1)

        # Render obstacles
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.obstacles[row, col] == self.OBSTACLE:
                    self.ax.add_patch(plt.Rectangle((col, self.nrow - row - 1), 1, 1, color="black"))

        # Render the path
        path = self.get_path()
        # path = self.get_dstar_path()
        
        for (x, y) in path:
            self.ax.add_patch(plt.Rectangle((y, self.nrow - x - 1), 1, 1, color="cyan", alpha=0.5))

        # Render agent
        agent_x, agent_y = self.agent_xy
        self.ax.add_patch(
            plt.Circle((agent_y + 0.5, self.nrow - agent_x - 0.5), 0.3, color="blue", label="Agent")
        )

        # Render start position
        start_x, start_y = self.start_xy
        self.ax.add_patch(
            plt.Circle((start_y + 0.5, self.nrow - start_x - 0.5), 0.3, color="green", label="Start")
        )

        # Render goal position
        goal_x, goal_y = self.goal_xy
        self.ax.add_patch(
            plt.Circle((goal_y + 0.5, self.nrow - goal_x - 0.5), 0.3, color="red", label="Goal")
        )

        self.ax.set_xlim(0, self.ncol)
        self.ax.set_ylim(0, self.nrow)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        self.ax.set_title(f"Step: {self.n_iter}, Reward: {self.reward:.2f}")
        self.ax.legend()

        plt.pause(0.001)
    
    def create_agent_patch(self):
        """
        Create a Circle patch for the agent.

        @NOTE: If agent position is (x,y) then, to properly render it, we have to pass (y,x) as center to the Circle patch.
        """
        return mpl.patches.Circle(
            (self.agent_xy[1]+.5, self.agent_xy[0]+.5), 
            0.3, 
            facecolor='orange', 
            fill=True, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=100,
        )

    def update_agent_patch(self):
        """
        @NOTE: If agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) as center to the Circle patch.
        """
        self.agent_patch.center = (self.agent_xy[1]+.5, self.agent_xy[0]+.5)
        return None
    

    def create_white_patch(self, x, y):
        """
        Render a white patch in the given position.
        """
        return mpl.patches.Circle(
            (y+.5, x+.5), 
            0.4, 
            color='white', 
            fill=True, 
            zorder=99,
        )

    def close(self, *args):
        """
        Close the environment.
        """
        plt.close(self.fig)
        sys.exit()