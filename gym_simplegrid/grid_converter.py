from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt


class GridConverter:
    def __init__(self, field_length: float, field_width: float, grid_size: int = 24):
        """
        Initialize grid converter
        
        Args:
            field_length: Length of the field in meters
            field_width: Width of the field in meters
            grid_size: Number of cells in each dimension
        """
        self.field_length = field_length
        self.field_width = field_width
        self.grid_size = grid_size
        
        # Calculate cell dimensions
        self.cell_length = field_length / grid_size
        self.cell_width = field_width / grid_size
        
        self.neighbors = {}



    def create_grid(self, max_obstacles = 10) -> np.ndarray:
        """
        Create a grid representation of the environment with random obstacles placed in the grid.
        
        Args:
            max_obstacles: Maximum number of obstacles to place in the grid at random locations.            
        Returns:
            Binary grid where 1 represents obstacles and 0 represents free space
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # For each cell, check if it intersects with any obstacle
        num_obstacles = 0

        while(num_obstacles < max_obstacles):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            grid[x, y] = 1
            num_obstacles += 1
            
        # print(f"Number of obstacles: {num_obstacles}")
        return grid
    
    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous coordinates to grid coordinates"""
        grid_x = int((x + self.field_length/2) / self.cell_length)
        grid_y = int((y + self.field_width/2) / self.cell_width)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_size-1))
        grid_y = max(0, min(grid_y, self.grid_size-1))
        
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to continuous coordinates"""
        x = -self.field_length/2 + grid_x * self.cell_length + self.cell_length/2
        y = -self.field_width/2 + grid_y * self.cell_width + self.cell_width/2
        return x, y
    
    def is_unoccupied(self, pos) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """

        row, col = self.continuous_to_grid(pos[0], pos[1])

        return self.occupancy_grid_map[row][col] == 1
    