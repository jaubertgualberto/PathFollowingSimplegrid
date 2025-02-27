import heapq
import numpy as np
from typing import List, Tuple


"""
D* Liter Algorithm: https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf
"""


class DStarLite:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], id=0):

        self.grid = grid.copy() 
        self.start = start
        self.goal = goal
        self.g = {}
        self.rhs = {}
        self.U = []
        self.rows, self.cols = grid.shape

        self.Km = 0

        # Initialize
        # print("Initialize ", id)

        # Initialize costs
        for x in range(self.rows):
            for y in range(self.cols):
                self.g[(x, y)] = float('inf')
                self.rhs[(x, y)] = float('inf')

        self.rhs[self.goal] = 0

        key = self._calculate_key(self.goal)
        heapq.heappush(self.U, (key, self.goal))
        
        # Verify initial conditions
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is outside grid bounds")
        if not (0 <= goal[0] < self.rows and 0 <= goal[1] < self.cols):
            raise ValueError(f"Goal position {goal} is outside grid bounds")


    def _calculate_key(self, u: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate the priority key for a given u.

        Args:
            u: Tuple representing the cell (x, y)

        Returns:
            Tuple representing the priority key.
        """ 
        # Return [min(g, rhs) + h(start, u) + km, min(g, rhs)]
        g_rhs = min(self.g[u], self.rhs[u])
        h = self.heuristic(self.start, u)
        # k1, k2
        return (g_rhs + h + self.Km, g_rhs)


    def _update_vertex(self, u: Tuple[int, int] ):
        """
        Update vertex
        """

        if u != self.goal:
            valid_neighbors = [n for n in self._get_neighbors(u) if self._is_valid_coordinates(n)]
            if valid_neighbors:
                self.rhs[u] = min(
                    self.g[neighbor] + self._get_edge_cost(u, neighbor)
                    for neighbor in valid_neighbors
                )
        # if u in self.U:
        self._remove_from_queue(u)

        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self._calculate_key(u), u))

    def _get_edge_cost(self, current: Tuple[int, int], neighbor: Tuple[int, int]) -> float:
        """
        Calculate edge cost with penalties for obstacles
        """
        if not self._is_valid_coordinates(neighbor):
            return float('inf')
            
        base_cost = 1.0
        obstacle_penalty = 50.0
        
        # Apply penalty if moving through obstacle 
        if self.grid[neighbor] == 1:
            # Prevent not reaching goal if blocked by robot
            if neighbor == self.goal:  # Allow reaching goal even if blocked
                return base_cost * 2  # Small penalty for blocked goal
            return base_cost * obstacle_penalty
            
        return base_cost

    def _is_valid_coordinates(self, u: Tuple[int, int]) -> bool:
        """
        Check if coordinates are within grid bounds
        """
        return (0 <= u[0] < self.rows and 
                0 <= u[1] < self.cols)

    def compute_shortest_path(self):
        """
        Compute shortest path
        """
        # print("Compute shortest path")
        
        while self.U:
            k_old, u = heapq.heappop(self.U)
            k_new = self._calculate_key(u)


            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))

            # Inconsistent u (Overconsistent)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for neighbor in self._get_neighbors(u):
                    if self._is_valid_coordinates(neighbor):
                        self._update_vertex(neighbor)

            # Consistent u (Underconsistent)
            else:
                self.g[u] = float('inf')
                self._update_vertex(u)
                for neighbor in self._get_neighbors(u):
                    if self._is_valid_coordinates(neighbor):
                        self._update_vertex(neighbor)
                    

    def reconstruct_path(self) -> List[Tuple[int, int]]:
        """
        Reconstruct path with enhanced error handling
        """
            
        path = []
        current = self.start
        max_path_length = self.rows * self.cols  # Maximum possible path length
        
        while current != self.goal and len(path) < max_path_length:
            path.append(current)
            neighbors = [n for n in self._get_neighbors(current) 
                       if self._is_valid_coordinates(n)]
            
            # No possible cells to move to
            if not neighbors:
                break
                
            # Choose next step based on lowest cost
            current = min(neighbors, 
                        key=lambda n: (self.g[n] + self._get_edge_cost(current, n)))
        
        # Add final position
        path.append(current)  

            
        return path
    
 
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Compute the heuristic (Manhattan distance).

        Args:
            a: Tuple representing the first cell (x, y)
            b: Tuple representing the second cell (x, y)

        Returns:
            Heuristic value.
        """
        # Euclidean distance
        # return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # Minkowski distance (p = 1 -> Manhattan distance, p = 2 -> Euclidean distance)
        p = 1.7
        distance = (abs(a[0] - b[0])**p + abs(a[1] - b[1])**p) ** (1/p)
        return distance

        # Manhattan distance
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])

   
    def _remove_from_queue(self, u: Tuple[int, int]):
        """
        Remove a u from the priority queue if it exists.

        Args:
            u: Tuple representing the cell (x, y)
        """
        self.U = [(k, s) for k, s in self.U if s != u]
        heapq.heapify(self.U)

   
    def _get_neighbors(self, u: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the neighbors of a given cell.

        Args:
            u: Tuple representing the cell (x, y)

        Returns:
            List of neighboring cells.
        """
        x, y = u
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]] # 4-connected
        return neighbors

    def update_grid(self, updated_cells: List[Tuple[int, int]], current_grid: np.ndarray):
        """
        Update grid with new obstacle data and recompute affected vertices.
        """

        for cell in updated_cells:
            self.grid[cell] = current_grid[cell]
            self._update_vertex(cell)


    def _is_valid(self, u: Tuple[int, int]) -> bool:
        """
        Check if a cell is valid (inside grid and not an obstacle), except for the goal.
        
        Args:
            u: Tuple representing the cell (x, y)

        Returns:
            True if valid, False otherwise.
        """
        x, y = u
        # Allow the goal to be valid even if it is in an obstacle
        if u == self.goal:
            return True
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] == 0
    
  
    