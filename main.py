import os
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Settings:
    max_t: int = 1000
    n_timestamps: int = 100
    start_position: tuple[int, int] = (0,0)
    n_grid: int = 10
    grid_file: int = None

    grid_increment: float = 0.0
    bias_val: float = 0.1
    sigma: float = 0.1
    binning_factor: int = 1
    grid_interval: int = 100
    
    plot_steps: bool = False
    plot: bool = False


# Grid
def load_grid(settings):
    if settings.grid_file:
        abs_path = os.path.dirname(os.path.abspath(__file__))
        print(abs_path)
        grid_file = abs_path + "\\grids\\" + settings.grid_file
        grid = np.loadtxt(grid_file, delimiter=' ')
        return grid
    else:
        grid_max = max(settings.n_grid // 10, 1)
        grid = np.random.randint(grid_max + 1, size=(settings.n_grid, settings.n_grid))
        return grid
    

def update_grid(cur_grid, start_grid, cur_pos, inc):
    cur_grid = cur_grid + inc
    cur_grid = np.minimum(cur_grid, start_grid)
    cur_grid[cur_pos[1], cur_pos[0]] = 0
    return cur_grid


def bin_grid(grid, settings):
    new_shape = int(settings.n_grid/settings.binning_factor)
    binned_grid = grid.reshape(new_shape, settings.binning_factor,
                               new_shape, settings.binning_factor).sum(3).sum(1)
    return binned_grid


def smoothen_grid(grid, sigma):
    smooth_grid = gaussian_filter(grid, [sigma, sigma], mode='constant')
    smooth_grid[smooth_grid < 0.001] = 0
    return smooth_grid


# Drone
def move_drone(cur_pos, move, n):
    cur_pos = cur_pos + move
    # make sure the drone doesnt leave the grid
    cur_pos = np.maximum(cur_pos, [0,0])
    cur_pos = np.minimum(cur_pos, [n-1, n-1])
    return cur_pos.astype(int)


# path 
def choose_path_random(settings):
    start_time = time.perf_counter()
    path = np.zeros([settings.n_timestamps,2])
    for i in range(settings.n_timestamps):
        path[i] = np.random.randint(3, size=(2,)) - 1

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return path, elapsed_time * 1000


def get_valid_moves(pos, n):
    possible_moves =  [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    valid_moves = []
    for dx, dy in possible_moves:
        temp_pos = pos + np.array([dx,dy])
        if (0 <= temp_pos[0] < n) & (0 <= temp_pos[1] < n):
            valid_moves.append((dx, dy))
    return valid_moves


def calculate_bias(position, grid, settings):
    binned_grid = bin_grid(grid, settings)

    bias_val = settings.bias_val
    max_loc1 = np.unravel_index(np.argmax(binned_grid), binned_grid.shape)
    max_loc = np.array([max_loc1[1] + 0.5,
                        max_loc1[0] + 0.5]) * settings.binning_factor
    diff = max_loc - position
    length = np.sqrt(diff[0]**2 + diff[1]**2)

    direction = (round(diff[0] / length), round(diff[1] / length)) if length != 0 else 0

    # If the drone is in the bin, there is no need for a bias
    if (abs(diff[0]) <= settings.binning_factor/2) & (abs(diff[0]) <= settings.binning_factor/2):
        bias_val = 0

    return (direction, bias_val)


def choose_path_max_neighbour( grid, settings):
    start_time = time.perf_counter()
    starting_grid = grid.copy()
    real_grid = grid.copy()
    percieved_grid  = smoothen_grid(grid.copy(), settings.sigma)

    bias = calculate_bias(settings.start_position, percieved_grid, settings)

    positions = np.zeros([settings.n_timestamps+1,2], dtype= int)
    positions[0] = settings.start_position
    path = np.zeros([settings.n_timestamps,2])

    for i in range(settings.n_timestamps):
        max_val = 0
        current_pos = positions[i].astype(int)
        valid_moves = get_valid_moves(current_pos, settings.n_grid)

        for dx, dy in valid_moves:
            temp_pos = move_drone(current_pos, [dx, dy], settings.n_grid)
            val = percieved_grid[temp_pos[1], temp_pos[0]]
            if (dx, dy) == bias[0]:
                val += bias[1]
            if val > max_val:
                max_val = val
                move = np.array([dx, dy])

        # If there is no better move, move randomly
        if max_val == 0:
            move = valid_moves[np.random.randint(len(valid_moves))]

        path[i] = move
        positions[i+1] = move_drone(current_pos, [move[0], move[1]], settings.n_grid)
        real_grid = update_grid(real_grid, starting_grid, positions[i+1], settings.grid_increment)
        percieved_grid[current_pos[1] + move[1], current_pos[0] + move[0]] = 0

        # Every interval reset the bias
        if i%settings.grid_interval == 0:
            percieved_grid  = smoothen_grid(real_grid.copy(), settings.sigma)
            bias = calculate_bias(current_pos, percieved_grid, settings)
            # print(bias)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return path, elapsed_time * 1000, grid


def execute_path(settings):
    starting_grid = load_grid(settings)

    # Run the algorithm for the path
    path, time_ms, last_percieved_grid = choose_path_max_neighbour( starting_grid.copy(), settings)
    
    if time_ms > settings.max_t:
        print("Algorithm failed to compute in time") 

    # Execute path
    start_score = starting_grid[settings.start_position[0], settings.start_position[1]]
    current_grid = update_grid(starting_grid, starting_grid, settings.start_position, 0)

    # Initiate plot
    if settings.plot_steps & settings.plot:
        fig, ax = plt.subplots()
        im = ax.imshow(starting_grid, cmap="viridis")
        drone_plot, = ax.plot(settings.start_position[0], settings.start_position[1], 'ko')
        fig.colorbar(im)
        fig.show()

    # Initialize loop
    travelled = np.zeros([settings.n_timestamps + 1, 2])
    travelled[0] = settings.start_position
    current_position = settings.start_position
    total_score = start_score

    # Loop through the moves
    for i, move in enumerate(path):
        current_position = move_drone(current_position, move, settings.n_grid)
        travelled[i+1] = current_position
        total_score += current_grid[current_position[1], current_position[0]]
        current_grid = update_grid(current_grid, starting_grid, current_position, settings.grid_increment)

        if settings.plot_steps & settings.plot:
            drone_plot.set_data(current_position[0], current_position[1])
            im.set_data(smoothen_grid(current_grid,settings.sigma))
            plt.pause(0.1)
            plt.draw()

    print(f"Algorithm time : {time_ms//1} ms")
    print(f"Algorithm score: {total_score}")
    print(f"Total grid     : {sum(sum(starting_grid))}")

    # Final plot
    if (not settings.plot_steps) & (settings.plot):
        fig, ax = plt.subplots()
        im = ax.imshow(starting_grid, cmap="viridis")
        drone_plot, = ax.plot(travelled[:,0], travelled[:,1])
        fig.colorbar(im)
        plt.show()

    return total_score


if __name__ == "__main__":
    
    # Settings
    my_settings = Settings(
        ## Inputs
        max_t = 1000,
        n_timestamps=100,
        n_grid=20,
        grid_file='20.txt', # None
        start_position= (0,0),
        grid_increment=0.01,

        ## Plot options
        plot = True,
        plot_steps= False
    )

    my_settings.sigma = 0.3
    my_settings.bias_val = my_settings.n_grid/20
    my_settings.binning_factor = my_settings.n_grid//10
    my_settings.grid_interval = my_settings.n_grid//2

    execute_path(my_settings)














