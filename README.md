# Drone_Path_Planning

This repository contains a python program with a drone path planning algorithm.

The drone moves through a given grid. Every grid location has a certain value which is achieved if the drone passes the grid location.The goal is to find a path with an as high as possible total value. 

The drone can move 1 grid position every timestamp. The grid also increases the value of the grid positions every timestamp, up to the initial value. 

## Algorithm
The path planning algorithm uses a highest neighbour search with a bias and a gaussian smoothing filter. For each move the drone moves to the highest neighbour. However, a bias has been added to the neighbouring grid scores, in the direction of the highest density of grid scores. In addition to the bias, a gaussian smoothing filter is applied to the grid in order to create a more gradual increase or decrease in grid scores. The bias and the gaussian smoothing are updated at a decreased rate. 

## Running the script
The required pip packages can be found in requirements.txt

The main.py script can be used to run the path planning algorithm with different settings.

The settings are:
* max_t: max algorithm time in ms 
* n_timestamps: max number of timestamps/movements
* n_grid: Grid size [n_grid, n_grid]
* grid_file: if not None it can select one of the default grids from the grid folder
* start_position: start position of the drone
* grid_increment: the value the grid increments up to its maximum every timestamp

The plot options are:
* plot: view the resulting path
* plot_steps: view every move