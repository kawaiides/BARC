from common import *

import numpy as np
from typing import *

# concepts:
# flip

# description:
# In the input you will see a grid as part of the pattern.
# To make the output, 
# 1. flip the grid horizontally with y-axis on the left side of the grid, make it twice larger.
# 2. flip it down with x-axis on the bottom side of the grid.
# 3. Concate the flipped grid in step 2 to the top and bottom of the grid in step 1.
# In total the grid is 6 times larger than the input grid.

def main(input_grid):
    # Create the output grid 6 times larger than the input grid
    n, m = input_grid.shape
    output_grid = np.zeros((n * 2, m * 3), dtype=int)

    # Step 1: Flip the grid horizontally with y-axis on the left side of the grid, concate it to the left.
    # Place it in the middle of the output grid
    flip_grid = np.flipud(input_grid)
    blit_sprite(grid=output_grid, sprite=flip_grid, x=0, y=m)
    blit_sprite(grid=output_grid, sprite=input_grid, x=n, y=m)

    # Step 2: Flip it down with x-axis on the bottom side of the grid, concate it to the bottom and top.
    original_object = output_grid[:, m :2 * m]
    filp_down_object = np.fliplr(original_object)
    blit_sprite(grid=output_grid, sprite=filp_down_object, x=0, y=m * 2)
    blit_sprite(grid=output_grid, sprite=filp_down_object, x=0, y= 0)
    return output_grid

def generate_input():
    # Generate grid of size n x m
    n, m = np.random.randint(2, 6), np.random.randint(2, 6)
    grid = np.zeros((n, m), dtype=int)

    # Randomly choose 1 color
    color = np.random.choice(list(Color.NOT_BLACK))

    # Randomly choose the density of the color
    density = np.random.randint(2, n * m) / (n * m)

    # Randomly scatter the color in the grid
    grid = random_scatter_points(grid=grid, color=color, density=density)

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
