from common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, boolean indicator

# description:
# In the input you will see a 3x3 grid with red pixels scattered randomly.
# To make the output grid, you should recognize if the input grid has mirror symmetry along the y-axis.
# If the input grid has mirror symmetry along the y-axis, output a 1x1 grid with a blue pixel.
# Otherwise, output a 1x1 grid with an orange pixel.

def main(input_grid):
    # Find the possible symmetry of the input grid.
    best_symmetry = detect_mirror_symmetry(grid=input_grid)

    # Check if the symmetry is symmetric along the y-axis.
    has_y_axis_symmetry = False
    for symmetry in best_symmetry:
        if symmetry.mirror_y is not None:
            has_y_axis_symmetry = True
            break
    
    # Output a 1x1 grid with a blue pixel if the input grid has mirror symmetry along the y-axis.
    if has_y_axis_symmetry:
        output_grid = np.array([[Color.BLUE]])
    else:
        output_grid = np.array([[Color.ORANGE]])
    
    return output_grid

def generate_input():
    n, m = 3, 3
    grid = np.zeros((n, m), dtype=int)
    
    # Randomly generate a 3x3 grid with symmetric pattern or not.
    has_y_axis_symmetry = np.random.choice([True, False])
    symmetry_type = "horizontal" if has_y_axis_symmetry else "not_symmetric"
    density = random.choice([0.3, 0.4, 0.5, 0.6])
    grid = random_sprite(n=3, m=3, density=density, color_palette=[Color.RED], symmetry=symmetry_type)
    
    # If the pattern is not symmetric, scatter some black pixels on the grid to make it not symmetric.
    if not has_y_axis_symmetry:
        # Randomly 40% colored pixels on the grid
        target_density = 0.4
        target_number_of_pixels = int(target_density * m * n)
        for i in range(target_number_of_pixels):
            x = np.random.randint(0, n)
            y = np.random.randint(0, m)
            grid[x, y] = Color.BLACK

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
