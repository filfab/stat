import numpy as np
from numpy.random import Generator, MT19937
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

rng = Generator(MT19937())

n_points = 500
dimensions = [1, 2, 5, 10, 50, 100, 500]
results_r_d = []

def generate_uniform_points(n_points, dimension):
    return rng.uniform(0, 1, (n_points, dimension))

def calculate_euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b)**2))

def compute_distance_metrics(points):
    n = len(points)
    d_min_list = []
    d_max_list = []
    
    for i in range(n):
        distances_for_point = []
        
        for j in range(n):
            if i == j:
                continue 
            
            dist = calculate_euclidean_distance(points[i], points[j])
            distances_for_point.append(dist)
            
        d_min_list.append(min(distances_for_point))
        d_max_list.append(max(distances_for_point))

    return np.array(d_min_list), np.array(d_max_list)

def calculate_r_d(d_min_array, d_max_array):
    r_d = np.mean((d_max_array - d_min_array) / d_min_array)
    return r_d

def draw_plot(dimensions, r_d_results):
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, r_d_results, marker='o', linestyle='-', color='b')
    plt.xlabel('Dimension (d)')
    plt.ylabel('R(d) = (dmax - dmin) / dmin')
    plt.title('Wykres R(d)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def main():
    for d in dimensions:
        points = generate_uniform_points(n_points, d)
        d_min_arr, d_max_arr = compute_distance_metrics(points)
        r_val = calculate_r_d(d_min_arr, d_max_arr)
        results_r_d.append(r_val)
        
        print(f"R(d = {d}) = {r_val:.4f}")
    draw_plot(dimensions, results_r_d)
    
if __name__ == "__main__":
    main()