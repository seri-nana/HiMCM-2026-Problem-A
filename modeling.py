import pandas as pd
import numpy as np

# Load and preprocess data from the CSV
def load_data(file_path, criteria_indices):
    df = pd.read_csv(file_path)
    data = df.iloc[criteria_indices].T[1:].astype(float).values
    return data, df.columns[1:].tolist()

# Generalized TOPSIS function
def topsis(data, weights, maximize_criteria=None):
    if maximize_criteria is None:
        maximize_criteria = [True] * data.shape[1]

    def normalize(values, maximize):
        min_val, max_val = min(values), max(values)
        return [(v - min_val) / (max_val - min_val) if maximize else (max_val - v) / (max_val - min_val) for v in values]

    norm_data = np.array([normalize(data[:, i], maximize_criteria[i]) for i in range(data.shape[1])]).T
    weighted_data = norm_data * weights

    ideal_solution = np.max(weighted_data, axis=0)
    negative_ideal_solution = np.min(weighted_data, axis=0)

    dist_to_ideal = np.sqrt(np.sum((weighted_data - ideal_solution) ** 2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_data - negative_ideal_solution) ** 2, axis=1))
    relative_closeness = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

    return relative_closeness

# Example Analysis function
def analyze_topsis(file_path, criteria_indices, weights, maximize_criteria):
    data, city_names = load_data(file_path, criteria_indices)
    rel_closeness = topsis(data, weights, maximize_criteria)
    
    ranked_cities = sorted(zip(city_names, rel_closeness), key=lambda x: x[1], reverse=True)
    
    print("TOPSIS Ranking for Safety:")
    for rank, (city, score) in enumerate(ranked_cities, 1):
        print(f"{rank}. {city}: {score:.4f}")

"""
# Running the example
file_path = 'HiMCM Cities Data - Sheet1.csv'
criteria_indices = [5, 8, 9, 10]  # Example indices for safety-related metrics
weights = [1, 1, 1, 1]  # Hypothetical uniform weights for demonstration
maximize_criteria = [False, True, True, True]  # Minimize crime, maximize others

analyze_topsis(file_path, criteria_indices, weights, maximize_criteria)
"""