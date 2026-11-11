import pandas as pd
import numpy as np

# Load and inspect the CSV file for alignment
path = 'C:\\Users\\abrar\\Downloads\\HIMCM\\HiMCM Cities Data - Sheet1.csv'
df = pd.read_csv(path)

# Check row indices carefully in alignment with what was pasted
crime_index = pd.to_numeric(df.iloc[5][1:], errors='coerce')
police_to_civ = pd.to_numeric(df.iloc[8][1:], errors='coerce')
firefighter_to_civ = pd.to_numeric(df.iloc[9][1:], errors='coerce')
medical_to_civ = pd.to_numeric(df.iloc[10][1:], errors='coerce')

# Print extracted data to verify correct procedure
print(f"Crime Index Raw: {crime_index}")
print(f"Police to Civ Raw: {police_to_civ}")
print(f"Firefighter to Civ Raw: {firefighter_to_civ}")
print(f"Medical to Civ Raw: {medical_to_civ}")

# Trace and validate potential nulls, correct logic here
valid_indices = (
    crime_index.notna() & 
    police_to_civ.notna() & 
    firefighter_to_civ.notna() & 
    medical_to_civ.notna()
)

# Capture valid city names and data with the correct indexing
city_names = df.columns[1:][valid_indices].tolist()  # Skip "Cities" header
crime_index = crime_index[valid_indices].tolist()
police_to_civ = police_to_civ[valid_indices].tolist()
firefighter_to_civ = firefighter_to_civ[valid_indices].tolist()
medical_to_civ = medical_to_civ[valid_indices].tolist()

# Check dataset length validation to prevent loss
print(f"Valid Cities: {len(city_names)}, Data Points per Category: {len(crime_index)}, {len(police_to_civ)}, {len(firefighter_to_civ)}, {len(medical_to_civ)}")

# Define normalization with the correct framework for options
def normalize(values, invert=False):
    min_val, max_val = min(values), max(values)
    normalized = [
        (1 - (x - min_val) / (max_val - min_val)) if invert else ((x - min_val) / (max_val - min_val))
        for x in values
    ]
    return normalized

# Normalize for criteria and invert crime index
crime_index_norm = normalize(crime_index, invert=True)
police_to_civ_norm = normalize(police_to_civ)
firefighter_to_civ_norm = normalize(firefighter_to_civ)
medical_to_civ_norm = normalize(medical_to_civ)

# Weights equalized
weights = np.array([1, 1, 1, 1])

# Calculate weighted normalized values
weighted_crime_index = np.array(crime_index_norm) * weights[0]
weighted_police_to_civ = np.array(police_to_civ_norm) * weights[1]
weighted_firefighter_to_civ = np.array(firefighter_to_civ_norm) * weights[2]
weighted_medical_to_civ = np.array(medical_to_civ_norm) * weights[3]

# Ideal and negative-ideal solutions
ideal_solution = np.array([
    max(weighted_crime_index), 
    max(weighted_police_to_civ),
    max(weighted_firefighter_to_civ), 
    max(weighted_medical_to_civ)
])
negative_ideal_solution = np.array([
    min(weighted_crime_index), 
    min(weighted_police_to_civ), 
    min(weighted_firefighter_to_civ), 
    min(weighted_medical_to_civ)
])

# Euclidean distance calculation improvements
def distance_to_ideal(point, ideal):
    return np.sqrt(np.sum((point - ideal) ** 2))

dist_to_ideal = [
    distance_to_ideal([c, p, f, m], ideal_solution)
    for c, p, f, m in zip(weighted_crime_index, weighted_police_to_civ, weighted_firefighter_to_civ, weighted_medical_to_civ)
]
dist_to_negative_ideal = [
    distance_to_ideal([c, p, f, m], negative_ideal_solution) 
    for c, p, f, m in zip(weighted_crime_index, weighted_police_to_civ, weighted_firefighter_to_civ, weighted_medical_to_civ)
]

# Closeness calculation refinement
relative_closeness = [
    n / (i + n) if (i + n) > 0 else 0 for i, n in zip(dist_to_ideal, dist_to_negative_ideal)
]

# Final city rankings
topsis_results = list(zip(city_names, relative_closeness))
topsis_results.sort(key=lambda x: x[1], reverse=True)

# Print the comprehensive rankings with all available cities
print("TOPSIS Ranking for Safety Across All Captured Cities:")
for rank, (city, score) in enumerate(topsis_results, 1):
    print(f"{rank}. {city}: {score:.4f}")