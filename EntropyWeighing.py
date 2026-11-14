
# Downloading Google Sheets
    # pip install pandas numpy scipy
    # in terminal
import pandas as pd
import csv
import numpy as np
import scipy
from pprint import pprint

#Read Google Sheets
df = pd.read_csv("HiMCM Cities Data - Sheet1.csv", index_col=0)
#print(df.head())
if __name__ == "__main__":
    #Look at safety data only
    df_safety = df.loc[
        [
            "Numbeo Perceived Crime Index",
            "Police to civilian ratio",
            "Firefighter to civilian ratio",
            "Medical to civilian ratio"
        ]
    ].T

    print(df_safety.head())


    #list of cities
    cities_list = df.columns.tolist()
    print("cities list:")
    print(cities_list)

    # safety info
    crime_index = pd.to_numeric(df_safety["Numbeo Perceived Crime Index"]).tolist()
    print("crime index:")
    print(crime_index)

    police_ratio = pd.to_numeric(df_safety["Police to civilian ratio"]).tolist()
    print("Police to civilian ratio: ")
    print(police_ratio)

    firefighter_ratio = pd.to_numeric(df_safety["Firefighter to civilian ratio"]).tolist()
    print("Firefighter to civilian ratio: ")
    print(firefighter_ratio)

    medical_ratio = pd.to_numeric(df_safety["Medical to civilian ratio"]).tolist()
    print("Medical to civilian ratio")
    print(medical_ratio)

def normalize(values, positive=True):
    '''
    parameter values: is a list. e.g. medical_ratio
    parameter positive: = True if having a higher value is better
                False if having a lower value is better. (e.g. crime)



    '''
    max_val = max(values)
    min_val = min(values)

    if max_val == min_val:
        return [0] * len(values)

    elif positive:
        return [(x-min_val) / (max_val - min_val) for x in values]

    else:
        return [(max_val - x) / (max_val - min_val) for x in values]
    #end of def normalize

if __name__ == "__main__":
    crime_index_norm = normalize(crime_index, positive=False)
    print("Normalized Crime Index")
    print(str(crime_index_norm))

    police_ratio_norm = normalize(police_ratio, positive=True)
    print("Normalized Police Ratio")
    print(str(police_ratio_norm))

    firefighter_ratio_norm = normalize(firefighter_ratio, positive=True)
    print("Normalized Firefighter Ratio")
    print(str(firefighter_ratio_norm))

    medical_ratio_norm = normalize(medical_ratio, positive=True)
    print("Normalized Medical Ratio")
    print(str(medical_ratio_norm))



'''
def entropy_weight_subtopic()
    Xmin = 0
    Xmax = 0
    
'''
if __name__ == "__main__":
    #make normalized safety matrix.
    normalized_safety_matrix = pd.DataFrame({
        "Crime Index": crime_index_norm,
        "Police Ratio": police_ratio_norm,
        "Firefighter Ratio": firefighter_ratio_norm,
        "Medical Ratio": medical_ratio_norm
    }, index=cities_list)

    print(normalized_safety_matrix)


#proportional contribution

def prop_contrib(normalized_subtopic_list):
    #takes list parameter e.g. crime_index_norm
    # divides each element in list by sum of list
    total = sum(normalized_subtopic_list)

    if total==0:
        return [0] * len(normalized_subtopic_list)

    return [x/total for x in normalized_subtopic_list]

# Calculate proportional contributions
if __name__ == "__main__":
    crime_prop = prop_contrib(crime_index_norm)
    print("Proportional Crime Index:")
    print(crime_prop)

    police_prop = prop_contrib(police_ratio_norm)
    print("Proportional Police Ratio:")
    print(police_prop)
    #print("sum police_prop")
    #print(sum(police_prop))  was 1. check.

    firefighter_prop = prop_contrib(firefighter_ratio_norm)
    print("Proportional Firefighter Ratio:")
    print(firefighter_prop)

    medical_prop = prop_contrib(medical_ratio_norm)
    print("Proportional Medical Ratio:")
    print(medical_prop)

    proportional_safety_matrix = pd.DataFrame({
        "Crime Index": crime_prop,
        "Police Ratio": police_prop,
        "Firefighter Ratio": firefighter_prop,
        "Medical Ratio": medical_prop
    }, index=cities_list)

    print(proportional_safety_matrix)


# Calculate informational entropy (e j)

def calculate_entropy(prop_list):
    n = len(prop_list) #n is number of cities. should be 15

    #np.log(p) calculates ln(p)

    entropy_sum = sum(
        p * np.log(p) for p in prop_list if p > 0
    )

    return -entropy_sum / np.log(n)

if __name__ == "__main__":
    # Calculate entropy for each safety subtopic
    # here is entropies for safety subtopics.
    crime_entropy = calculate_entropy(crime_prop)
    police_entropy = calculate_entropy(police_prop)
    firefighter_entropy = calculate_entropy(firefighter_prop)
    medical_entropy = calculate_entropy(medical_prop)

    print("Information Entropy:")
    print("Crime:", crime_entropy)
    print("Police:", police_entropy)
    print("Firefighter:", firefighter_entropy)
    print("Medical:", medical_entropy)


def doEntropy(df_subtopic, positive_list):
    '''
    df_subtopic: dataframe with cities as rows, subtopics as columns. 
        encapsulates data for one topic.
    positive_list:
        True = higher is better
        False = lower is better
    '''
    
    # Normalize each subtopic
    normalized_data = {}

    for j, column in enumerate(df_subtopic.columns):
        values = pd.to_numeric(df_subtopic[column]).tolist()

        normalized_data[column] = normalize(
            values,
            positive=positive_list[j]
        )

    #make a normalized matrix in case need to return.
    normalized_matrix = pd.DataFrame(
        normalized_data,
        index=df_subtopic.index
    )

    # Proportional contributions
    proportional_data = {}

    for column in normalized_matrix.columns:
        proportional_data[column] = prop_contrib(
            normalized_matrix[column].tolist()
        )

    proportional_matrix = pd.DataFrame(
        proportional_data,
        index=df_subtopic.index
    )

    # Entropy for each subtopic
    entropy = {}

    for column in proportional_matrix.columns:
        entropy[column] = calculate_entropy(
            proportional_matrix[column].tolist()
        )

    entropy = pd.Series(entropy)

    # Calculate diversity coefficients
    # entropy: 0 is high contrast, 1 is low contrast.
    # diversity: 0 is low contrast, 1 is high contrast
    diversity = 1 - entropy # diversity is a panda

    # Calculate entropy weights
    # Normalize diversity so it sums to one, these normalized values are weights
    if diversity.sum() == 0:
        weights = diversity * 0 + 1 / len(diversity)
    else:
        weights = diversity / diversity.sum()

    # return normalized_matrix, proportional_matrix, entropy, diversity, weights

    return weights

if __name__ == "__main__":
    print("HERE!")
    safety_entropy = doEntropy(df_safety, [False, True, True, True])
    print("HERE!!!")
    print(safety_entropy)
'''
entropy is a dictionary. 
{subtopic1: entropy1,
 subtopic2: entropy2, 
 subtopic3: entropy3,
 subtopic4: entropy4}

'''

#take out print statements b/c they show in import.