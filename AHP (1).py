'''

AHP weight:
[safety, economy, transportation, education]
These weights sum to 1. We will decide them, then check them via 
Consistency Index checking applied to the matrix made from 
the 4 weights we decide.



'''

import csv
import numpy as np
import scipy
from pprint import pprint
import pandas as pd

from EntropyWeighing import doEntropy
from modeling import analyze_topsis


'''
In topicScore, criteria_indices and df_subtopic are redundant. They link 
spreadsheet index - 2 = name of subtopic. 
'''
df = pd.read_csv("HiMCM Cities Data - Sheet1.csv", index_col=0)
df.index = df.index.str.strip()

df_safety = df.loc[
    [
        "Numbeo Perceived Crime Index",
        "Police to civilian ratio",
        "Firefighter to civilian ratio",
        "Medical to civilian ratio"
    ]
].T

safety_criteria_indices = [5, 8, 9, 10]
safety_pos = [False, True, True, True]

df_economy = df.loc[
    [
        "Income : Cost of Living ratio", 
        "Unemployment rate"
    ]
].T

economy_criteria_indices = [25, 26]
economy_pos = [True, False]

df_transport = df.loc[
    [
        "Do kids have to pay? (0 pay completely; 1 for discounted or free)",
        "Transport coverage (train and bus, percent of residents within 500m walking distance to bus and 1km walking from rail)"
    ]
].T

transport_criteria_indices = [28, 33]
transport_pos = [True, True]

df_education = df.loc[
    [
        "High School Graduation Rates",
        "Youth Unemployment Rates (ages 15-24), kids who try to find work but fail"
    ]
].T

education_criteria_indices = [37, 38]
education_pos = [True, False]

#Write the other 3 dfs!
#make the other 3 indices too.

def objTopicScore(criteria_indices, df_subtopic, maximize_criteria):
    # weights are weights of subtopics within a topic.
    weights = doEntropy(df_subtopic, maximize_criteria).tolist()

    obj_topic_scores = analyze_topsis(criteria_indices, weights, maximize_criteria)

    return obj_topic_scores

#obj scores. Need to do for economy, transportation, and education.
print("here!!")
safety_obj_score = objTopicScore(safety_criteria_indices, df_safety, safety_pos)
#print(safety_obj_score)
economy_obj_score = objTopicScore(economy_criteria_indices, df_economy, economy_pos)
transport_obj_score = objTopicScore(transport_criteria_indices, df_transport, transport_pos)
education_obj_score = objTopicScore(education_criteria_indices, df_education, education_pos)

## Here, assuming we have the 4 numbers
# e.g.
# Safety, Economy, Transportation, Education
#ahp_weights = [0.40, 0.20, 0.15, 0.25]
ahp_weights = [0.20, 0.20, 0.20, 0.20]

# Multiply safety ahp weight to safety score to get subjectively weighted score
# Do for other 3
# Add them. to get final score for city.

#CHECK THIS LATER!!!!!!!!

def calculate_final_scores(
    safety_scores,
    economy_scores,
    transportation_scores,
    education_scores,
    ahp_weights
):
    # Convert each list of tuples into dictionaries
    safety = dict(safety_scores)
    economy = dict(economy_scores)
    transportation = dict(transportation_scores)
    education = dict(education_scores)

    # Get city names
    cities = safety.keys()

    final_scores = {}

    for city in cities:
        score = (
            ahp_weights[0] * safety[city] +
            ahp_weights[1] * economy[city] +
            ahp_weights[2] * transportation[city] +
            ahp_weights[3] * education[city]
        )

        final_scores[city] = score

    # Rank cities from highest to lowest
    ranked_cities = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_cities

ranked_cities = calculate_final_scores(safety_obj_score, economy_obj_score, transport_obj_score, education_obj_score, ahp_weights)

#print(ranked_cities)

print("\nFINAL CITY RANKINGS")
print("-" * 45)
print(f"{'Rank':<8}{'City':<25}{'Final Score':>12}")
print("-" * 45)

for rank, (city, score) in enumerate(ranked_cities, 1):
    print(f"{rank:<8}{city:<25}{score:>12.4f}")

print("-" * 45)


# Before running, must create the economy, transport, and education scores.
