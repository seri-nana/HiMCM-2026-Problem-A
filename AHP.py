'''

AHP weight:
[safety, economy, transportation, education]
These weights sum to 1. We will decide them, then check them via 
Consistency Index checking applied to the matrix made from 
the 4 weights we decide.



'''
import pandas as pd
import csv
import numpy as np
import scipy
from pprint import pprint

from EntropyWeighing import doEntropy
from modeling import analyze_topsis


'''
In topicScore, criteria_indices and df_subtopic are redundant. They link 
spreadsheet index - 2 = name of subtopic. 
'''
df = pd.read_csv("HiMCM Cities Data - Sheet1.csv", index_col=0)

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
print(safety_obj_score)


## Here, assuming we have the 4 numbers
# e.g.
# Safety, Economy, Transportation, Education
ahp_weights = [0.40, 0.20, 0.15, 0.25]

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



# Before running, must create the economy, transport, and education scores.
