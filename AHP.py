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

#obj scores
safety_obj_score = objTopicScore(safety_criteria_indices, df_safety, safety_pos)
print(safety_obj_score)



