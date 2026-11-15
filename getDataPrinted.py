
#import csv
import numpy as np
#import scipy
#from pprint import pprint
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
'''

'''
weight_safety = doEntropy(df_safety, safety_pos).tolist()
print(weight_safety)

weight_economy = doEntropy(df_economy, economy_pos).tolist()
print(weight_economy)

weight_transport = doEntropy(df_transport, transport_pos).tolist()
print(weight_transport)

weight_education = doEntropy(df_education, education_pos).tolist()
print(weight_education)



