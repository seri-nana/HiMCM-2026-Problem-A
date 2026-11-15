import numpy as np
import pandas as pd



'''
Does the highest ranked city remain best choice when it changes up stuff?
Probability of ranking first
Deviation from weights that we picked
% of times it ranked first.
save each one for 5%, 10%, 15%

'''

'''

AHP weight:
[safety, economy, transportation, education]
These weights sum to 1. We will decide them, then check them via 
Consistency Index checking applied to the matrix made from 
the 4 weights we decide.



'''
'''
import csv
import scipy
from pprint import pprint
'''
from EntropyWeighing import doEntropy
from modeling import analyze_topsis
from AHP import objTopicScore
from AHP import calculate_final_scores



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
#ahp_weights = [0.25, 0.25, 0.25, 0.25]
#ahp_weights = [0.35, 0.28, 0.25, 0.12]
# Safety, Economy, Transportation, Education
ahp_weights = [0.4710, 0.2840, 0.0736, 0.1715]

# Multiply safety ahp weight to safety score to get subjectively weighted score
# Do for other 3
# Add them. to get final score for city.

#CHECK THIS LATER!!!!!!!!

original_ranked_cities = calculate_final_scores(safety_obj_score, economy_obj_score, transport_obj_score, education_obj_score, ahp_weights)

#print(ranked_cities)

#================================

import numpy as np
import pandas as pd



def monte_carlo_ahp(
    safety_score,
    economy_score,
    transport_score,
    education_score,
    original_weights,
    variation,
    simulations
):

    rng = np.random.default_rng()

    original_weightsArr = np.array(original_weights, dtype=float)
    original_weightsArr /= original_weightsArr.sum()

    all_city_scores = []

    # Get city names from one calculation
    original_result = calculate_final_scores(
        safety_score,
        economy_score,
        transport_score,
        education_score,
        original_weightsArr.tolist()
    )

    city_names = [item[0] for item in original_result]

    for i in range(simulations):

        random_factors = rng.uniform(
            1 - variation,
            1 + variation,
            size=4
        )

        new_weights = original_weightsArr * random_factors
        new_weights /= new_weights.sum()

        new_scores = calculate_final_scores(
            safety_score,
            economy_score,
            transport_score,
            education_score,
            new_weights.tolist()
        )

        # Extract numerical scores only
        score_values = [item[1] for item in new_scores]

        all_city_scores.append(score_values)

    # Create DataFrame with city names as columns
    scores_df = pd.DataFrame(
        all_city_scores,
        columns=city_names
    )

    return scores_df


# -----------------------------------------
# MONTE CARLO PARAMETERS
# -----------------------------------------





NUM_SIMULATIONS = 1000


# -----------------------------------------
# RUN 5% SIMULATION
# -----------------------------------------

variation = 0.05

scores_5 = monte_carlo_ahp(
    safety_obj_score,
    economy_obj_score,
    transport_obj_score,
    education_obj_score,
    ahp_weights,
    variation,
    NUM_SIMULATIONS
)

scores_5.to_csv(
    "monte_carlo_5percent_city_scores.csv",
    index=False
)

print("5% simulation completed and saved!")


# -----------------------------------------
# RUN 10% SIMULATION
# -----------------------------------------

variation = 0.10

scores_10 = monte_carlo_ahp(
    safety_obj_score,
    economy_obj_score,
    transport_obj_score,
    education_obj_score,
    ahp_weights,
    variation,
    NUM_SIMULATIONS
)

scores_10.to_csv(
    "monte_carlo_10percent_city_scores.csv",
    index=False
)

print("10% simulation completed and saved!")


# -----------------------------------------
# RUN 15% SIMULATION
# -----------------------------------------

variation = 0.15

scores_15 = monte_carlo_ahp(
    safety_obj_score,
    economy_obj_score,
    transport_obj_score,
    education_obj_score,
    ahp_weights,
    variation,
    NUM_SIMULATIONS
)

scores_15.to_csv(
    "monte_carlo_15percent_city_scores.csv",
    index=False
)

print("15% simulation completed and saved!")


print("\nAll Monte Carlo simulations completed!")


# -----------------------------------------
# ORIGINAL CITY RANKINGS
# -----------------------------------------

original_ranks = {
    "Bern, Switzerland": 1,
    "Reykjavik, Iceland": 2,
    "Zurich, Switzerland": 3,
    "Tokyo, Japan": 4,
    "Singapore, Singapore": 5,
    "Seattle, USA": 6,
    "Ottawa, Canada": 7,
    "Seoul, South Korea": 8,
    "Munich, Germany": 9,
    "Paris, France": 10,
    "Taipei, Taiwan": 11,
    "Hong Kong, SAR of China": 12,
    "Melbourne CBD, Australia": 13,
    "Boston, USA": 14,
    "Vancouver, Canada": 15,
    "San Jose, USA": 16,
    "New York, USA": 17,
    "London, UK": 18,
    "Shanghai, China": 19
}


original_scores = {
    "Bern, Switzerland": 0.6875,
    "Reykjavik, Iceland": 0.5163,
    "Zurich, Switzerland": 0.4254,
    "Tokyo, Japan": 0.4185,
    "Singapore, Singapore": 0.4086,
    "Seattle, USA": 0.4001,
    "Ottawa, Canada": 0.3894,
    "Seoul, South Korea": 0.3787,
    "Munich, Germany": 0.3775,
    "Paris, France": 0.3769,
    "Taipei, Taiwan": 0.3670,
    "Hong Kong, SAR of China": 0.3649,
    "Melbourne CBD, Australia": 0.3529,
    "Boston, USA": 0.2732,
    "Vancouver, Canada": 0.2492,
    "San Jose, USA": 0.2321,
    "New York, USA": 0.2313,
    "London, UK": 0.2300,
    "Shanghai, China": 0.1977
}



# -----------------------------------------
# SUMMARY FUNCTION
# -----------------------------------------

def create_summary(scores_df, original_ranks):

    assert set(scores_df.columns) == set(original_ranks.keys()), \
        "City names do not match!"

    # Mean simulated score for each city
    mean_scores = scores_df.mean()

    # Standard deviation for each city's score
    std_scores = scores_df.std()
    
    # Difference between simulated mean and original score
    mean_deviation = mean_scores - pd.Series(original_scores)

    # Find the city ranked first in each simulation
    first_place = scores_df.idxmax(axis=1)

    # Calculate percentage of times each city ranked first
    first_place_percent = (
        first_place.value_counts(normalize=True) * 100
    )

    # Create summary table
    summary = pd.DataFrame({
        "Original Rank": pd.Series(original_ranks),
        "Mean Simulated Score": mean_scores,
        "Score Standard Deviation": std_scores,
        "Mean Deviation": mean_deviation,
        "% Ranked #1": first_place_percent
    })

    # Sort by original ranking
    summary = summary.sort_values("Original Rank")

    return summary


# -----------------------------------------
# GENERATE THREE SUMMARY TABLES
# -----------------------------------------

summary_5 = create_summary(scores_5, original_ranks)
summary_10 = create_summary(scores_10, original_ranks)
summary_15 = create_summary(scores_15, original_ranks)


# -----------------------------------------
# SAVE TO CSV
# -----------------------------------------

summary_5.to_csv("summary_5percent.csv")
summary_10.to_csv("summary_10percent.csv")
summary_15.to_csv("summary_15percent.csv")


# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------

print("\n========== 5% VARIATION ==========")
print(summary_5.round(4).to_string())

print("\n========== 10% VARIATION ==========")
print(summary_10.round(4).to_string())

print("\n========== 15% VARIATION ==========")
print(summary_15.round(4).to_string())




