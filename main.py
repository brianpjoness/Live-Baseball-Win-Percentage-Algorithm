
import pandas as pd
import numpy as np
from pybaseball import statcast
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = statcast(start_dt = "2023-04-01", end_dt="2023-04-7")


keep_cols = [
    'game_pk', 'game_date', 'inning', 'inning_topbot',
    'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score', 'home_team', 'away_team',
     'events', 'description', "balls", "strikes", "pitch_number", "home_score", "away_score"
]

df = df[keep_cols]
df["score_diff"] = df["bat_score"] - df["fld_score"]
df["is_top_inning"] = (df["inning_topbot"] == "Top").astype(int)
df["Outs"] = df["outs_when_up"]
df["runners_on"] = df[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)  # since it was na values before we make it to notna and then sum it
df["batting_team"] = df.apply(
    lambda row: row["away_team"] if row['is_top_inning'] == 1 else row["home_team"], axis=1
)
df["pitching_team"] = df.apply(
    lambda row: row["home_team"] if row["is_top_inning"] == 1 else row["away_team"], axis=1
)
df["batting_team_is_home"] = (df["home_team"] == df["batting_team"]).astype(int)

final_plays = df.sort_values(['game_pk', 'inning', 'outs_when_up', 'pitch_number']).groupby('game_pk').tail(1)  # tail 1 returns the last row
# this gets the very last play of a game

final_plays["home_win"] = final_plays["home_score"]  > final_plays["away_score"]
final_plays['away_win'] = final_plays['home_score'] < final_plays['away_score']
final_plays['winner'] = np.where(final_plays['home_win'], final_plays['home_team'], final_plays['away_team'])  # where(condition, value if true, value if false)

df = df.merge(final_plays[['game_pk', 'winner']], on='game_pk', how='left')

# Add a binary win column for the batting_team on each play
df['is_batting_team_winner'] = (df['batting_team'] == df['winner']).astype(int)

df = df.drop(columns="winner")


def prepare_modeling_data(df):
    feature_cols = [
        'score_diff',           # Current score difference (batting team perspective)
        'is_top_inning',        # Whether it's top of inning
        'Outs',                 # Number of outs
        'runners_on',           # Number of runners on base
        'batting_team_is_home', # Whether batting team is home
        'inning',               # What inning it is
        'balls',                # Current ball count
        'strikes'               # Current strike count
    ]
    target = 'is_batting_team_winner'
 # return modeling df, feature cols, target

    modeling_df = df[feature_cols + [target, "game_pk", "game_date"]].dropna()

    print("Dataframe Shape")
    print(modeling_df.shape)
    print("Target Distribution: ")
    print(modeling_df.value_counts(normalize=True))
    return modeling_df, feature_cols, target

modeling_df, feature_cols, target = prepare_modeling_data(df)

def temporal_train_test_split(df, test_size =0.2):
    # important to split data based on the game to avoid data leakage
    # go by date
    df = df.sort_values(["game_date", "game_pk"])
    split_index = int(len(df) * (1 - test_size))
    test_df = df[split_index:]
    train_df = df[:split_index]

    return train_df, test_df

# return train and test df
print(temporal_train_test_split(modeling_df))

def train_baseline_models(train_df, test_df, feature_names):
    # train logistic regression and random forest models
    # make sure to do scaler for logistic regression
    # do model results and feature importances
    hi = 1



# return models and results, print importanec




def evaluate_models(results):
    # compare model performance
    # make dataframe of the comparison with results
    # you can also plot it if you want

    # return the comparison
    hi = 1

def analyze_predictions_by_game_situation(df, predictions, target):
    # analyze how model performs in differnet game situations
    # with an analysis_df you can have a column for predictions and correct (.0.5)

    # you can do performance of the model based on inning

    # you can make a score diff bucket based off of how accurate the model is depending on different score difference ranges

    # same thing for runners on base
    hi = 1


def run_modeling_pipeline(df):
    # finsih modeling pipeline
    hi = 1
# prepare data with corresponding function and put it into the corresponding variables
# split data with appropriate function we made into test and train df

# then we have to divide into x and y

# train models with correct function

# evaluate

# analyze best model
# outpput best model name

# then do the predictions by game situation

# return stuff you gotta return

