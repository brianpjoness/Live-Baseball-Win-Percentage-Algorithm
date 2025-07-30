
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
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = statcast(start_dt = "2023-04-01", end_dt="2023-04-30")


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

    print("---------------------------------------------------------------------------------------------------------------------------")
    print("Dataframe Shape: ")
    print(modeling_df.shape)
    print("Target Distribution: ")
    print(modeling_df.value_counts(normalize=True))
    print("--------------------------------------------------------------------------------------------------------------------------")
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
train_df, test_df = temporal_train_test_split(modeling_df)

def train_baseline_models(train_df, test_df, feature_names):
    # train logistic regression and random forest models
    # make sure to do scaler for logistic regression
    # do model results and feature importances

    scaler = StandardScaler()
    X_train= scaler.fit_transform(train_df[feature_names])
    y_train = train_df["is_batting_team_winner"]

    X_test =  scaler.fit_transform(test_df[feature_names])
    y_test = test_df["is_batting_team_winner"]

    log_reg = LogisticRegression(penalty="l1", C=0.01, max_iter=100, class_weight="balanced", solver="liblinear")
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)



    # regression results
    print("Logistic Regression Results: ")

    # return models and results, print importanec

    print("Accuracy: ", accuracy_score(y_test, y_pred))  # percentage of correct predictions out of all predictions
    print(classification_report(y_test, y_pred)) # precision, recall, f1 score
    print("ROC AUC: ", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])) # receiver operating characteristic area under the curve
    # shows how well classifier seperates positive and negative classes 1 is perfect 0.5 is guessing
    print("Log Loss: ", log_loss(y_test, log_reg.predict_proba(X_test))) # how confident the model is in its predictions lower is better (0 perfect)
    print("--------------------------------------------------------------------------------------------------------------------------")

    # split up the param grid so i can try lbfgs since it only works with l2
    param_grid = [
        {
            'solver': ['lbfgs'],
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200],
            'class_weight': [None, 'balanced'],
        },
        {
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200],
            'class_weight': [None, 'balanced'],
        },
        {
            'solver': ['saga'],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200],
            'class_weight': [None, 'balanced'],
        }
    ]
    model = LogisticRegression()
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best Parameters: ", grid.best_params_)
    print("--------------------------------------------------------------------------------------------------------------------------")
    return X_train, X_test, y_train, y_test, y_pred, log_reg


X_train, X_test, y_train, y_test, y_pred, log_reg = train_baseline_models(train_df, test_df, feature_cols)



def analyze_predictions_by_game_situation(df, y_predictions, y_test):
    # inning, score dif, on base
    def create_score_diff_buckets(score_diff):
        # converts raw score into different categories
        if score_diff >= 5:
            return "Up by 5+"
        elif score_diff >= 2:
            return "Up by 2-4"
        elif score_diff == 1:
            return "Up by 1"
        elif score_diff == 0:
            return "Tied"
        elif score_diff == -1:
            return "Down by 1"
        elif score_diff >= -4:
            return "Down by 2-4"
        else:
            return "Down by 5+"


    predictions_df = df[["inning", "score_diff"]].copy() # double brackets
    predictions_df["Predicted Result"] = y_predictions
    predictions_df["Actual Result"] = y_test
    predictions_df["score_diff"] = predictions_df["score_diff"].apply(create_score_diff_buckets)

    inning_summary = predictions_df.groupby("inning").apply(
        lambda group: pd.Series({       # variable group is dataframe containing all the rows for a specific inning
            "accuracy": (group["Predicted Result"] == group["Actual Result"]).mean(), # if predicted result is the same as actual result it is assigned a 1, ow its a 0
            # then you take the average of the 1s and 0s and this gives the accuracy
            "count": len(group)   # this makes a column for how many rows are in each innings df
        }),
        include_groups=False # gets rid of depracation warning
    )

    score_diff_summary = predictions_df.groupby("score_diff").apply(
        lambda group: pd.Series({
            "accuracy": (group["Predicted Result"] == group["Actual Result"]).mean(),
            "count": len(group)
        }),
        include_groups=False
    )



    return (inning_summary,
            score_diff_summary)

print(analyze_predictions_by_game_situation(test_df, y_pred, y_test))

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

