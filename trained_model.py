import pickle
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
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV

df = statcast(start_dt = "2023-04-01", end_dt="2023-07-30")


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
        'score_diff',           # Current score df (batting pov)
        'is_top_inning',        # Whether it's top of inning
        'Outs',                 # Number of outs
        'runners_on',           # Number of runners on base
        'batting_team_is_home', # Whether batting team is home
        'inning',               # What inning it is
        'balls',                # Current ball count
        'strikes'             # Current strike count

    ]
    target = 'is_batting_team_winner'
 # return modeling df, feature cols, target

    modeling_df = df[feature_cols + [target, "game_pk", "game_date"]].dropna()

    print("=" * 50)
    print("Dataframe Shape: ")
    print(modeling_df.shape)
    print("=" * 50)
    print("Target Distribution: ")
    print(modeling_df.value_counts(normalize=True))
    print("=" * 50)
    return modeling_df, feature_cols, target



def temporal_train_test_split(df, test_size =0.2):
    # important to split data based on the game to avoid data leakage
    # go by date
    df = df.sort_values(["game_date", "game_pk"])
    split_index = int(len(df) * (1 - test_size))
    test_df = df[split_index:]
    train_df = df[:split_index]

    return train_df, test_df

# return train and test df


def train_baseline_models(train_df, test_df, feature_names):
    # train logistic regression and random forest models
    # make sure to do scaler for logistic regression
    # do model results and feature importances

    scaler = StandardScaler()
    X_train= scaler.fit_transform(train_df[feature_names])
    y_train = train_df["is_batting_team_winner"]

    X_test =  scaler.transform(test_df[feature_names]) # important that it is scaler.transform to avoid data leakage
    # fit_transform is calculating and applying with the mean and standard deviation of that dataset
    # for test you want to just use transform because that takes the scaler.fit_transform that is stored and was used on X_train
    # if you fit_transform on the test it is data leakage
    y_test = test_df["is_batting_team_winner"]

    # logistic regression model
    # rf = LogisticRegression(penalty="l1", C=0.01, max_iter=100, class_weight="balanced", solver="liblinear")
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)

    # random forest model
    # rf = RandomForestClassifier(n_estimators=150,        # number of trees in the forest
    #                         max_depth=10,          # let trees grow fully (can overfit on small data)
    #                         min_samples_split=5,     # minimum number of samples to split a node
    #                         min_samples_leaf=5,      # minimum samples required at a leaf node
    #                         max_features='log2',     # √num_features, good for classification
    #                         bootstrap=True,          # whether to sample with replacement
    #                         random_state=42,         # for reproducibility
    #                         n_jobs=-1,
    #                         class_weight="balanced")
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)

    xgb = XGBClassifier(n_estimators=100, max_depth=None, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)


    # regression results
    print("Model Results: ")

    # return models and results, print importanec

    print("Accuracy: ", accuracy_score(y_test, y_pred))  # percentage of correct predictions out of all predictions
    print(classification_report(y_test, y_pred)) # precision, recall, f1 score
    print("ROC AUC: ", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])) # receiver operating characteristic area under the curve
    # shows how well classifier seperates positive and negative classes 1 is perfect 0.5 is guessing
    print("Log Loss: ", log_loss(y_test, xgb.predict_proba(X_test))) # how confident the model is in its predictions lower is better (0 perfect)
    print("=" * 50)

    # split up the param grid so i can try lbfgs since it only works with l2
    # param_grid = [
    #     {
    #         'solver': ['lbfgs'],
    #         'penalty': ['l2'],
    #         'C': [0.01, 0.1, 1, 10, 100],
    #         'max_iter': [100, 200],
    #         'class_weight': [None, 'balanced'],
    #     },
    #     {
    #         'solver': ['liblinear'],
    #         'penalty': ['l1', 'l2'],
    #         'C': [0.01, 0.1, 1, 10, 100],
    #         'max_iter': [100, 200],
    #         'class_weight': [None, 'balanced'],
    #     },
    #     {
    #         'solver': ['saga'],
    #         'penalty': ['l1', 'l2'],
    #         'C': [0.01, 0.1, 1, 10, 100],
    #         'max_iter': [100, 200],
    #         'class_weight': [None, 'balanced'],
    #     }
    # ]
    # model = LogisticRegression()
    # grid = GridSearchCV(model, param_grid, cv=5)
    # grid.fit(X_train, y_train)
    #
    # print("Best Parameters: ", grid.best_params_)
    # print("-" * 50)

    # random forest param grid

    # param_grid = {
    #     'n_estimators': [150, 200],  # Number of trees in the forest
    #     'max_depth': [5, 10, 15],  # Maximum depth of trees
    #     'min_samples_split': [6, 5, 4,],  # Min samples required to split a node
    #     'min_samples_leaf': [5, 7, 10],  # Min samples required at each leaf node
    #     'max_features': ["log2"],  # Number of features to consider at each split
    #     'bootstrap': [True, False],  # Whether bootstrap samples are used
    #     'class_weight': [None, 'balanced'],
    #     "n_jobs": [-1]
    # }
    # model = RandomForestClassifier()
    # grid = HalvingRandomSearchCV(model, param_grid, cv=5)  # apparentl gridsearchcv is just as good since we dont have too many features
    # grid.fit(X_train, y_train)
    # print("Best Parameters: ", grid.best_params_)
    # print("-" * 50)




    # get the importances of the features
    # feature_importance = pd.Series(rf.coef_[0], index=feature_names) # this is how you do feature importance for log reg
    feature_importance = pd.Series(xgb.feature_importances_, index=feature_names)  # this is how you do feature importance for rf
    print("Feature Importances: ")
    print(feature_importance)
    print("=" * 50)

    return X_train, X_test, y_train, y_test, y_pred, xgb, scaler


def save_model_and_scalar(model, scaler):
    # this saves the trained model and the scaler so we can use it in our predictor file
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f) # this saves the model

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model and Scaler saved successfully")

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

    print(inning_summary)
    print("\n" + "=" * 50 + "\n")
    print(score_diff_summary)

    return (inning_summary,
            score_diff_summary)



def run_modeling_pipeline(df):
    modeling_df, feature_cols, target = prepare_modeling_data(df)
    train_df, test_df = temporal_train_test_split(modeling_df)
    X_train, X_test, y_train, y_test, y_pred, xgb, scaler = train_baseline_models(train_df, test_df, feature_cols)
    (analyze_predictions_by_game_situation(test_df, y_pred, y_test))

    save_model_and_scalar(xgb, scaler)
    return xgb, scaler



model, scaler = run_modeling_pipeline(df)