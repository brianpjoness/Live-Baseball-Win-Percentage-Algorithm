
import pandas as pd
import numpy as np
from pybaseball import statcast

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = statcast(start_dt = "2023-04-01", end_dt="2023-04-07")


keep_cols = [
    'game_pk', 'game_date', 'inning', 'inning_topbot',
    'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
    'bat_score', 'fld_score', 'home_team', 'away_team',
     'events', 'description', "balls", "strikes"
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
print(df["batting_team_is_home"].head(75))

# print(list(df.columns))
# print(df.head())