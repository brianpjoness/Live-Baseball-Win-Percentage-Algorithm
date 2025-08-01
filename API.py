
import requests
import json
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_todays_games():  # data on todays game
    url = "https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1" # url that provides schedule of games in MLB api
    response = requests.get(url)
    return response.json()  # python dictionary with information about todays mlb games

def get_live_game_data(game_pk):  # outputs live game info based on the game number you enter
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    response = requests.get(url)
    return response.json() # returns a dictionary



def extract_game_features(live_data):  # turns live mlb data to the format of the model

    # gets the game features from the live data that are needed for the model
    # important to use try-except since depending on the game state the api structure changes

    try:
        game_data = live_data["gameData"]
        live_data_plays = live_data["liveData"]
        linescore = live_data_plays["linescore"]

        # current game state
        current_play = live_data_plays["plays"]["currentPlay"]
        linescore = live_data_plays["linescore"] # linescore has a lot of stats

        # now we get game info
        home_team = game_data["teams"]["home"]["abbreviation"]
        away_team = game_data["teams"]["away"]["abbreviation"]

        current_inning = linescore["currentInning"]
        is_top_inning = linescore["inningHalf"] == "Top"   # True or False depending on if its top or bottom of inning

        # balls, strikes and outs
        count = current_play["count"]
        balls = count["balls"]
        strikes = count["strikes"]
        outs = count["outs"]

        # score of each team
        home_score = linescore["teams"]["home"].get("runs", 0)  # .get returns the value of runs or 0 if it doesnt exist
        away_score = linescore["teams"]["away"].get("runs", 0)

        # this determines logic of the batting teams (home vs away)
        if is_top_inning:
            batting_team = away_team
            batting_team_score = away_score
            pitching_team_score = home_score
            batting_team_is_home = 0
        else:
            batting_team = home_team
            batting_team_score = home_score
            pitching_team_score = away_score
            batting_team_is_home = 1

        score_diff = batting_team_score - pitching_team_score  # score difference

        # runners on base
        runners_on = 0
        try:
            offense = linescore.get("offense", {})
            bases = ["first", "second", "third"]
            for base in bases:   # goes through all of the bases and if there is a runner on that individual base it increments runners on
                if offense.get(base) is not None:
                    runners_on += 1
        except Exception as e: # Exception is an error object and this is passed into e so if there is an error it says what the error is
            print(f"Warning: Could not count runners on base: {e}")
            runners_on = 0

        # the features for the model
        features = {
            'score_diff': score_diff,
            'is_top_inning': int(is_top_inning),
            'Outs': outs,
            'runners_on': runners_on,
            'batting_team_is_home': batting_team_is_home,
            'inning': current_inning,
            'balls': balls,
            'strikes': strikes
        }

        # data to use for display  (not part of the features just useful to see)
        features['_metadata'] = {
            'batting_team': batting_team,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score
        }

        return features

    except KeyError as e: # error caused from missing dictionaries key
        print(f"Error extracting features - missing key: {e}")
        return None
    except Exception as e: # exception on the other hand catches all errors
        print(f"Unexpected error: {e}")
        return None


def main():

    # information on todays games
    games = get_todays_games()

    # this gives the information on all of the games today
    print("Today's games:")
    for date_data in games.get('dates', []):
        for game in date_data.get('games', []):  # goes through each game today
            status = game['status']['detailedState']
            away = game['teams']['away']['team']['name']
            home = game['teams']['home']['team']['name']
            print(f"Game {game['gamePk']}: {away} @ {home} ({status})")

    test_game_pk = 776923

    try:
        live_data = get_live_game_data(test_game_pk)
        features = extract_game_features(live_data)  # gets the features from live data
        print(f"Extracted features: {features}")

    except Exception as e:
        print(f"Error testing: {e}")


main()