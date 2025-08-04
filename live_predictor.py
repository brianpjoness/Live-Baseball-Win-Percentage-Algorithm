
import pickle
import pandas as pd
from datetime import datetime
from API import get_live_game_data, extract_game_features, get_todays_games  # allows you to import from your directory
import time
import json


class LiveMLBPredictor:
    # makes live predictions using model

    def __init__(self, model_path, scaler_path):  # constructor method that runs automatically with the class
        with open(model_path, "rb") as f:  # opens model file in read binary mode
            self.model = pickle.load(f) # loads modek

        with open(scaler_path, 'rb') as f:  # loads scaler file
            self.scaler = pickle.load(f)

        self.feature_names = [   # .self allows oyu to use in it all methods in the class
            'score_diff', 'is_top_inning', 'Outs', 'runners_on',
            'batting_team_is_home', 'inning', 'balls', 'strikes'
        ]
    # these are the features we use for the model
    def predict_game(self, game_pk):
        # makes predictions for live game

        live_data = get_live_game_data(game_pk) # since we imported it from API file
        if not live_data: # if there is no live game
            return None

        features = extract_game_features(live_data)  # this extracts the features
        if not features:
            return None

        metadata = features.pop("_metadata")  # gets rid of metadata from features and stores in this variable
        feature_df = pd.DataFrame([features])

        # scales the game data using the transformation held in the scaler object
        X_scaled = self.scaler.transform(feature_df[self.feature_names])

        # make prediction
        win_probability = self.model.predict_proba(X_scaled)[0][1] # probability of win

        if features["is_top_inning"]:
            half_inning = "↑"
        else:
            half_inning = "↓"
        return { # returns all the corresponding information for the game
            'game_pk': game_pk,
            'batting_team': metadata['batting_team'],
            'win_probability': round(win_probability, 3),
            'home_team': metadata['home_team'],
            'away_team': metadata['away_team'],
            'score': f"{metadata['away_team']} {metadata.get('away_score', 0)} - {metadata.get('home_score', 0)} {metadata['home_team']}",
            'situation': f"Inning {features['inning']}{half_inning}, {features['Outs']} outs, {features['runners_on']} runners on",
            'timestamp': datetime.now().isoformat(),
            "pitcher": metadata["pitcher"],
            "batter": metadata["batter"],
            "is_top_inning": features["is_top_inning"],
            "count": f"{features["balls"]}-{features["strikes"]}"
        }

    def predict_all_live_games(self): # provides just the win probability for each game in progress (for batting team)
        games_data = get_todays_games() # games that are going on now
        live_games = [] # make a list of games that are live

        for date_data in games_data.get('dates', []): # iterates through game data and if the status is in progress or live it adds the game to live games
            for game in date_data.get('games', []):
                if game['status']['detailedState'] in ['In Progress', 'Live']:
                    live_games.append(game['gamePk'])

        print(f"Found {len(live_games)} live games")

        predictions = []
        for game_pk in live_games:
            try:
                prediction = self.predict_game(game_pk) # uses function predict game which creates dictionary with different keys
                if prediction:
                    predictions.append(prediction)
                    print(f" {prediction['batting_team']} win prob: {prediction['win_probability']} - {prediction['situation']} | {prediction["score"]}")

            except Exception as e:
                print(f"Error predicting game {game_pk}: {e}")

        return predictions

def display_game_info(prediction): # formats the data for the current game id that was selected
    if not prediction:
        return



    print(f"\nGame {prediction['game_pk']}: {prediction['away_team']} @ {prediction['home_team']}")
    print(f"Score: {prediction['score']}")
    print(f"Batting Team: {prediction['batting_team']} ({prediction['win_probability'] * 100:.1f}% win probability)")
    print(f"Situation: {prediction['situation']}")
    print(f"Pitcher: {prediction['pitcher']}")
    print(f"Batter: {prediction['batter']}")
    print(f"Count: {prediction["count"]}")
    print(f"Updated: {datetime.fromisoformat(prediction['timestamp']).strftime('%I:%M:%S %p')}")
    print("-" * 50)



