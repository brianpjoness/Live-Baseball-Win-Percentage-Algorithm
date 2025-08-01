
import pickle
import pandas as pd
from datetime import datetime
from .API import get_live_game_data, extract_game_features, get_live_game_data, get_todays_games  # allows you to import from your directory

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
        if not live_data: # if there is no game
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

        return {
            'game_pk': game_pk,
            'batting_team': metadata['batting_team'],
            'win_probability': round(win_probability, 3),
            'home_team': metadata['home_team'],
            'away_team': metadata['away_team'],
            'score': f"{metadata['away_team']} {metadata.get('away_score', 0)} - {metadata.get('home_score', 0)} {metadata['home_team']}",
            'situation': f"Inning {features['inning']}, {features['Outs']} outs, {features['runners_on']} runners on",
            'timestamp': datetime.now().isoformat()
        }

