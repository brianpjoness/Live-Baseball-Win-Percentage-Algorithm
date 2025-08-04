
import pickle
import pandas as pd
import json
import time
from datetime import datetime
from API import get_live_game_data, extract_game_features, get_todays_games
from live_predictor import *
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import threading
from display import app, update_predictions_continuously



def main():

    # games availiable

    games = get_todays_games()

    # this gives the information on all of the games today
    print("Today's games:")
    for date_data in games.get('dates', []):
        for game in date_data.get('games', []):  # goes through each game today
            status = game['status']['detailedState']
            away = game['teams']['away']['team']['name']
            home = game['teams']['home']['team']['name']
            print(f"Game {game['gamePk']}: {away} @ {home} ({status})")

    test_game_pk = 776890

    try:
        live_data = get_live_game_data(test_game_pk)
        features = extract_game_features(live_data)  # gets the features from live data
        print(f"Extracted features: {features}")

    except Exception as e:
        print(f"Error testing: {e}")

    print("=" * 200)

    # feature extraction

    try:
        live_data = get_live_game_data(test_game_pk)
        features = extract_game_features(live_data)

    except Exception as e:
        print(f"Error testing: {e}")

    # Uncomment when you have saved your model:
    predictor = LiveMLBPredictor('trained_model.pkl', 'scaler.pkl')
    predictions = predictor.predict_all_live_games()

    prediction = predictor.predict_game(test_game_pk)
    print("\nDetailed Game Information:")
    display_game_info(prediction)


    # THIS IS THE FLASK STUFF

    print("\nStarting background prediction updates...")
    background_thread = threading.Thread(target=update_predictions_continuously, daemon=True)
    background_thread.start()


    print("\nStarting Flask web server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
main()