
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

app = Flask(__name__)
CORS(app)

# Global variable to store latest predictions
latest_predictions = []

# Initialize predictor
predictor = LiveMLBPredictor('trained_model.pkl', 'scaler.pkl')


def update_predictions_continuously():
    """Background thread to update predictions every 30 seconds"""
    global latest_predictions

    while True:
        try:
            print(f"\n{datetime.now().strftime('%H:%M:%S')} - Updating predictions...")
            latest_predictions = predictor.predict_all_live_games()
            print(f"Updated {len(latest_predictions)} predictions")

        except Exception as e:
            print(f"Error updating predictions: {e}")

        time.sleep(30)  # Wait 30 seconds


# API Routes
@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get latest predictions"""
    return jsonify(latest_predictions)


@app.route('/api/game/<int:game_pk>')
def get_single_game(game_pk):
    """API endpoint to get prediction for a specific game"""
    try:
        prediction = predictor.predict_game(game_pk)
        if prediction:
            return jsonify(prediction)
        else:
            return jsonify({"error": "Game not found or not live"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Main dashboard route
@app.route('/')
def dashboard(): # flask allows you to write html in python
    """Serve the main dashboard page"""
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Live Predictions Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: white;
        }

        .header {
            text-align: center;
            padding: 2rem 0;
            background: rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .controls {
            max-width: 1200px;
            margin: 0 auto 2rem;
            padding: 0 1rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }

        .refresh-btn {
            background: linear-gradient(45deg, #ff6b6b, #ee5a6f);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        }

        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }

        .status {
            padding: 8px 16px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }

        .games-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
        }

        .game-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .game-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            background: rgba(255, 255, 255, 0.15);
        }

        .game-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .game-id {
            font-size: 0.8rem;
            opacity: 0.7;
            background: rgba(255, 255, 255, 0.1);
            padding: 4px 8px;
            border-radius: 10px;
        }

        .teams {
            text-align: center;
            margin-bottom: 1rem;
        }

        .matchup {
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .score {
            font-size: 1.1rem;
            color: #ffd700;
            font-weight: bold;
        }

        .probability-section {
            text-align: center;
            margin: 1.5rem 0;
        }

        .batting-team {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            color: #4ecdc4;
        }

        .probability-bar {
            width: 100%;
            height: 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
            position: relative;
        }

        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            border-radius: 15px;
            transition: width 1s ease;
            position: relative;
        }

        .probability-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
            z-index: 2;
        }

        .game-situation {
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        }

        .situation-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .situation-row:last-child {
            margin-bottom: 0;
        }

        .label {
            opacity: 0.8;
        }

        .value {
            font-weight: bold;
        }

        .timestamp {
            text-align: center;
            margin-top: 1rem;
            font-size: 0.8rem;
            opacity: 0.6;
        }

        .no-games {
            text-align: center;
            padding: 3rem;
            opacity: 0.7;
            font-size: 1.2rem;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            font-size: 1.1rem;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .error {
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid rgba(255, 107, 107, 0.5);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚾ MLB Live Predictions</h1>
        <p>Real-time win probability predictions for live games</p>
    </div>

    <div class="controls">
        <button class="refresh-btn" onclick="refreshPredictions()">🔄 Refresh Predictions</button>
        <div class="status" id="status">Ready</div>
    </div>

    <div class="games-container" id="gamesContainer">
        <div class="loading">
            <div class="spinner"></div>
            Loading live games...
        </div>
    </div>

    <script>
        function createGameCard(game) {
            const probabilityPercent = (game.win_probability * 100).toFixed(1);
            const lastUpdated = new Date(game.timestamp).toLocaleTimeString();

            return `
                <div class="game-card">
                    <div class="game-header">
                        <div class="game-id">Game ${game.game_pk}</div>
                    </div>

                    <div class="teams">
                        <div class="matchup">${game.away_team} @ ${game.home_team}</div>
                        <div class="score">${game.score}</div>
                    </div>

                    <div class="probability-section">
                        <div class="batting-team">🏏 ${game.batting_team} Batting</div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${probabilityPercent}%"></div>
                            <div class="probability-text">${probabilityPercent}% Win Probability</div>
                        </div>
                    </div>

                    <div class="game-situation">
                        <div class="situation-row">
                            <span class="label">Situation:</span>
                            <span class="value">${game.situation}</span>
                        </div>
                        <div class="situation-row">
                            <span class="label">Count:</span>
                            <span class="value">${game.count}</span>
                        </div>
                        <div class="situation-row">
                            <span class="label">Pitcher:</span>
                            <span class="value">${game.pitcher}</span>
                        </div>
                        <div class="situation-row">
                            <span class="label">Batter:</span>
                            <span class="value">${game.batter}</span>
                        </div>
                    </div>

                    <div class="timestamp">Last updated: ${lastUpdated}</div>
                </div>
            `;
        }

        function displayGames(games) {
            const container = document.getElementById('gamesContainer');

            if (!games || games.length === 0) {
                container.innerHTML = '<div class="no-games">No live games currently in progress</div>';
                return;
            }

            container.innerHTML = games.map(createGameCard).join('');
        }

        function updateStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.style.background = isError ? 'rgba(255, 107, 107, 0.3)' : 'rgba(255, 255, 255, 0.1)';
        }

        async function refreshPredictions() {
            updateStatus('🔄 Refreshing...');

            try {
                const response = await fetch('/api/predictions');

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const games = await response.json();
                displayGames(games);
                updateStatus(`✅ Updated ${games.length} live games`);

            } catch (error) {
                console.error('Error fetching predictions:', error);
                updateStatus('❌ Error loading predictions', true);
                document.getElementById('gamesContainer').innerHTML = 
                    '<div class="error">Failed to load predictions. Error: ' + error.message + '</div>';
            }
        }

        // Initialize dashboard
        refreshPredictions();

        // Auto-refresh every 30 seconds
        setInterval(refreshPredictions, 30000);
    </script>
</body>
</html>'''

    return render_template_string(html_template)