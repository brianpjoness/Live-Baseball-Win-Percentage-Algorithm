
import requests
import json

def get_todays_games():  # data on todays game
    url = "https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1" # url that provides schedule of games in MLB api
    response = requests.get(url)
    return response.json()  # python dictionary with information about todays mlb games

def get_live_game_data(game_pk):  # outputs live game info based on the game number you enter
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    response = requests.get(url)
    return response.json()

