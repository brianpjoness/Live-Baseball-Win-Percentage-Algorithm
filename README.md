This is project utilizes Machine Learning to calculate the win probability of baseball teams during live games. 
The program makes adjustments for various reasons such as the score, men on base, or outs.

The project is made up of several key parts:
1. The XGBoost model trained on previous baseball game information using pybaseball
2. Using mlbstats api to access live games
3. And then using the trained model to make predictions on the live data
4. Finally, key details about the game such as win probability are displayed using flask
