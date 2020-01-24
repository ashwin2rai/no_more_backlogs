from flask import Flask, render_template, request
import pandas as pd


from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Open the URL in another tab, and then go to /game/game+name'

@app.route('/game/<name>')
def hello_name(name):
    games = pd.read_csv('ps4_game_list_predicted.csv')
    textval = 'The Game: '+ str(games[games['TitlesHTML'] == name]['Titles'].values)+' is predicted to be a '+str(games[games['TitlesHTML'] == name]['SuccessPredictText'].values)+', predicted with a probability of ' +str(games[games['TitlesHTML'] == name]['SuccessPredictProb'].values)+ '%.'
    return textval

if __name__ == '__main__':
    app.run(debug=True) #will run locally http://127.0.0.1:5000/
