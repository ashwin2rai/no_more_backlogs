from flask import Flask, render_template, request
import pandas as pd


from flask import Flask
app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def hello_world():
    return render_template('Homepage.html')

@app.route('/preset',methods=["GET","POST"])
def preset_output():
    game_titlehtml = request.args.get('demo_input')
    games = pd.read_csv('ps4_game_list_predicted.csv')
    game_info = games[games['TitlesHTML'] == game_titlehtml]
    game_same_pubs = games[games['Publishers'] == game_info['Publishers'].values[0]][0:3]
    game_same_pubs = game_same_pubs.drop(game_info.index,axis=0)
    
    game_url_temp = "https://store.playstation.com/store/api/chihiro/00_09_000/container/US/en/19/UP2034-CUSA04841_00-NMSDIGITAL000001/image?w=320&h=320&bg_color=000000&opacity=100&_version=00_09_000"
    
    template = """<div class="media-object stack-for-small">
            <div class="media-object-section">
              <img class="thumbnail" src="{}">
            </div>
            <div class="media-object-section">
              <h5>Game: {}</h5>
              <p>Publisher: {}</p>
              <p>Developer: {}</p>
              <p>Aggregate Review: {}</p>
              <p>Prediction: {}</p>
              <p>Probability: {}</p>
            </div>
          </div>"""
    
    pub_text = ''
    
    if game_same_pubs.empty:
        pub_text = '<p>No games with the same publishers found.</p>'
    else:
        for row_tuple in game_same_pubs.itertuples():
            pub_text_temp = template.format(game_url_temp,row_tuple.Titles,
                                            row_tuple.Publishers,
                                           row_tuple.Developers,
                                           row_tuple.Reviews,
                                           row_tuple.SuccessPredictText,
                                           row_tuple.SuccessPredictProb) 
            pub_text = pub_text + pub_text_temp
    
    
    with open('templates/Gamepage.html') as file:
        a = file.read()
        a = a.replace('MAT115',pub_text)

    with open('templates/Gamepage2.html',mode = 'w') as file:
        file.write(a)
        
    del a
    
    game_same_devs = games[games['Developers'] == game_info['Developers'].values[0]][0:3]
    game_same_devs = game_same_devs.drop(game_info.index,axis=0)

    pub_text = ''
    
    if game_same_devs.empty:
        pub_text = '<p> No games with the same developers found. </p>'
    else:
        for row_tuple in game_same_devs.itertuples():
            pub_text_temp = template.format(game_url_temp,row_tuple.Titles,
                                            row_tuple.Publishers,
                                           row_tuple.Developers,
                                           row_tuple.Reviews,
                                           row_tuple.SuccessPredictText,
                                           row_tuple.SuccessPredictProb) 
            pub_text = pub_text + pub_text_temp
    
    with open('templates/Gamepage2.html') as file:
        a = file.read()
        a = a.replace('MAT215', pub_text)

    with open('templates/Gamepage3.html',mode = 'w') as file:
        file.write(a)
    
    return render_template('Gamepage3.html', game_name = str(game_info['Titles'].values[0]),
                          game_pub = str(game_info['Publishers'].values[0]),
                          game_dev = str(game_info['Developers'].values[0]),
                          game_rev = str(game_info['Reviews'].values[0]),
                          game_pred = str(game_info['SuccessPredictText'].values[0]),
                          game_pred_prob = str(game_info['SuccessPredictProb'].round(2).values[0]),
                          game_url = game_url_temp)

@app.route('/nopage')
def nopage_output():
    return render_template('Nopage.html')


@app.route('/game/<name>')
def hello_name(name):
    games = pd.read_csv('ps4_game_list_predicted.csv')
    textval = 'The Game: '+ str(games[games['TitlesHTML'] == name]['Titles'].values)+' is predicted to be a '+str(games[games['TitlesHTML'] == name]['SuccessPredictText'].values)+', predicted with a probability of ' +str(games[games['TitlesHTML'] == name]['SuccessPredictProb'].values)+ '%.'
    return textval

if __name__ == '__main__':
    app.run(debug=True) #will run locally http://127.0.0.1:5000/
