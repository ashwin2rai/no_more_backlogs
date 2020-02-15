from flask import Flask, render_template, request
import pandas as pd
import numpy as np


app = Flask(__name__)

def create_postgres_authdict(user, password, hostandport, dbname ='', fname = 'SQLAuth.sql',save=True):
    import pickle
    
    sqldb_dict = {}
    sqldb_dict['username'] = user
    sqldb_dict['password'] = password
    sqldb_dict['host:port'] = hostandport
    sqldb_dict['dbname'] = dbname
    
    if save:
        try:
            pickle.dump(sqldb_dict,open(fname,'wb'))
        except:
            raise IOError('ERROR: Could not save SQL credentials. Check path.')
    else:
        return create_posgresurl(sqldb_dict = sqldb_dict)


def create_posgresurl(fname = 'SQLAuth.sql', sqldb_dict=None):
    import pickle
    
    if not sqldb_dict:
        try:
            sqldb_dict = pickle.load(open(fname,'rb'))
        except:
            raise IOError('ERROR: Could not load SQL credentials. Check path.')
    return 'postgres://{}:{}@{}/{}'.format(sqldb_dict['username'],sqldb_dict['password'],sqldb_dict['host:port'],sqldb_dict['dbname'])

def sql_readaspd(db_dets, query):
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool    
    
    engine = create_engine(db_dets,poolclass=NullPool)
    con_var = engine.connect()
    rs_var = con_var.execute(query)
    con_var.close
    df = pd.DataFrame(rs_var.fetchall(),columns = rs_var.keys())
    df.index = df['index']
    df = df.drop(['index'],axis=1)
    try:
        df = df.drop(['Unnamed: 0'],axis=1)
    except:
        pass
    return df

@app.route('/',methods=["GET","POST"])
def hello_world():
    return render_template('Homepage.html')

@app.route('/preset',methods=["GET","POST"])
def preset_output():
    game_titlehtml = request.args.get('demo_input')
    sqlurl = 'postgres://user:pass@host/dbname'

    game_query = lambda x: 'SELECT * FROM investigamepred WHERE "TitlesHTML" = \'{}\''.format(x)
    publisher_query = lambda x: 'SELECT * FROM investigamepred WHERE "Publishers" = \'{}\' LIMIT 4'.format(x)
    dev_query = lambda x: 'SELECT * FROM investigamepred WHERE "Developers" = \'{}\' LIMIT 4'.format(x)
    
    game_info = sql_readaspd(create_posgresurl(),game_query(game_titlehtml))
#    game_info = sql_readaspd(sqlurl,game_query(game_titlehtml))
    
    game_same_pubs = sql_readaspd(create_posgresurl(),publisher_query(game_info['Publishers'].values[0]))
#    game_same_pubs = sql_readaspd(sqlurl,publisher_query(game_info['Publishers'].values[0]))
    game_same_pubs = game_same_pubs.drop(game_info.index,axis=0)

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
              <p>Confidence: {}%</p>
            </div>
          </div>"""

    pub_text = ''

    if game_same_pubs.empty:
        pub_text = '<p>No games with the same publishers found.</p>'
    else:
        for row_tuple in game_same_pubs.itertuples():
            pub_text_temp = template.format(row_tuple.GameCard,row_tuple.Titles,
                                            row_tuple.Publishers,
                                           row_tuple.Developers,
                                           row_tuple.Reviews,
                                           row_tuple.SuccessPredictText,
                                           np.around([row_tuple.SuccessPredictProb][0]*100,decimals=2))
            pub_text = pub_text + pub_text_temp

    with open('templates/Gamepage.html') as file:
        html_file = file.read()
        html_file = html_file.replace('MAT115',pub_text)

    with open('templates/Gamepage2.html',mode = 'w') as file:
        file.write(html_file)

    del html_file

    game_same_devs = sql_readaspd(create_posgresurl(),dev_query(game_info['Developers'].values[0]))
#    game_same_devs = sql_readaspd(sqlurl,dev_query(game_info['Developers'].values[0]))
    game_same_devs = game_same_devs.drop(game_info.index,axis=0)

    pub_text = ''

    if game_same_devs.empty:
        pub_text = '<p> No games with the same developers found. </p>'
    else:
        for row_tuple in game_same_devs.itertuples():
            pub_text_temp = template.format(row_tuple.GameCard,row_tuple.Titles,
                                            row_tuple.Publishers,
                                           row_tuple.Developers,
                                           row_tuple.Reviews,
                                           row_tuple.SuccessPredictText,
                                           np.around([row_tuple.SuccessPredictProb][0]*100, decimals=2))
            pub_text = pub_text + pub_text_temp

    with open('templates/Gamepage2.html') as file:
        html_file = file.read()
        html_file = html_file.replace('MAT215', pub_text)

    with open('templates/Gamepage3.html',mode = 'w') as file:
        file.write(html_file)

    return render_template('Gamepage3.html', game_name = str(game_info['Titles'].values[0]),
                          game_pub = str(game_info['Publishers'].values[0]),
                          game_dev = str(game_info['Developers'].values[0]),
                          game_rev = str(game_info['Reviews'].values[0]),
                          game_pred = str(game_info['SuccessPredictText'].values[0]),
                          game_pred_prob = str(game_info['SuccessPredictProb'].mul(100).round(2).values[0]),
                          game_url = str(game_info['GameCard'].values[0]))

@app.route('/nopage')
def nopage_output():
    return render_template('Nopage.html')

@app.route('/aboutme')
def aboutme_output():
    return render_template('Aboutme.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True) 
#    app.run(debug=True) #will run locally http://127.0.0.1:5000/
