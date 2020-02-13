from .config import data_dir_var 
from .config import sql_db

def create_datadir_link(data_path = data_dir_var,
                        filename = 'text.csv'):
    return str(data_path/filename)


def get_web_content(addr, ret = 'html'):
    import requests
    from lxml import html
    from bs4 import BeautifulSoup
    """
    Gets content from a webpage and returns BSoup object or html object or both
    
    Parameters
    ----------
    addr: str
        webpage address
    ret: str, optional
        tag that decides what object the function returns.
        if 'html' function returns html object, 'soup' function returns BSoup object
        'htmlsoup' returns both objects
        Default = 'html'
    
    Returns
    -------
    BSoup object
        if ret tag is 'soup'
    html object
        if ret tag is 'html'
    tuple of (html object, BSoup object)
        if ret tag is 'htmlsoup'
        
    Notes
    -----
    [None]

    """
    try:
        pageContent = requests.get(addr) 
    except:
        raise GeneratorExit('ERROR: Webpage inaccessible, please correct webpage address.')
        
    if ret == 'html':
        return html.fromstring(pageContent.content)
    elif ret == 'soup':
        return BeautifulSoup(pageContent.text, 'html.parser')
    elif ret == 'htmlsoup':
        return (html.fromstring(pageContent.content),BeautifulSoup(pageContent.text, 'html.parser'))
    else:
        raise ValueError('ERROR: ret tag is invalid in get_web_content')

def complete_gamedb(Complete_game_db, succes_prob):
    import pandas as pd
    import numpy as np
    pred_prob = np.where(succes_prob[:,0]>succes_prob[:,1],succes_prob[:,0],succes_prob[:,1])
    predictions = np.where(succes_prob[:,0]>succes_prob[:,1],0,1)
    Complete_game_db['SuccessPredict'] = np.nan
    Complete_game_db['SuccessPredict'] = pd.DataFrame(predictions,index=Complete_game_db.index,columns=['SuccessPredict'])
    Complete_game_db['SuccessPredictProb'] = np.nan
    Complete_game_db['SuccessPredictProb'] = pd.DataFrame(pred_prob,index=Complete_game_db.index,columns=['SuccessPredictProb'])
    Complete_game_db['SuccessPredictText'] = Complete_game_db['SuccessPredict'].map({0:'Failure',1:'Success'})
 

def create_reddit_OAuth(client_id, api_key, username, password, 
                        user_agent_key, filename = 'RedditAuth.sav'):
    import pickle
    reddit_auth = {}
    reddit_auth = {'client_id': client_id,
               'API_key': api_key,
               'username': username,
               'password': password,
               'user_agent': user_agent_key
              }
    try:
        pickle.dump(reddit_auth, open(filename, 'wb'))
    except:
        raise IOError('ERROR: Could not write Reddit credentials file. Check path.')
    del reddit_auth

def write_tocsv(df, data_path=None, fname = 'file.csv'):
    if not data_path:
        fpath = create_datadir_link(filename = fname)
    else:
        fpath = create_datadir_link(data_path = data_path, filename = fname)
    try:
        df.to_csv(fpath)
    except:
        print('WARNING: Could not save file, check if dataframe was created properly or path is right.')

def create_postgres_authdict(user, password, hostandport, dbname ='', dir_path = sql_db, fname = 'SQLAuth.sql',save=True):
    import pickle
    
    sqldb_dict = {}
    sqldb_dict['username'] = user
    sqldb_dict['password'] = password
    sqldb_dict['host:port'] = hostandport
    sqldb_dict['dbname'] = dbname
    
    if save:
        try:
            pickle.dump(sqldb_dict,open(create_datadir_link(data_path = dir_path, filename=fname),'wb'))
        except:
            raise IOError('ERROR: Could not save SQL credentials. Check path.')
    else:
        return create_posgresurl(sqldb_dict = sqldb_dict)

def create_posgresurl(dir_path = sql_db, fname = 'SQLAuth.sql', sqldb_dict=None):
    import pickle
    
    if not sqldb_dict:
        try:
            sqldb_dict = pickle.load(open(create_datadir_link(data_path = dir_path, filename=fname),'rb'))
        except:
            raise IOError('ERROR: Could not load SQL credentials. Check path.')
    return 'postgres://{}:{}@{}/{}'.format(sqldb_dict['username'],sqldb_dict['password'],sqldb_dict['host:port'],sqldb_dict['dbname'])

def sql_readaspd(db_dets, query):
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool    
    import pandas as pd
    
    engine = create_engine(db_dets,poolclass=NullPool)
    con_var = engine.connect()
    rs_var = con_var.execute(query)
    con_var.close
    df = pd.DataFrame(rs_var.fetchall())
    df.columns = rs_var.keys()
    
    return df

def sql_pd_write(df, db_dets, table_name):
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool    
    import pandas as pd
    
    engine = create_engine(db_dets, poolclass=NullPool)
    con_var = engine.connect()
    df.to_sql(table_name, con=engine)
    con_var.close
