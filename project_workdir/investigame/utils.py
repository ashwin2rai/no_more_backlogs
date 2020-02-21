from .config import data_dir_var
from .config import sql_db

def create_datadir_link(data_path=data_dir_var,
                        filename='text.csv'):
    """
    Used to create a str that contains path to file using a Path object.
    
    Parameters
    ----------
    data_path: Path object, optional
        Path to file, passed as a Path object using the Path library.
        Default data_dir_var taken from config file
    
    filename: str, optional
        Name of the file
        Default 'text.csv'
    
    Returns
    -------
    str
        Path to filename
    """
    return str(data_path / filename)


def get_web_content(addr, ret='html'):
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
    
    Raises
    ------
    ValueError: if ret tag is invalid
        
    """
    import requests
    from lxml import html
    from bs4 import BeautifulSoup

    try:
        pageContent = requests.get(addr)
    except:
        raise GeneratorExit('ERROR: Webpage inaccessible, please correct webpage address.')

    if ret == 'html':
        return html.fromstring(pageContent.content)
    elif ret == 'soup':
        return BeautifulSoup(pageContent.text, 'html.parser')
    elif ret == 'htmlsoup':
        return (html.fromstring(pageContent.content), BeautifulSoup(pageContent.text, 'html.parser'))
    else:
        raise ValueError('ERROR: ret tag is invalid in get_web_content')


def complete_gamedb(Complete_game_db, succes_prob):
    """
    Maps predictions to interpretable success signals in the dataframe     
    
    Parameters
    ----------
    Complete_game_db: Pandas.DataFrame, shape(cols,N) where N is number of games
        DataFrame containing the list of games
        
    succes_prob: array, shape (N X 2)
        Probability matrix obtained after performin predict_proba using binary classifier 
        
    Returns
    -------
    None
        Cols are added to Complete_game_db
        
    """
    import pandas as pd
    import numpy as np
    pred_prob = np.where(succes_prob[:, 0] > succes_prob[:, 1], succes_prob[:, 0], succes_prob[:, 1])
    predictions = np.where(succes_prob[:, 0] > succes_prob[:, 1], 0, 1)
    Complete_game_db['SuccessPredict'] = np.nan
    Complete_game_db['SuccessPredict'] = pd.DataFrame(predictions, index=Complete_game_db.index,
                                                      columns=['SuccessPredict'])
    Complete_game_db['SuccessPredictProb'] = np.nan
    Complete_game_db['SuccessPredictProb'] = pd.DataFrame(pred_prob, index=Complete_game_db.index,
                                                          columns=['SuccessPredictProb'])
    Complete_game_db['SuccessPredictText'] = Complete_game_db['SuccessPredict'].map({0: 'Failure', 1: 'Success'})


def create_reddit_OAuth(client_id, api_key, username, password,
                        user_agent_key, filename='RedditAuth.sav'):
    """
    Creates a dictionary with Reddit API OAuth credentials and saves it as a pickled file for later loading..    
    
    Parameters
    ----------
    client_id: str
    
    api_key: str

    username: str

    password: str

    user_agent_key: str
    
    filename: str, optional
        Filename for the pickled file containing the dictionary.
        Defailt RedditAuth.sav
       
    Returns
    -------
    None
        Dictionary is saved as a pickled file

    Raises
    ------
    IOError: If file cannot be saved.
        
    """

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


def write_tocsv(df, data_path=None, fname='file.csv'):
    """
    Writes a DataFrame to csv using the paths provided in the config files.
    
    Parameters
    ----------
    df: Pandas.DataFrame
    
    data_path: Path object, optional
        The path to the file generated using the Path library
        Default None
    
    filename: str, optional
        Filename for the csv file.
        Defailt file.sav
       
    Returns
    -------
    None
             
    """
    if not data_path:
        fpath = create_datadir_link(filename=fname)
    else:
        fpath = create_datadir_link(data_path=data_path, filename=fname)
    try:
        df.to_csv(fpath)
    except:
        print('WARNING: Could not save file, check if dataframe was created properly or path is right.')


def create_postgres_authdict(user, password, hostandport, dbname='', dir_path=sql_db, fname='SQLAuth.sql', save=True):
    """
    Creates a dictionary with PostGresSQL credentials and saves it as a pickled file containing a dictionary
    with credientials for later loading or creates a URL for immediate use.
    
    Parameters
    ----------
    user: str
        username for DataBase
    
    password: str

    hostandport: str
        Hostname and port for ex: 55.11.22.44:5432

    dbname: str, optinal
        Database name
        Default ''

    dir_path: Path object, optional
        Path to save the file. Default is pulled from config files.
        Default sql_db
    
    filename: str, optional
        Filename for the pickled file containing the dictionary.
        Defailt SQLAuth.sql

    save: bool, optional
        If save is true, dictionary is saved otherwise a url is returned.
        Defailt True
       
    Returns
    -------
    None
        if save is True
    Str
        PostgreSQL url to connect to a Database

    Raises
    ------
    IOError: If file cannot be saved.
        
    """

    import pickle

    sqldb_dict = {}
    sqldb_dict['username'] = user
    sqldb_dict['password'] = password
    sqldb_dict['host:port'] = hostandport
    sqldb_dict['dbname'] = dbname

    if save:
        try:
            pickle.dump(sqldb_dict, open(create_datadir_link(data_path=dir_path, filename=fname), 'wb'))
        except:
            raise IOError('ERROR: Could not save SQL credentials. Check path.')
    else:
        return create_posgresurl(sqldb_dict=sqldb_dict)


def create_posgresurl(sqldb_dict=None, dir_path=sql_db, fname='SQLAuth.sql'):
    """
    Creates a PostGresSQL url that can be used to connect to a Database. 
    
    Parameters
    ----------
    sqldb_dict: dict, optional
        If None the function will load the SQL credentials pickled file. Otherwise SQL credentials can be explicitely
        provided using this parameter.
    
    dir_path: Path object, optional
        Path to SQL credentials pickled file. SQL credentials can be created using create_postgres_authdict() or
        explicitely provided. Default is pulled from config files.
        Unnecessary if sqldb_dict is provided.
        Default sql_db
      
    filename: str, optional
        Filename of the SQL credentials pickled file.
        Defailt SQLAuth.sql
       
    Returns
    -------
    Str
        PostgreSQL url to connect to a Database

    Raises
    ------
    IOError: If credentials file cannot be loaded.
        
    """
    import pickle

    if not sqldb_dict:
        try:
            sqldb_dict = pickle.load(open(create_datadir_link(data_path=dir_path, filename=fname), 'rb'))
        except:
            raise IOError('ERROR: Could not load SQL credentials. Check path.')
    return 'postgres://{}:{}@{}/{}'.format(sqldb_dict['username'], sqldb_dict['password'], sqldb_dict['host:port'],
                                           sqldb_dict['dbname'])


def sql_readaspd(db_dets, query):
    """
    Pull a SQL query output and read it into a Pandas DataFrame 
    
    Parameters
    ----------
    db_dets: str
        URL of a postgreSQL database that will be connected to.
    
    query: str
        SQL query
       
    Returns
    -------
    Pandas.DataFrame
        DataFrame containing output of the query
        
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    import pandas as pd

    engine = create_engine(db_dets, poolclass=NullPool)
    con_var = engine.connect()
    rs_var = con_var.execute(query)
    con_var.close
    df = pd.DataFrame(rs_var.fetchall())
    df.columns = rs_var.keys()

    return df


def sql_pd_write(df, db_dets, table_name):
    """
    Transfer a Pandas DataFrame to a new SQL Table
    
    Parameters
    ----------
    df: Pandas.DataFrame
    
    db_dets: str
        URL of a postgreSQL database that will be connected to.
    
    table_name: str
       
    Returns
    -------
    None
        
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    engine = create_engine(db_dets, poolclass=NullPool)
    con_var = engine.connect()
    df.to_sql(table_name, con=engine)
    con_var.close
