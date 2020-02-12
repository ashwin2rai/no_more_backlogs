from .config import data_dir_var 

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
        print('Webpage inaccessible, please correct webpage address')
        
    if ret == 'html':
        return html.fromstring(pageContent.content)
    elif ret == 'soup':
        return BeautifulSoup(pageContent.text, 'html.parser')
    elif ret == 'htmlsoup':
        return (html.fromstring(pageContent.content),BeautifulSoup(pageContent.text, 'html.parser'))
    else:
        print('Error: ret tag is invalid in get_web_content')
        return None
    
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

    pickle.dump(reddit_auth, open(filename, 'wb'))
    del reddit_auth

def write_tocsv(df, data_path=None, fname = 'file.csv'):
    if not data_path:
        fpath = create_datadir_link(filename = fname)
    else:
        fpath = create_datadir_link(data_path = data_path, filename = fname)
    try:
        df.to_csv(fpath)
    except:
        print('Could not save file, check if dataframe was created properly or path is right.')
