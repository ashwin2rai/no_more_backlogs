from pathlib import Path  # pathlib is seriously awesome!

data_dir_var = Path.cwd().parent/'data'

reddit_auth = Path.cwd().parent/'data'/'RedditAuth' 
#You can use create_reddit_OAuth in the utils to create the OAuth File
#Enter path + filename that points to the Reddit OAuth credentials file created using utils function create_reddit_OAuth

sql_db = Path.cwd().parent/'data' #Path to Folder with SQL authentication details


