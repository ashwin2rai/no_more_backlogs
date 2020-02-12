# -*- coding: utf-8 -*-
from .config import reddit_auth
from .utils import create_datadir_link

import pandas as pd
import numpy as np
import re
import pickle
import praw
from psaw import PushshiftAPI

class GetRedditComments:
    
    def __init__(self, game_list, comment_window = 30):
        self.game_df = game_list
        self.game_df['CommentWindow'] = (self.game_df['ReleaseDate_Agg'] -  pd.to_timedelta(comment_window, unit='d'))
        indx = self.game_df['TitlesHTML'][self.game_df['TitlesHTML'].str.len() < 6].index
        self.game_df['TitlesHTML'].loc[indx] = self.game_df['TitlesHTML'].loc[indx] + '+PS4'

    def create_reddit_instance(self, reddit_auth_dict = None):
        if reddit_auth_dict is None:
            reddit_auth_dict = pickle.load(open(create_datadir_link(reddit_auth,''),'rb'))
        reddit = praw.Reddit(client_id = reddit_auth_dict['client_id'],
                    client_secret = reddit_auth_dict['API_key'],
                    password = reddit_auth_dict['password'],
                    username = reddit_auth_dict['username'],
                    user_agent=reddit_auth_dict['user_agent'])
        
        del reddit_auth_dict
        self.reddit_inst = reddit
    
    def _create_psaw_instance(self, reddit):
        return PushshiftAPI(reddit)

    def get_reddit_comments(self, max_response_cache = 500,
                            subreddit_list = ['gaming','ps4','Games','gamernews','gamedev','rpg','DnD'],
                           verbose = True):
        
        df = self.game_df
        psaw_api = self._create_psaw_instance(self.reddit_inst)
        
        df['Game_comments'] = np.nan
        for row_tuple in df.itertuples():
            cache = []
            for subreddit in subreddit_list:
            
                if len(cache) >= max_response_cache:
                    break
            
                comment_list = psaw_api.search_comments(q=row_tuple.TitlesHTML, subreddit=subreddit)
                                               #before=row_tuple.ReleaseDate_Agg.strftime('%Y-%d-%m'),
                                               #after = row_tuple.CommentWindow.strftime('%Y-%d-%m') 
            
                punc_dict = {ord('\''):None,ord('+'):' ',ord(';'):' ',ord('\"'):' ',
                         ord('/'):None, ord('('):' ', ord(')'):' ', ord('['):' ',
                         ord(']'):' ', ord(':'):' ',ord('#'):' ',ord('~'):' ',
                         ord('-'):' ',ord('—'):' ',ord('*'):' ', ord('!'):' ',ord('&'):' ',
                         ord(','):' ', ord('?'):' ', ord('.'):' ', ord('™'):' ',ord('®'):' '}  
            
                for comment in comment_list:
                    cache.append(re.sub(r"http.+\s|http.+$"," ",comment.body.replace('\n','').replace('\xa0',' ').replace('\t',' ').translate(punc_dict)))
                    if len(cache) >= max_response_cache:
                        break
                        
            game_comment =[]
            game_comment.append(" ".join(cache))
            df.loc[row_tuple.Index,'Game_comments'] = game_comment
            if verbose:
                print("Game: {}, comment length: {}".format(row_tuple.TitlesHTML,len(game_comment[0])))