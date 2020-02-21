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
    """
    Class GetWikiGameTable: Used to pull game related comments using the
    Reddit API (PRAW) and the PushShift.io API (PSAW)

    Initialization Parameters
    ----------
    game_list: Pandas.DataFrame
        List of PS4 Games ideally obtained from GetWikiGameTable.
        
    comment_window: int, optional
        The windown of days before release date that comments need to be searched for.
    """

    def __init__(self, game_list, comment_window=30):
        self.game_df = game_list
        self.game_df['CommentWindow'] = (self.game_df['ReleaseDate_Agg'] - pd.to_timedelta(comment_window, unit='d'))
        indx = self.game_df['TitlesHTML'][self.game_df['TitlesHTML'].str.len() < 6].index
        self.game_df['TitlesHTML'].loc[indx] = self.game_df['TitlesHTML'].loc[indx] + '+PS4'

    def create_reddit_instance(self, reddit_auth_dict=None):
        """
        Create a Reddit API (PRAW) instance using a Reddit OAuth file 
        
        Parameters
        ----------
        reddit_auth_dict: dict, optional
            This dictionary will contain Reddit API details for creating a PRAW instance.
            If None, then the Reddit OAuth pickled file will be loaded. 
            User can create this file using the create_reddit_OAuth funciton in utils. User will have configure the path to the file in config to point to the right file. By default it is ../data/
            Default None
            
        Returns
        -------
        self
            Reddit PRAW instance will be contained in self.reddit_inst
        
        Raises
        ------
        IOError: if file cannot be read

        """
        if reddit_auth_dict is None:
            try:
                reddit_auth_dict = pickle.load(open(create_datadir_link(reddit_auth, ''), 'rb'))
            except:
                raise IOError('ERROR: Could not load Reddit authentication file. Please check filename or path.')
        reddit = praw.Reddit(client_id=reddit_auth_dict['client_id'],
                             client_secret=reddit_auth_dict['API_key'],
                             password=reddit_auth_dict['password'],
                             username=reddit_auth_dict['username'],
                             user_agent=reddit_auth_dict['user_agent'])
        self.reddit_inst = reddit
        return self

    def _create_psaw_instance(self, reddit):
        """
        Create a PushShift PSAW instance using a Reddit instance 
        
        Parameters
        ----------
        reddit: PRAW instamce
            
        Returns
        -------
        PSAW instance object
        
        """
        return PushshiftAPI(reddit)

    def get_reddit_comments(self, max_response_cache=500,
                            subreddit_list=None,
                            verbose=True):
        """
        Pulls comments for games using PSAW API, where the games appear in the game list DataFrame
        passed during instaniciating.
        
        Parameters
        ----------
        max_response_cache: int, optional
            Maximum number of comments pulled per game
            Default 500
        
        subreddit_list: list, optional
            List of subreddits to trawl for comments
            Default ['gaming','ps4','Games','gamernews','gamedev','rpg','DnD']

        verbose: bool, optional
            Prints a message summarizing Game name and comment text pulled
            Default True
            
        Returns
        -------
        None
            Game comments are updated in a column called 'Game_Comments' in the game list DataFrame.

        """
        if not subreddit_list:
            subreddit_list = ['gaming', 'ps4', 'Games', 'gamernews', 'gamedev', 'rpg', 'DnD']

        df = self.game_df

        psaw_api = self._create_psaw_instance(self.reddit_inst)

        df = df[df['Reviews'].notna()].dropna().reset_index().drop(['index'], axis=1)
        df['Game_comments'] = np.nan

        for row_tuple in df.itertuples():
            cache = []
            for subreddit in subreddit_list:

                if len(cache) >= max_response_cache:
                    break

                comment_list = psaw_api.search_comments(q=row_tuple.TitlesHTML, subreddit=subreddit)
                # ,before=row_tuple.ReleaseDate_Agg.strftime('%Y-%d-%m'),
                # after = row_tuple.CommentWindow.strftime('%Y-%d-%m') )

                punc_dict = {ord('\''): None, ord('+'): ' ', ord(';'): ' ', ord('\"'): ' ',
                             ord('/'): None, ord('('): ' ', ord(')'): ' ', ord('['): ' ',
                             ord(']'): ' ', ord(':'): ' ', ord('#'): ' ', ord('~'): ' ',
                             ord('-'): ' ', ord('—'): ' ', ord('*'): ' ', ord('!'): ' ', ord('&'): ' ',
                             ord(','): ' ', ord('?'): ' ', ord('.'): ' ', ord('™'): ' ', ord('®'): ' '}

                for comment in comment_list:
                    cache.append(re.sub(r"http.+\s|http.+$", " ",
                                        comment.body.replace('\n', '').replace('\xa0', ' ').replace('\t',
                                                                                                    ' ').translate(
                                            punc_dict)))
                    if len(cache) >= max_response_cache:
                        break

            game_comment = []
            game_comment.append(" ".join(cache))
            df.loc[row_tuple.Index, 'Game_comments'] = game_comment
            if verbose:
                print("Game: {}, comment length: {}".format(row_tuple.TitlesHTML, len(game_comment[0])))
        self.game_df = df
