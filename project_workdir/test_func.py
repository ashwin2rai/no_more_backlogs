#Very basic unit testing for investigame package, run using pytest
# > pytest

import numpy as np

from investigame import GetWikiGameTable
from investigame import PricingAndRevs
from investigame import GetRedditComments

def test_table():
    WikiTable = GetWikiGameTable()
    df = WikiTable.get_wiki_table_list(WikiTable.html_add_0_m, WikiTable.xpath_0_m).get_wiki_table_list(WikiTable.html_add_m_z, WikiTable.xpath_m_z).get_wiki_table_df().game_df
    
    #Test if wiki scraping was successful
    assert df.shape[0] > 2000, "Wiki table was not scraped properly, not enough rows"
    assert df.shape[1] >= 8, "Wiki table was not scraped properly, not enough columns"
    for col in ['Titles', 'Genres', 'Developers', 'Publishers']:
        assert col in df.columns, "Wiki table was not scraped properly, relevant columns not present"
    
    game_dets =  PricingAndRevs(WikiTable.game_df.iloc[np.random.randint(df.shape[0],size=10),:])
    game_dets.get_all_price_histsandrev()
    game_dets.get_prepdf_withreg(days = None, publisher_medval = 18, publisher_lowval = 3, dev_medval = 4, dev_lowval = 1).get_successcol(threshold = -0.0001,verbose=True)
    df = game_dets.game_df
    
    #Test if a small subset of games are being scraped sucessfully
    assert df.shape[1] >= 23, "Price scraping did not complete sucessfully, not enough columns present"
    assert df.Success.dtype.name == 'int32', "Column success was not generated sucessfully"
    
    game_dets_withcomments = GetRedditComments(game_dets.game_df.drop(['ReleaseDate_NA','ReleaseDate_JP',
       'ReleaseDate_EU'],axis=1))
    game_dets_withcomments.create_reddit_instance().get_reddit_comments()
    df = game_dets_withcomments.game_df
    
    #Test if game comments are being pulled from Reddit
    assert df.shape[0] > 0, "Comments are not being pulled from Reddit, check Reddit API instance"
    assert df['Game_comments'].str.len().sum() > 1, "No comments are being pulled from Reddit"
    