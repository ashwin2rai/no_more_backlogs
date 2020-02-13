# -*- coding: utf-8 -*-
from investigame import GetWikiGameTable
from investigame import PricingAndRevs
from investigame import GetRedditComments
from investigame import write_tocsv

from autoclassifier import AutoClassifier


WikiTable = GetWikiGameTable()

WikiTable.get_wiki_table_list(WikiTable.html_add_0_m, WikiTable.xpath_0_m).get_wiki_table_list(WikiTable.html_add_m_z, WikiTable.xpath_m_z).get_wiki_table_df()

write_tocsv(WikiTable.game_df, fname = 'ps4_wiki_list')

game_dets =  PricingAndRevs(WikiTable.game_df)

game_dets.get_all_price_histsandrev().write_dict()

game_dets.get_prepdf_withreg(days = None, publisher_medval = 18, publisher_lowval = 3, dev_medval = 4, dev_lowval = 1).get_successcol(threshold = -0.0001,verbose=True)

write_tocsv(game_dets.game_df, fname = 'game_list_notext')

game_dets_withcomments = GetRedditComments(game_dets.game_df.drop(['ReleaseDate_NA','ReleaseDate_JP',
       'ReleaseDate_EU'],axis=1))

game_dets_withcomments.create_reddit_instance().get_reddit_comments()

write_tocsv(game_dets_withcomments.game_df, fname = 'game_list_withtext')

game_pred = AutoClassifier(game_dets_withcomments.game_df[['Reviews', 'InitPrice',
       'MappedGenres', 'MappedPublishers', 'MappedDevelopers',
                    'Release_Year', 'Release_Month', 'Release_Day',
       'Game_comments','Success']])

game_pred.preprocess_block()
game_pred.shallow_model_fit()