# -*- coding: utf-8 -*-

# Load important classes and functions
from investigame import GetWikiGameTable
from investigame import PricingAndRevs
from investigame import GetRedditComments
from investigame import write_tocsv
from investigame import create_posgresurl
from investigame import create_postgres_authdict
from investigame import sql_pd_write
from investigame import complete_gamedb
from investigame import create_datadir_link

from autoclassifier import AutoClassifier

# Get a list of all PS4 games from Wiki
WikiTable = GetWikiGameTable()
WikiTable.get_wiki_table_list(WikiTable.html_add_0_m, WikiTable.xpath_0_m). \
    get_wiki_table_list(WikiTable.html_add_m_z, WikiTable.xpath_m_z).get_wiki_table_df()
write_tocsv(WikiTable.game_df, fname='ps4_wiki_list')

# Get the pricing histories, reviews, and game cover image details for each game in the game table
game_dets = PricingAndRevs(WikiTable.game_df)
game_dets.get_all_price_histsandrev().write_dict()
game_dets.get_prepdf_withreg(days=None, publisher_medval=18, publisher_lowval=3, dev_medval=4,
                             dev_lowval=1).get_successcol(threshold=-0.0001, verbose=True)
write_tocsv(game_dets.game_df, fname='game_list_notext.csv')

# Get comments for each game from Reddit
game_dets_withcomments = GetRedditComments(game_dets.game_df.drop(['ReleaseDate_NA', 'ReleaseDate_JP',
                                                                   'ReleaseDate_EU'], axis=1))
game_dets_withcomments.create_reddit_instance().get_reddit_comments()
write_tocsv(game_dets_withcomments.game_df, fname='game_list_withtext.csv')

# Select relevant columns as features and preprocess the files
game_pred = AutoClassifier(game_dets_withcomments.game_df[['Reviews', 'InitPrice',
                                                           'MappedGenres', 'MappedPublishers', 'MappedDevelopers',
                                                           'Release_Year', 'Release_Month', 'Release_Day',
                                                           'Game_comments', 'Success']], text_col='Game_comments')
game_pred.preprocess_block()

# Fit several binary classifier models
game_pred.shallow_model_fit()

# game_df = pd.read_csv(create_datadir_link(filename='game_list_withtext.csv'), index_col = 0)
game_df = game_dets_withcomments.game_df

# Run the preprocesser on the dataframe again.
# This process is unnecessary but it demonstrates how users would run the model
# for additional data they may collect over time.
game_success = AutoClassifier(game_df[['Reviews', 'InitPrice',
                                       'MappedGenres', 'MappedPublishers', 'MappedDevelopers',
                                       'Release_Year', 'Release_Month', 'Release_Day',
                                       'Game_comments', 'Success']], text_col='Game_comments')
feature_set = game_success.preprocess_block(load_preproc='preproc.sav')

# Load the previously trained model and create predictions for each game and save them
succes_prob = game_success.load_model(clf='gradbooststep').predict_proba(feature_set)

# Create interpretable signals for predictions which will be used in the Web-App outputs
complete_gamedb(game_df, succes_prob)

# Finally save the database to a CSV file or push it straight to the SQL Database
write_tocsv(game_df, fname='game_list_withpreds.csv')
# create_postgres_authdict(user='username', password='3124hash', hostandport='51.22.22.22:5432', dbname ='gamedb')
# sql_pd_write(game_df, create_posgresurl(), 'investigamepred')
