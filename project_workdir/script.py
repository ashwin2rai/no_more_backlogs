# -*- coding: utf-8 -*-
from investigame import GetWikiGameTable
from investigame import write_tocsv
import autoclassifier


WikiTable = GetWikiGameTable()

WikiTable.get_wiki_table_list(WikiTable.html_add_0_m, WikiTable.xpath_0_m).get_wiki_table_list(WikiTable.html_add_m_z, WikiTable.xpath_m_z).get_wiki_table_df()

write_tocsv(WikiTable.game_df, fname = 'ps4_wiki_list')

