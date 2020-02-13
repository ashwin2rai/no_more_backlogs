# -*- coding: utf-8 -*-
from .utils import get_web_content
from .utils import create_datadir_link

import pandas as pd
import numpy as np
import pickle
import re
import time
import statsmodels.api as sm

class PricingAndRevs:
    
    def __init__(self, game_list):
        self.game_df = game_list
        self.game_price_dict = None
        self.pricehist_reg = r'\"x\"\:\s\"(\d{4}-\d{2}-\d{2})\"\,\s\"y\"\:\s(\d+\.\d+)'
        self.review_xpath = '/html/body/div[1]/div[2]/div[3]/div[1]/div[1]/div/div/div/div/a/text()'
        self.pic_xpath = '/html/body/div[1]/div[2]/div[3]/div[1]/div[1]/div/div/img/@data-src'
        self.pshist_link = 'https://psprices.com/region-us/search/?q={}&platform=PS4&dlc=hide'
        
    def get_all_price_histsandrev(self, df = None, col_name = 'TitlesHTML', 
                                  sleep_sec = 1, 
                                  pshist_link = None,
                                  pricehist_reg = None,
                                  review_xpath = None,
                                  pic_xpath = None):
        """
        Get price histories (as a DataFrame) and reviews from PSprices.com for a dataframe that includes a list of game search titles
        
        Parameters
        ----------
        df: Pandas.DataFrame
            DataFrame that contains a list of HTML ready search keywords for certain games
        
        col_name: str, optional
            The name of the column that contains HTML ready search keywords for certain games
            Default 'TitlesHTML'
            
        sleep_sec: int, optional
            Amount of time in seconds to sleep between request calls from website
            Default 1
            
        pshist_link: str, optional
            A website link with a placeholder to fit keywords and then search on the website
            Default 'https://psprices.com/region-us/search/?q={}&platform=PS4&dlc=hide'
        
        Returns
        -------
        Dict
            game_name:{pricehistory-date dataframe} as key:value pairs.
        
        Pandas.DataFrame (Shadow)
            Additional columns added to df 
        
        Notes
        -----
        Make a copy of dataframe df if you do not want the original to be modified.
        """    
        if not pshist_link:
            pshist_link = self.pshist_link
        if not review_xpath:
            review_xpath = self.review_xpath
        if not pic_xpath:
            pic_xpath = self.pic_xpath
        if not pricehist_reg:
            pricehist_reg = self.pricehist_reg
        if not df:
            df = self.game_df
        
        game_price_dict ={}
        link_obt = []
        rev_obt = []
        game_ph_obt = []
        links = df[col_name].apply(lambda x: pshist_link.format(x))
        df['SearchLinks'] = links
        
        for title, link in zip(df['Titles'], links):
            (game_price_dict[title], link_tf, rev_val, game_ph) = self._get_price_historyandrev_psprices(link, game_name=title, reg_string = pricehist_reg, rev_xpath = review_xpath, pic_xpath = pic_xpath)
            link_obt.append(link_tf)
            rev_obt.append(rev_val)
            game_ph_obt.append(game_ph)
            print(f"Price history for game: {title} was found: {link_tf}")
            time.sleep(sleep_sec)
        
        df['HistScrapped'] = link_obt
        df['Reviews'] = rev_obt
        df['GameCard'] = game_ph_obt
        df['GameCard'] = df['GameCard'].fillna('https://store.playstation.com/store/api/chihiro/00_09_000/container/US/en/19/UP4478-PCSE00649_00-2015072013150003/image?w=240&h=240&bg_color=000000&opacity=100&_version=00_09_000')
        
        self.game_df = df
        self.game_price_dict = game_price_dict
        return self
        

    def _get_price_historyandrev_psprices(self, addr, reg_string, 
                                          rev_xpath, pic_xpath, game_name = None):
        """
        Get price histories (as a DataFrame) and reviews from PSprices.com for a certain game, given the right link.
        
        Parameters
        ----------
        addr: str
            Website link for the game
        
        game_name: str, optional
            The name of the game to look for. 
            Only needed if there's a possibility that the website link might go into the search page instead of the game page.
            Default None
            
        reg_string: str, optional
            The regex string to extract date and prices
            Default r'\"x\"\:\s\"(\d{4}-\d{2}-\d{2})\"\,\s\"y\"\:\s(\d+\.\d+)'
        
        Returns
        -------
        Tuple of (Pandas.DataFrame, Bool, Float)
            DataFrame consists of price history, dates, and other derivatives
            Bool provides an indicator if the data was successfully scraped or not
            Float is the value of review which may be NaN if it cannot be scraped
            If the price history cannot be extracted, will return (None, False, NaN)
        
        Notes
        -----
        Specifically used to extract price history and dates and reviews from Psprices.com
        
        """
        (tree, soup) = get_web_content(addr,'htmlsoup')
        pusd = 'Price, USD' in soup.text
        sres = 'search results' in soup.text
        nfnd = 'Sorry, nothing found' in soup.text
        
        if pusd:
            reg_output = self._get_reg_output(soup, reg_string)
            rev_val = self._get_review(tree, rev_xpath)
            game_ph = self._get_pic_link(tree, pic_xpath)
            return (self._convert_price_todataframe(reg_output),True, rev_val, game_ph)
        elif sres and not nfnd:
            new_addr = self._get_price_history_link(tree, game_name)
            if new_addr is not None:
                (tree_new, soup_new) = get_web_content(new_addr,'htmlsoup')
                reg_output = self._get_reg_output(soup_new,reg_string)
                rev_val = self._get_review(tree_new, rev_xpath)
                game_ph = self._get_pic_link(tree, pic_xpath)
                return (self._convert_price_todataframe(reg_output),True, rev_val, game_ph)
            else:
                return (None,False,np.nan,np.nan)
        else:
            return (None,False,np.nan,np.nan)
     
    def _get_review(self, tree, 
                   xpath_adr ='/html/body/div[1]/div[2]/div[3]/div[1]/div[1]/div/div/div/div/a/text()'):
        """
        Extract specific text 
        
        Parameters
        ----------
        reg_output: List of lists of str
        
        Returns
        -------
        Pandas.DataFrame
        
        Notes
        -----
        Specifically used to convert the regex extraction of price history and dates into a DataFrame.
        
        """
        try:
            return float(tree.xpath(xpath_adr)[0])
        except:
            return np.nan
    
    def _get_pic_link(self, tree, xpath_adr = '/html/body/div[1]/div[2]/div[3]/div[1]/div[1]/div/div/img/@data-src'):
        try:
            return tree.xpath(xpath_adr)[0]
        except:
            return np.nan 
        
    def _get_reg_output(self, soup, reg_string):
        """
        Accepts a soup object, finds the right script tag with pricing details and returns the dates and price history using the find_reg function.
        
        Parameters
        ----------
        soup: BSoup object
        
        reg_str: str
            RegEx string        
            
        Returns
        -------
        List of Lists of str, or List of str
            return type can change depending on RegEx string
        
        Notes
        -----
        This function is specifically used to extract the price and date from the pricing history HTML script tag text.
        """
        script_text = self._find_pricing_script(soup.find_all('script'))
        script_text_cut = script_text[0:script_text.find('PS+')]
        return self._find_reg(reg_string, script_text_cut) 
    
    def _find_reg(self, reg_str, content):
        """
        Extracts data using a RegEx string
        
        Parameters
        ----------
        reg_str: str
            RegEx string
            
        content: str
            text on which the regex extract needs to take place
            
        Returns
        -------
        re.findall(reg_str,content): List of Lists of str, or List of str
            return type can change depending on RegEx string
        
        Raises
        ------
        Error (str): if no text is extracted
        
        Notes
        -----
        This function is specifically used to extract the price and date from the pricing history HTML script tag text.
        """
        reg_find = re.findall(reg_str,content)
        assert reg_find is not None, "Could not extract any content, check regex string"
        return reg_find

  
    
    def _find_pricing_script(self, script_tag):
        """
        Returns the right script tag from a given html text
        
        Parameters
        ----------
        script_tag: str
            html text returned extracted from a BSoup object 
            
        Returns
        -------
        Pandas.DataFrame object
        
        Raises
        ------
        Error (str): if issues with list of list commonly from scraping issues
        Error (str): if issues with datetime conversion
        
        Notes
        -----
        This function is specifically used to convert a specific WikiTable scraped data into a dataframe.
        """
        find = False
        counter = -1
        while not find:
            counter += 1
            try:
                script_text = script_tag[counter].text
            except:
                print('Could not find script with the price history, check web address.')
                
            if 'Price, USD' in script_text:
                find = True
                
        return script_text
    
    def _convert_price_todataframe(self, reg_output):
        """
        Converts a List of lists of str into a DataFrame. First column is assumed to be a datetime col.
        Second column is assumed to be a float column. This conversion is automatically performed.
        Third column will be the cumulative number of days extracted from the datetime column.
        Fourth column is the normalized logtransform of the float column.
        
        Parameters
        ----------
        reg_output: List of lists of str
        
        Returns
        -------
        Pandas.DataFrame
        
        Notes
        -----
        Specifically used to convert the regex extraction of price history and dates into a DataFrame.
        
        """
        try:
            price_history = pd.DataFrame(reg_output, columns = ['Date','Price'])
        except:
            print('Could not convert scraped price history to dataframe. Check regex function.')
        try:
            price_history['Date'] = pd.to_datetime(price_history['Date'],infer_datetime_format=True,errors='coerce')
        except:
            print('Could not convert scraped price history dates into datetime format')
        try:
            price_history['Price'] = pd.to_numeric(price_history['Price'],errors='coerce')
        except:
            print('Could not convert scraped prices into float format.')
        
        price_history['NormLogPrice'] = self._convert_normlogprice(price_history['Price'])
        price_history['Days'] = self._convert_date_to_cumdays(price_history['Date'])
        return price_history
    
    def _convert_normlogprice(self, series):
        """
        Log transforms a Pandas series after normalizing it by dividing the series by the first element.
        
        Parameters
        ----------
        series: Pandas.Series object
        
        Returns
        -------
        Pandas.Series
        
        """
        return np.log(series.div(series[0]))
    
    def _convert_date_to_cumdays(self, series):
        """
        Converts an ordered datetime object to cumulative no. of days starting from 0.
        
        Parameters
        ----------
        series: Pandas.Series.datetime object
        
        Returns
        -------
        Pandas.Series of element type float
        
        """
    
        newseries = series.copy()
        newseries[0] = 0
        newseries.iloc[1:] = series.diff().iloc[1:].cumsum().dt.days
        return newseries.astype(float)
    
    def _get_price_history_link(self, tree, game_name,
                               search_hits  = '/html/body/div[1]/div[2]/div/div/a/span/span/text()',
                               search_links = '/html/body/div[1]/div[2]/div/div/a/@href', 
                               addon = 'https://psprices.com'):
    
        """
        Get the link to a the website of the target video game assuming the landing page is the search page.
        
        Parameters
        ----------
        tree: lxml.html object
            Object containing html that can be extracted using Xpath or CSS Locators
        
        game_name: str
            The name of the game to look for. 
    
        search_hits: str, optional
            The Xpath address to extract all the games present on the search landing page
            Default  '/html/body/div[1]/div[2]/div/div/a/span/span/text()'
            
        search_links: str, optional
            The Xpath address to extract all the links for the games present on the search landing page
            Default  '/html/body/div[1]/div[2]/div/div/a/@href'
    
        addon: str, optional
            The main website address which will be appended to the game link extracted by the scraping algorithm to create a complete web link.
            Default 'https://psprices.com'
        
        Returns
        -------
        Str
            Website link for the game. Returns None if game is not found.
        
        """
        
        punc_dict={ord('\''):None, ord(':'):None, ord('#'):None, ord('~'):None, 
                   ord('('):None, ord(')'):None, ord('—'):None, ord('/'):' ', 
               ord('&'):None, ord(';'):' ', ord('!'):None, ord(','):None, ord('?'):None, ord('.'):None}  
        
        game_name = game_name.lower().translate(punc_dict).replace('  ',' ').replace('  ',' ')
        
        game_link_dict = {key.lower().translate(punc_dict).replace('  ',' ').replace('  ',' '):value 
                          for key, value in zip(tree.xpath(search_hits),tree.xpath(search_links))}
        
        if game_name in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name]
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        elif game_name+' standard edition' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' standard edition']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        elif game_name+' gold edition' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' gold edition']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        elif game_name+' deluxe edition' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' deluxe edition']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        elif game_name+' ps4' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' ps4']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        elif game_name+' playstation 4 edition' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' playstation 4 edition']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None      
        elif game_name+' enhanced edition' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' enhanced edition']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        elif game_name+' game of the year edition' in game_link_dict.keys():
            try:
                return addon + game_link_dict[game_name+' game of the year edition']
            except:
                print(f"Did not find {game_name} in search results. Returning None")
                return None
        else:
            return None  
        
    def write_dict(self, data_path=None, fname = 'pricehist_pkl.sav'):
        if not data_path:
            fpath = create_datadir_link(filename = fname)
        else:
            fpath = create_datadir_link(data_path = data_path, filename = fname)
        try:
            pickle.dump(self.game_price_dict, open(str(fpath), 'wb'))
        except:
            print('Could not save file, check if dataframe was created or path is right.')

    def get_prepdf_withreg(self, days = None, 
                           publisher_medval = 18, publisher_lowval = 3,
                          dev_medval = 4, dev_lowval = 1, genre_map = None):
        df = self.game_df
        game_price_dict = self.game_price_dict
        
        if not genre_map:
            genre_map = create_datadir_link(filename = 'genre.csv')
             
        df['LogRegSlope'] = np.nan
        df['LogRegIntcpt'] = np.nan
        df['InitPrice'] = np.nan
    
        suc_df = df[df['Reviews'].notnull()]
        slope = []
        intercept = []
        init_price = []
        
        for title in suc_df['Titles']:
            sl_int = self._logprice_reg_params(game_price_dict[title],x_col = 'Days', 
                                         y_col = 'NormLogPrice', days = days, reg_model=False)
            slope.append(sl_int[1])
            intercept.append(sl_int[0])
            init_price.append(game_price_dict[title]['Price'][0])
    
        df['LogRegSlope'] = pd.DataFrame(slope,index=suc_df.index,columns=['LogRegSlope'])
        df['LogRegIntcpt'] = pd.DataFrame(intercept,index=suc_df.index,columns=['LogRegIntcpt'])
        df['InitPrice'] = pd.DataFrame(init_price,index=suc_df.index,columns=['InitPrice'])
        
        self._convert_genres(df, genre_map)
        self._convert_catstrength(df,'Publishers',publisher_medval,publisher_lowval)
        self._convert_catstrength(df,'Developers',dev_medval,dev_lowval)
        
        df['ReleaseDate_Agg'] = np.where(df['ReleaseDate_NA'].isna(),df['ReleaseDate_EU'],df['ReleaseDate_NA'])
        df['ReleaseDate_Agg'] = np.where(df['ReleaseDate_Agg'].isna(),df['ReleaseDate_JP'],df['ReleaseDate_Agg'])
        df['ReleaseDate_Agg'] = pd.to_datetime(df['ReleaseDate_Agg'],infer_datetime_format=True,errors='coerce')
        df['Release_Year'] = df['ReleaseDate_Agg'].dt.year
        df['Release_Month'] = df['ReleaseDate_Agg'].dt.month
        df['Release_Day'] = df['ReleaseDate_Agg'].dt.day
        
        return self
    
    def _logprice_reg_params(self, df_lim, x_col = 'Days', y_col = 'NormLogPrice', days = None, reg_model = False):
        
        if not days:
            df = df_lim
        else:
            df = df_lim[df_lim['Days']<=days].copy()
            
        reg_X = df[x_col].values
        reg_X = sm.add_constant(reg_X)
        model = sm.OLS(df[y_col].values, reg_X).fit() 
        
        if np.inf in np.abs(model.params):
            if reg_model:
                return([np.nan,np.nan],None)
            else:
                return([np.nan,np.nan])
       
        if reg_model:
            return (model.params, model)
        else:
            return model.params
        
    def _convert_genres(self, df, genre_map):
        genre_transfrm = pd.read_csv(genre_map)
        genre = {}
        for row_label,  row_array_Series in genre_transfrm.iterrows():
            genre[row_array_Series['Old']] = row_array_Series['New']
    
        df['MappedGenres'] = df['Genres'].map(genre)
        try:
            assert df['MappedGenres'].notna().all()
        except:
            print("WARNING: The Genre CSV needs to be updated. The following Genres need to be mapped:")
            print([*df['Genres'][df['MappedGenres'].isna()].values])
            
            
    def _convert_catstrength(self, df,col_name, med_val, low_val):
        val_cts = df[col_name].value_counts()
        val_dict = {}
        
        for name in val_cts[val_cts > med_val].index:
            val_dict[name] = 'High'+ col_name
            
        for name in val_cts[(val_cts <= med_val) & (val_cts > low_val)].index:
            val_dict[name] = 'Med'+ col_name
    
        for name in val_cts[val_cts <= low_val].index:
            val_dict[name] = 'Low'+ col_name
        
        new_col_name = 'Mapped{}'.format(col_name)
        
        df[new_col_name] = df[col_name].map(val_dict)

    def get_successcol(self, col='LogRegSlope',threshold = -0.0001, verbose=False):   
        self.game_df['Success'] = np.where(self.game_df[col] > threshold, 0,1)
        if verbose:
            print(self.game_df['Success'][self.game_df[col].notna()].value_counts())
